import glob
import os.path
import pathlib
import numpy as np
from typing import Union

import click
import textgrid as tg
import torch
import torchaudio
import yaml
from matplotlib import pyplot as plt
from tqdm import tqdm

from trainCLS.FVFBL15CLS import FBLCLS


def pbase_config(topc: dict, basec_list: list[str]) -> dict:
    bcfg = {}

    for i in basec_list:
        bcfgs = load_yaml(i)
        bcfg.update(bcfgs)
        bcfgsp = bcfgs.get('base_config')
        if bcfgsp is not None:
            tmpcfg = pbase_config(topc=bcfg, basec_list=bcfgsp)
            bcfg.update(tmpcfg)

    bcfg.update(topc)
    return bcfg


def load_yaml(path: str) -> dict:
    with open(path, encoding='utf8') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data


def get_config(path: Union[str, pathlib.Path]) -> dict:
    topc = load_yaml(path=path)
    basec = topc.get('base_config')
    if basec is not None:
        cfg = pbase_config(topc=topc, basec_list=basec)
    else:
        cfg = topc
    if cfg.get('base_config') is not None:
        del cfg['base_config']
    return cfg


@torch.no_grad()
def get_music_chunk(
        y,
        *,
        frame_length=2048,
        hop_length=512,
        pad_mode="constant",
):
    """
    :param y: T
    :param frame_length: int
    :param hop_length: int
    :param pad_mode:
    :return: T
    """
    # padding = (int(frame_length // 2), int(frame_length // 2))
    padding = (int((frame_length - hop_length) // 2),
               int((frame_length - hop_length + 1) // 2))

    y = torch.nn.functional.pad(y, padding, pad_mode)
    y_f = y.unfold(0, frame_length, hop_length)

    return y_f


def find_segments_from_softmax(arr, time_scale, min_segment_threshold=10, max_gap=5, class_thresholds=None, use_argmax=False):
    """
    Find segments in softmax output array for each class.

    :param arr: numpy array of softmax probabilities with shape [n_samples, n_classes]
    :param time_scale: scale to convert frame index to time
    :param min_segment_threshold: minimum segment length threshold (in frames)
    :param max_gap: maximum allowed gap within a segment
    :param class_thresholds: dictionary of thresholds for each class, e.g. {1: 0.4, 2: 0.3, 3: 0.5}
    :param use_argmax: if True, assigns each frame to the class with highest probability
    :return: dict of lists of tuples (start_time, end_time, class_index) of segments
    """
    segments_by_class = {}
    n_frames, n_classes = arr.shape

    if use_argmax:
        # Method 1: Assign each frame to the class with highest probability
        predictions = np.argmax(arr, axis=1)  # Get class with highest probability for each frame
        
        # Only consider non-zero (non-background) classes
        for class_idx in range(1, n_classes):
            segments = []
            start = None
            gap_count = 0
            
            for i in range(n_frames):
                if predictions[i] == class_idx:
                    if start is None:
                        start = i
                    gap_count = 0
                else:
                    if start is not None:
                        if gap_count < max_gap:
                            gap_count += 1
                        else:
                            end = i - gap_count - 1
                            if end >= start and (end - start) >= min_segment_threshold:
                                segments.append((start * time_scale, end * time_scale, class_idx))
                            start = None
                            gap_count = 0
            
            # Handle segment at the end of the array
            if start is not None and (n_frames - start) >= min_segment_threshold:
                segments.append((start * time_scale, (n_frames - 1) * time_scale, class_idx))
            
            segments_by_class[class_idx] = segments
    else:
        # Method 2: Use class-specific thresholds
        if class_thresholds is None:
            class_thresholds = {1: 0.4, 2: 0.3, 3: 0.4}  # Default thresholds
        
        for class_idx in range(1, n_classes):
            threshold = class_thresholds.get(class_idx, 0.4)  # Default to 0.4 if not specified
            segments = []
            start = None
            gap_count = 0
            
            for i in range(n_frames):
                if arr[i, class_idx] >= threshold:
                    if start is None:
                        start = i
                    gap_count = 0
                else:
                    if start is not None:
                        if gap_count < max_gap:
                            gap_count += 1
                        else:
                            end = i - gap_count - 1
                            if end >= start and (end - start) >= min_segment_threshold:
                                segments.append((start * time_scale, end * time_scale, class_idx))
                            start = None
                            gap_count = 0
            
            # Handle segment at the end of the array
            if start is not None and (n_frames - start) >= min_segment_threshold:
                segments.append((start * time_scale, (n_frames - 1) * time_scale, class_idx))
            
            segments_by_class[class_idx] = segments
    
    return segments_by_class


def find_overlapping_segments(start, end, segments, sp_dur):
    overlapping_segments = []
    merged_segments = []

    for segment in segments:
        segment_start, segment_end = segment
        # Check if there is any overlap
        if not (segment_end <= start or segment_start >= end):
            overlapping_segments.append(segment)

    if not overlapping_segments:
        return overlapping_segments

    # Merge adjacent segments if the gap is less than sp_dur
    current_start, current_end = overlapping_segments[0]

    for i in range(1, len(overlapping_segments)):
        next_start, next_end = overlapping_segments[i]

        if next_start - current_end < sp_dur:
            # Merge segments
            current_end = max(current_end, next_end)
        else:
            # No merge, save the current segment
            merged_segments.append((current_start, current_end))
            current_start, current_end = next_start, next_end

    # Append the last segment
    merged_segments.append((current_start, current_end))

    return merged_segments


def plot(sxp, segments, time_scale):
    x = range(len(sxp))
    x = [y * time_scale for y in x]
    y = sxp

    for start, end in segments:
        plt.axvspan(start, end, ymin=0, ymax=1, color='red', alpha=0.3)

    plt.plot(x, y)
    plt.show()


@torch.no_grad()
@click.command(help='')
@click.option('--ckpt_path', required=True, metavar='DIR', help='Path to the checkpoint')
@click.option('--wav_dir', required=True, metavar='DIR', help='Wav files')
@click.option('--tg_dir', required=True, metavar='DIR', help='Textgrid files')
@click.option('--tg_out_dir', required=True, metavar='DIR', help='Textgrid output dir')
@click.option('--ap_threshold', required=False, default=0.4, help='AP probability recognition threshold')
@click.option('--ep_threshold', required=False, default=0.3, help='EP probability recognition threshold')
@click.option('--gs_threshold', required=False, default=0.4, help='GS probability recognition threshold')
@click.option('--use_argmax', required=False, default=False, help='If True, use argmax prediction instead of thresholds')
@click.option('--ap_dur', required=False, default=0.08, help='The shortest duration of events, discarded below '
                                                             'this threshold, in seconds')
@click.option('--sp_dur', required=False, default=0.1, help='SP fragments below this threshold will adsorb to '
                                                            'adjacent non-SP segments, in seconds')
def export(ckpt_path, wav_dir, tg_dir, tg_out_dir, ap_threshold, ep_threshold, 
           gs_threshold, use_argmax, ap_dur, sp_dur):
    assert ckpt_path is not None, "Checkpoint directory (ckpt_dir) cannot be None"
    assert wav_dir is not None, "WAV directory (wav_dir) cannot be None"
    assert tg_dir is not None, "TextGrid directory (tg_dir) cannot be None"
    assert tg_out_dir is not None, "TextGrid output directory (tg_out_dir) cannot be None"
    assert tg_dir != tg_out_dir, ("TextGrid directory (tg_dir) and TextGrid output directory (tg_out_dir) cannot be "
                                  "the same")

    config_file = pathlib.Path(ckpt_path).with_name('config.yaml')

    assert os.path.exists(ckpt_path), f"Checkpoint path does not exist: {ckpt_path}"
    assert config_file.exists(), f"Config file does not exist: {config_file}"

    config = get_config(config_file)

    time_scale = 1.0 / (config['audio_sample_rate'] / config['hop_size'])

    model = FBLCLS(config)
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu')['model'])
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    tg_files = glob.glob(f'{tg_dir}/*.TextGrid')
    for tg_file in tqdm(tg_files):
        filename = os.path.basename(tg_file)
        filename, _ = os.path.splitext(filename)
        wav_path = os.path.join(wav_dir, filename + '.wav')
        out_tg_path = os.path.join(tg_out_dir, filename + '.TextGrid')
        if os.path.exists(wav_path):
            audio, sr = torchaudio.load(wav_path)
            audio = audio[0][None, :]
            if sr != config['audio_sample_rate']:
                audio = torchaudio.transforms.Resample(sr, config['audio_sample_rate'])(audio)

            mel = get_music_chunk(audio[0], frame_length=config['spec_win'], hop_length=config['hop_size']).unsqueeze(0)
            if torch.cuda.is_available():
                mel = mel.cuda()
            logits = model(mel)
            f_lg = torch.softmax(logits[0].transpose(0,1), dim=1)

            sxp = f_lg.cpu().numpy()
            token_map = {
                'None': 0,
                'AP': 1,
                'EP': 2,
                'GS': 3,
            }
            inv_token_map = {v: k for k, v in token_map.items()}
            
            # Define thresholds for each class
            class_thresholds = {
                1: ap_threshold,  # AP
                2: ep_threshold,  # EP
                3: gs_threshold   # GS
            }

            # Get segments for each class using either argmax or class-specific thresholds
            class_segments = find_segments_from_softmax(
                sxp, 
                time_scale, 
                min_segment_threshold=int(ap_dur / time_scale),
                class_thresholds=class_thresholds,
                use_argmax=use_argmax
            )
            
            # Combine all segments and sort them by start time
            all_segments = []
            for class_idx, segments in class_segments.items():
                all_segments.extend(segments)
            all_segments.sort(key=lambda x: x[0])  # Sort by start time

            textgrid = tg.TextGrid()
            textgrid.read(tg_file)
            words = textgrid[0]
            phones = textgrid[1]

            words_dict = {}
            word_cursor = 0
            for interval in words:
                if interval.minTime > word_cursor:
                    words_dict[word_cursor] = {"start": word_cursor, "end": interval.minTime,
                                              "text": "", "phones": []}
                words_dict[interval.minTime] = {"start": interval.minTime, "end": interval.maxTime,
                                              "text": interval.mark, "phones": []}
                word_cursor = interval.maxTime

            for interval in words:
                word_start = interval.minTime
                word_end = interval.maxTime

                for phone in phones:
                    if phone.minTime >= word_start and phone.maxTime <= word_end:
                        words_dict[interval.minTime]["phones"].append({"start": phone.minTime, "end": phone.maxTime,
                                                                      "text": phone.mark})

            out = []
            for k, v in words_dict.items():
                if v['text'] == "SP" or v['text'] == "":
                    sp_start = v['start']
                    sp_end = v['end']
                    
                    # Find segments that overlap with this SP/silence period
                    overlapping_segments = []
                    for segment in all_segments:
                        start, end, class_idx = segment
                        # Check if there is any overlap
                        if not (end <= sp_start or start >= sp_end):
                            overlapping_segments.append((start, end, class_idx))
                    
                    # Sort overlapping segments by start time
                    overlapping_segments.sort(key=lambda x: x[0])
                    
                    # Merge adjacent segments of the same class if the gap is less than sp_dur
                    merged_segments = []
                    if overlapping_segments:
                        current_start, current_end, current_class = overlapping_segments[0]
                        
                        for i in range(1, len(overlapping_segments)):
                            next_start, next_end, next_class = overlapping_segments[i]
                            
                            if next_start - current_end < sp_dur and current_class == next_class:
                                # Merge segments of the same class
                                current_end = max(current_end, next_end)
                            else:
                                # No merge, save the current segment
                                merged_segments.append((current_start, current_end, current_class))
                                current_start, current_end, current_class = next_start, next_end, next_class
                        
                        # Append the last segment
                        merged_segments.append((current_start, current_end, current_class))
                    
                    if not merged_segments:
                        out.append(v)  # Keep original SP if no overlap
                    else:
                        cursor = sp_start
                        
                        for i in range(len(merged_segments)):
                            seg_start, seg_end, class_idx = merged_segments[i]
                            label = inv_token_map[class_idx]  # Get label from class index
                            
                            # If there's a gap before the segment, add it as SP
                            if seg_start > cursor + sp_dur:
                                out.append({"start": cursor, "end": seg_start, "text": "SP", "phones": []})
                            
                            # Add the detected segment
                            out.append({"start": max(cursor, seg_start), "end": seg_end, 
                                       "text": label, "phones": []})
                            cursor = seg_end
                        
                        # If there's a gap after the last segment, add it as SP
                        if sp_end > cursor + sp_dur:
                            out.append({"start": cursor, "end": sp_end, "text": "SP", "phones": []})
                else:
                    out.append(v)  # Keep non-SP items unchanged

            out_tg = tg.TextGrid()
            tier_words = tg.IntervalTier(name="words")
            tier_phones = tg.IntervalTier(name="phones")

            for item in out:
                tier_words.add(minTime=item['start'], maxTime=item['end'], mark=item['text'])
                if item['text'] in ["SP", "AP", "EP", "GS"]:
                    tier_phones.add(minTime=item['start'], maxTime=item['end'], mark=item['text'])
                    continue
                for phone in item.get('phones', []):
                    tier_phones.add(minTime=phone['start'], maxTime=phone['end'], mark=phone['text'])

            out_tg.tiers.insert(0, tier_words)
            out_tg.tiers.insert(1, tier_phones)
            out_tg.write(out_tg_path)
        else:
            print(f"Miss wav file: {wav_path}")


if __name__ == '__main__':
    export()
