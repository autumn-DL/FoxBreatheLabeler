import glob
import os.path
import pathlib
from typing import Union

import click
import textgrid as tg
import torch
import torchaudio
import yaml
from matplotlib import pyplot as plt
from tqdm import tqdm

from trainCLS.FVFBLCLS import FBLCLS


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


def find_segments_dynamic(arr, time_scale, threshold=0.5, max_gap=5, ap_threshold=10):
    """
    Find segments in the array where values are mostly above a given threshold.

    :param time_scale: 1.0 / (config['audio_sample_rate'] / config['hop_size'])
    :param ap_threshold: minimum breathing time, unit: number of samples
    :param arr: numpy array of probabilities
    :param threshold: threshold value to consider a segment
    :param max_gap: maximum allowed gap of values below the threshold within a segment
    :return: list of tuples (start_index, end_index) of segments
    """
    segments = []
    start = None
    gap_count = 0

    for i in range(len(arr)):
        if arr[i] >= threshold:
            if start is None:
                start = i
            gap_count = 0
        else:
            if start is not None:
                if gap_count < max_gap:
                    gap_count += 1
                else:
                    end = i - gap_count - 1
                    if end >= start and (end - start) >= ap_threshold:
                        segments.append((start * time_scale, end * time_scale))
                    start = None
                    gap_count = 0

    # Handle the case where the array ends with a segment
    if start is not None and (len(arr) - start) >= ap_threshold:
        segments.append((start * time_scale, (len(arr) - 1) * time_scale))

    return segments


def plot(sxp, segments, time_scale):
    x = range(len(sxp))
    x = [y * time_scale for y in x]
    y = sxp

    for start, end in segments:
        plt.axvspan(start, end, ymin=0, ymax=1, color='red', alpha=0.3)

    plt.plot(x, y)
    plt.show()


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
def start():

    print(r'''          _____                    _____                    _____  
         /\    \                  /\    \                  /\    \ 
        /::\    \                /::\    \                /::\____\
       /::::\    \              /::::\    \              /:::/    /
      /::::::\    \            /::::::\    \            /:::/    / 
     /:::/\:::\    \          /:::/\:::\    \          /:::/    /  
    /:::/__\:::\    \        /:::/__\:::\    \        /:::/    /   
   /::::\   \:::\    \      /::::\   \:::\    \      /:::/    /    
  /::::::\   \:::\    \    /::::::\   \:::\    \    /:::/    /     
 /:::/\:::\   \:::\    \  /:::/\:::\   \:::\ ___\  /:::/    /      
/:::/  \:::\   \:::\____\/:::/__\:::\   \:::|    |/:::/____/       
\::/    \:::\   \::/    /\:::\   \:::\  /:::|____|\:::\    \       
 \/____/ \:::\   \/____/  \:::\   \:::\/:::/    /  \:::\    \      
          \:::\    \       \:::\   \::::::/    /    \:::\    \     
           \:::\____\       \:::\   \::::/    /      \:::\    \    
            \::/    /        \:::\  /:::/    /        \:::\    \   
             \/____/          \:::\/:::/    /          \:::\    \  
                               \::::::/    /            \:::\    \ 
                                \::::/    /              \:::\____\
                                 \::/____/                \::/    /
                                  ~~                       \/____/ 
                                                                   ''')
    print('\n')

@torch.no_grad()
@click.command(help='')
@click.option('--ckpt_path', required=True, metavar='DIR', help='Path to the checkpoint')
@click.option('--wav_dir', required=True, metavar='DIR', help='Wav files')
@click.option('--tg_dir', required=True, metavar='DIR', help='Textgrid files')
@click.option('--tg_out_dir', required=True, metavar='DIR', help='Textgrid output dir')
@click.option('--ap_threshold', required=False, default=0.4, help='Respiratory probability recognition threshold')
@click.option('--ap_dur', required=False, default=0.08, help='The shortest duration of breathing, discarded below '
                                                             'this threshold, in seconds')
@click.option('--sp_dur', required=False, default=0.1, help='SP fragments below this threshold will adsorb to '
                                                            'adjacent AP, in seconds')
def export(ckpt_path, wav_dir, tg_dir, tg_out_dir, ap_threshold, ap_dur, sp_dur):
    start()
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
        wav_path = f'{wav_dir}/{filename}.wav'
        out_tg_path = f'{tg_out_dir}/{filename}.TextGrid'
        if os.path.exists(wav_path):
            audio, sr = torchaudio.load(wav_path)
            audio = audio[0][None, :]
            if sr != config['audio_sample_rate']:
                audio = torchaudio.transforms.Resample(sr, config['audio_sample_rate'])(audio)

            mel = get_music_chunk(audio[0], frame_length=config['spec_win'], hop_length=config['hop_size']).unsqueeze(0)
            if torch.cuda.is_available():
                mel = mel.cuda()
            ap_probability = model(mel)
            ap_probability = torch.sigmoid(ap_probability)

            sxp = ap_probability.cpu().numpy()[0][0]

            segments = find_segments_dynamic(sxp, time_scale, threshold=ap_threshold,
                                             ap_threshold=int(ap_dur / time_scale))
            # plot(sxp, segments, time_scale)

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
                    overlapping_segments = find_overlapping_segments(sp_start, sp_end, segments, sp_dur)
                    if len(overlapping_segments) == 0:
                        out.append(v)
                    elif len(overlapping_segments) == 1:
                        ap_start, ap_end = overlapping_segments[0]

                        cursor = sp_start
                        if sp_start + sp_dur <= ap_start < sp_end:
                            out.append({"start": cursor, "end": ap_start,
                                        "text": "SP"})
                            cursor = ap_start

                        if ap_end <= sp_end - sp_dur:
                            out.append({"start": cursor, "end": ap_end,
                                        "text": "AP"})
                            out.append({"start": ap_end, "end": sp_end,
                                        "text": "SP"})
                        else:
                            out.append({"start": cursor, "end": sp_end,
                                        "text": "AP"})
                    else:
                        cursor = sp_start
                        for i in range(len(overlapping_segments)):
                            ap_start, ap_end = overlapping_segments[i]

                            if ap_start > cursor:
                                out.append({"start": cursor, "end": ap_start,
                                            "text": "SP"})
                            if i == 0:
                                if sp_start + sp_dur <= ap_start < sp_end:
                                    out.append({"start": cursor, "end": ap_start,
                                                "text": "SP"})
                                    cursor = ap_start
                                out.append({"start": cursor, "end": ap_end,
                                            "text": "AP"})
                                cursor = ap_end
                            elif i == len(overlapping_segments) - 1:
                                if ap_end <= sp_end - sp_dur:
                                    out.append({"start": cursor, "end": ap_end,
                                                "text": "AP"})
                                    out.append({"start": ap_end, "end": sp_end,
                                                "text": "SP"})
                                else:
                                    out.append({"start": cursor, "end": sp_end,
                                                "text": "AP"})
                            else:
                                out.append({"start": cursor, "end": ap_end,
                                            "text": "AP"})
                                cursor = ap_end
                else:
                    out.append(v)

            out_tg = tg.TextGrid()
            tier_words = tg.IntervalTier(name="words")
            tier_phones = tg.IntervalTier(name="phones")

            for item in out:
                tier_words.intervals.append(tg.Interval(item['start'], item['end'], item['text']))
                if item['text'] == "SP" or item['text'] == "AP":
                    tier_phones.intervals.append(tg.Interval(item['start'], item['end'], item['text']))
                    continue
                for phone in item['phones']:
                    tier_phones.intervals.append(tg.Interval(phone['start'], phone['end'], phone['text']))

            out_tg.tiers.insert(0, tier_words)
            out_tg.tiers.insert(1, tier_phones)
            out_tg.write(out_tg_path)
        else:
            print(f"Miss wav file: {wav_path}")


if __name__ == '__main__':
    export()
