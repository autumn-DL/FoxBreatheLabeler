import glob
import os
import pathlib
from typing import Union

import click
import torch
import torchaudio
import yaml
from tqdm import tqdm

from trainCLS.FVFBLCLS import FBLCLS

print("PLEASE check the phoneme SP or pau in the HTK lab file frist!")
print("If phoneme SP or pau is not in the HTK lab file, please add it to the HTK lab file!")
print("Or you will get the HTK label file with no SP and AP's!")

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
    padding = (int((frame_length - hop_length) // 2),
               int((frame_length - hop_length + 1) // 2))

    y = torch.nn.functional.pad(y, padding, pad_mode)
    y_f = y.unfold(0, frame_length, hop_length)

    return y_f


def find_segments_dynamic(arr, time_scale, threshold=0.5, max_gap=5, ap_threshold=10):
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

    if start is not None and (len(arr) - start) >= ap_threshold:
        segments.append((start * time_scale, (len(arr) - 1) * time_scale))

    return segments


def read_lab_file(lab_file):
    with open(lab_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    segments = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 3:
            raise ValueError(f"Invalid lab file format in {lab_file}")
        start = int(parts[0]) / 1e7  # convert from 100ns to seconds
        end = int(parts[1]) / 1e7  # convert from 100ns to seconds
        phoneme = parts[2]
        segments.append((start, end, phoneme))
    return segments


def write_lab_file(segments, lab_file):
    with open(lab_file, 'w', encoding='utf-8') as file:
        for start, end, phoneme in segments:
            file.write(f"{int(start * 1e7)} {int(end * 1e7)} {phoneme}\n")


def find_overlapping_segments(start, end, segments, sp_dur):
    overlapping_segments = []
    merged_segments = []

    for segment in segments:
        segment_start, segment_end = segment
        if not (segment_end <= start or segment_start >= end):
            overlapping_segments.append(segment)

    if not overlapping_segments:
        return overlapping_segments

    current_start, current_end = overlapping_segments[0]

    for i in range(1, len(overlapping_segments)):
        next_start, next_end = overlapping_segments[i]

        if next_start - current_end < sp_dur:
            current_end = max(current_end, next_end)
        else:
            merged_segments.append((current_start, current_end))
            current_start, current_end = next_start, next_end

    merged_segments.append((current_start, current_end))

    return merged_segments


@torch.no_grad()
@click.command(help='')
@click.option('--ckpt_path', required=True, metavar='DIR', help='Path to the checkpoint')
@click.option('--wav_dir', required=True, metavar='DIR', help='Wav files')
@click.option('--lab_dir', required=True, metavar='DIR', help='Lab files')
@click.option('--lab_out_dir', required=True, metavar='DIR', help='Lab output dir')
@click.option('--ap_threshold', required=False, default=0.4, help='Respiratory probability recognition threshold')
@click.option('--ap_dur', required=False, default=0.08, help='The shortest duration of breathing, discarded below '
                                                             'this threshold, in seconds')
@click.option('--sp_dur', required=False, default=0.1, help='SP fragments below this threshold will adsorb to '
                                                            'adjacent AP, in seconds')
def export(ckpt_path, wav_dir, lab_dir, lab_out_dir, ap_threshold, ap_dur, sp_dur):
    assert ckpt_path is not None, "Checkpoint directory (ckpt_dir) cannot be None"
    assert wav_dir is not None, "WAV directory (wav_dir) cannot be None"
    assert lab_dir is not None, "Lab directory (lab_dir) cannot be None"
    assert lab_out_dir is not None, "Lab output directory (lab_out_dir) cannot be None"
    assert lab_dir != lab_out_dir, ("Lab directory (lab_dir) and Lab output directory (lab_out_dir) cannot be "
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

    lab_files = glob.glob(f'{lab_dir}/*.lab')
    for lab_file in tqdm(lab_files):
        filename = os.path.basename(lab_file)
        filename, _ = os.path.splitext(filename)
        wav_path = os.path.join(wav_dir, filename + '.wav')
        out_lab_path = os.path.join(lab_out_dir, filename + '.lab')
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

            lab_segments = read_lab_file(lab_file)

            out_segments = []
            for start, end, phoneme in lab_segments:
                if phoneme == "SP" or "pau":
                    overlapping_segments = find_overlapping_segments(start, end, segments, sp_dur)
                    if len(overlapping_segments) == 0:
                        out_segments.append((start, end, phoneme))
                    elif len(overlapping_segments) == 1:
                        ap_start, ap_end = overlapping_segments[0]
                        if start + sp_dur <= ap_start < end:
                            out_segments.append((start, ap_start, "SP"))
                            out_segments.append((ap_start, ap_end, "AP"))
                            if ap_end < end:
                                out_segments.append((ap_end, end, "SP"))
                        else:
                            out_segments.append((start, end, "AP"))
                    else:
                        cursor = start
                        for i in range(len(overlapping_segments)):
                            ap_start, ap_end = overlapping_segments[i]
                            if ap_start > cursor:
                                out_segments.append((cursor, ap_start, "SP"))
                                cursor = ap_start
                            out_segments.append((ap_start, ap_end, "AP"))
                            cursor = ap_end
                        if cursor < end:
                            out_segments.append((cursor, end, "SP"))
                else:
                    out_segments.append((start, end, phoneme))

            write_lab_file(out_segments, out_lab_path)
        else:
            print(f"Missing wav file: {wav_path}")


if __name__ == '__main__':
    export()
