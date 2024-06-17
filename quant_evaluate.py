import glob
import os
import pathlib
import time

import click
import numpy as np
import onnxruntime as ort
import torchaudio
import tqdm
import yaml


def load_config_from_yaml(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def batch_test(onnx_path, batch_data, time_scale, ap_threshold=0.4, ap_dur=0.08):
    session = ort.InferenceSession(onnx_path)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    total_time = 0
    results = []

    for data in tqdm.tqdm(batch_data):
        start_time = time.time()
        ap_probability = session.run([output_name], {input_name: [data]})[0]
        duration = time.time() - start_time
        total_time += duration
        sxp = sigmoid(ap_probability)[0][0]

        segments = find_segments_dynamic(sxp, time_scale, threshold=ap_threshold,
                                         ap_threshold=int(ap_dur / time_scale))
        results.append(segments)

    return results, total_time


def pad_signal(y, padding, pad_mode="constant"):
    if pad_mode == "constant":
        left_pad, right_pad = padding
        y_padded = np.pad(y, (left_pad, right_pad), mode='constant')
        return y_padded
    else:
        raise ValueError("Unsupported pad_mode")


def unfold_signal(y, frame_length, hop_length):
    num_frames = (len(y) - frame_length) // hop_length + 1
    unfolded = []
    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        unfolded.append(y[start:end])
    return np.array(unfolded)


def get_music_chunk(
        y,
        *,
        frame_length=2048,
        hop_length=512,
        pad_mode="constant",
):
    padding = (int((frame_length - hop_length) // 2),
               int((frame_length - hop_length + 1) // 2))

    y_padded = pad_signal(y, padding, pad_mode)
    y_f = unfold_signal(y_padded, frame_length, hop_length)

    return y_f


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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


def calculate_euclidean_distance(list1, list2):
    distances = []
    for pair1, pair2 in zip(list1, list2):
        for point1, point2 in zip(pair1, pair2):
            distance = np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
            distances.append(distance)
    return np.mean(distances)


@click.command(help='')
@click.option('--onnx_path', required=True, metavar='DIR', help='Path to the onnx')
@click.option('--wav_dir', required=True, metavar='DIR', help='Path to the wav folder')
@click.option('--ap_threshold', required=False, default=0.4, help='Respiratory probability recognition threshold')
@click.option('--ap_dur', required=False, default=0.08, help='The shortest duration of breathing, discarded below '
                                                             'this threshold, in seconds')
def infer(onnx_path, wav_dir, ap_threshold, ap_dur):
    config_file = pathlib.Path(onnx_path).with_name('config.yaml')
    onnx_quant_path = pathlib.Path(onnx_path).with_name('model_quant.onnx')
    assert os.path.exists(onnx_path), f"Onnx file does not exist: {onnx_path}"
    assert onnx_quant_path.exists(), f"Onnx file does not exist: {onnx_quant_path}"
    assert config_file.exists(), f"Config file does not exist: {config_file}"

    config = load_config_from_yaml(config_file)

    time_scale = 1.0 / (config['audio_sample_rate'] / config['hop_size'])

    wav_files = glob.glob(f'{wav_dir}/*.wav')

    mel_list = []
    for wav_path in wav_files:
        audio, sr = torchaudio.load(wav_path)
        audio = audio[0][None, :]
        if sr != config['audio_sample_rate']:
            audio = torchaudio.transforms.Resample(sr, config['audio_sample_rate'])(audio)

        mel = get_music_chunk(audio[0].numpy(), frame_length=config['spec_win'], hop_length=config['hop_size'])
        mel_list.append(mel)

    batch_res, batch_time = batch_test(onnx_path, mel_list, time_scale, ap_threshold, ap_dur)
    quant_batch_res, quant_time = batch_test(onnx_quant_path, mel_list, time_scale, ap_threshold, ap_dur)

    print('batch_time: ', batch_time, batch_res)
    print('quant_time: ', quant_time, quant_batch_res)

    similarity = calculate_euclidean_distance(batch_res, quant_batch_res)
    print(f"欧几里得距离平均值为: {similarity}")


if __name__ == '__main__':
    infer()
