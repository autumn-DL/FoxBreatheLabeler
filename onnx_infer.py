import os
import pathlib

import click
import numpy as np
import onnxruntime as ort
import torchaudio
import yaml
from matplotlib import pyplot as plt


def load_config_from_yaml(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def run_inference(onnx_model_path, input_data):
    session = ort.InferenceSession(onnx_model_path)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    results = session.run([output_name], {input_name: input_data})

    return results


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


def plot(sxp, segments, time_scale, mel_spectrogram=None):
    fig, ax = plt.subplots()

    x = range(len(sxp))
    x = [y * time_scale for y in x]
    y = sxp

    mel_spectrogram = 10 * np.log10(mel_spectrogram + 1e-6)  # Convert to log scale
    mel_spectrogram = (mel_spectrogram - mel_spectrogram.min()) / (
            mel_spectrogram.max() - mel_spectrogram.min())  # Normalize
    # Plot the mel spectrogram as background
    ax.imshow(mel_spectrogram, aspect='auto', origin='lower', cmap='viridis',
              extent=[0, len(sxp) * time_scale, 0, mel_spectrogram.shape[0]])

    # Overlay segments on the mel spectrogram
    for start, end in segments:
        ax.axvspan(start, end, ymin=0, ymax=1, color='red', alpha=0.2)

    # Overlay the probability curve
    ax.plot(x, y, color='white')

    plt.show()


@click.command(help='')
@click.option('--onnx_path', required=True, metavar='DIR', help='Path to the onnx')
@click.option('--wav_path', required=True, metavar='DIR', help='Path to the wav file')
@click.option('--ap_threshold', required=False, default=0.4, help='Respiratory probability recognition threshold')
@click.option('--ap_dur', required=False, default=0.08, help='The shortest duration of breathing, discarded below '
                                                             'this threshold, in seconds')
def infer(onnx_path, wav_path, ap_threshold, ap_dur):
    config_file = pathlib.Path(onnx_path).with_name('config.yaml')
    assert os.path.exists(onnx_path), f"Onnx file does not exist: {onnx_path}"
    assert config_file.exists(), f"Config file does not exist: {config_file}"

    config = load_config_from_yaml(config_file)

    time_scale = 1.0 / (config['audio_sample_rate'] / config['hop_size'])

    audio, sr = torchaudio.load(wav_path)
    audio = audio[0][None, :]
    if sr != config['audio_sample_rate']:
        audio = torchaudio.transforms.Resample(sr, config['audio_sample_rate'])(audio)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=config['audio_sample_rate'],
        n_fft=1024,
        hop_length=256,
        n_mels=128
    )
    mel_spectrogram = mel_transform(audio).squeeze().numpy()

    mel = get_music_chunk(audio[0].numpy(), frame_length=config['spec_win'], hop_length=config['hop_size'])

    ap_probability = run_inference(onnx_path, [mel])[0]
    sxp = sigmoid(ap_probability)[0][0]

    segments = find_segments_dynamic(sxp, time_scale, threshold=ap_threshold,
                                     ap_threshold=int(ap_dur / time_scale))
    plot(sxp * 128, segments, time_scale, mel_spectrogram)


if __name__ == '__main__':
    infer()
