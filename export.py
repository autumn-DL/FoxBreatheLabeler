import os.path
import pathlib
from typing import Union

import click
import onnx
import onnxsim
import torch
import torch.nn as nn
import yaml

from Models.CVNT import CVNT
from trainCLS.baseCLS import BasicCLS


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


class FblOnnx(BasicCLS):
    def __init__(self, config):
        super().__init__(config)
        self.model = CVNT(config, output_size=1)
        self.loss = nn.BCEWithLogitsLoss()
        self.gn = 0

        model_arg = config['model_arg']
        self.frame_length = model_arg.get('spec_win', 1024)

        self.hop_length = model_arg.get('hop_size', 882)

        self.padding = (int((self.frame_length - self.hop_length) // 2),
                        int((self.frame_length - self.hop_length + 1) // 2))

    def forward(self, x, mask=None):
        y = torch.nn.functional.pad(x, self.padding, "constant")
        batch_size, num_samples = y.shape
        num_frames = (num_samples - self.frame_length) // self.hop_length + 1
        heads = torch.arange(0, num_frames) * self.hop_length  # [N]
        offsets = torch.arange(0, self.frame_length)  # [L]
        indices = offsets[None, :] + heads[:, None]  # [N, L]
        y_f = y[:, indices]
        p = torch.sigmoid(self.model(y_f, mask=mask)).squeeze(1)
        return p


@torch.no_grad()
@click.command(help='')
@click.option('--ckpt_path', required=True, metavar='DIR', help='Path to the checkpoint')
@click.option('--onnx_path', required=True, metavar='DIR', help='Path to the onnx')
def export(ckpt_path, onnx_path):
    assert ckpt_path is not None, "Checkpoint directory (ckpt_dir) cannot be None"

    config_file = pathlib.Path(ckpt_path).with_name('config.yaml')

    assert os.path.exists(ckpt_path), f"Checkpoint path does not exist: {ckpt_path}"
    assert config_file.exists(), f"Config file does not exist: {config_file}"

    os.makedirs(pathlib.Path(onnx_path).parent, exist_ok=True)

    output_config = pathlib.Path(onnx_path).with_name('config.yaml')
    assert not os.path.exists(onnx_path), f"Error: The file '{onnx_path}' already exists."
    assert not output_config.exists(), f"Error: The file '{output_config}' already exists."

    config = get_config(config_file)

    model = FblOnnx(config)
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu')['model'])
    model.eval()

    waveform = torch.randn((3, 44100), dtype=torch.float32)

    if torch.cuda.is_available():
        model.cuda()
        waveform = waveform.cuda()

    with torch.no_grad():
        torch.onnx.export(
            model,
            waveform,
            onnx_path,
            input_names=['waveform'],
            output_names=['ap_probability'],
            dynamic_axes={
                'waveform': {0: 'batch_size', 1: 'n_samples'},
                'ap_probability': {0: 'batch_size', 1: 'n_samples'}
            },
            opset_version=17
        )
        onnx_model, check = onnxsim.simplify(onnx_path, include_subgraph=True)
        assert check, 'Simplified ONNX model could not be validated'
        onnx.save(onnx_model, onnx_path)
        print(f'Model saved to: {onnx_path}')

    out_config = {
        'audio_sample_rate': config['audio_sample_rate'],
        'hop_size': config['hop_size']
    }
    with open(output_config, 'w') as file:
        yaml.dump(out_config, file, default_flow_style=False, allow_unicode=True)


if __name__ == '__main__':
    export()
