import os.path
import pathlib
from typing import Union

import click
import onnxsim
import torch
import yaml
from onnxruntime.quantization import quantize_dynamic, QuantType

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
@click.command(help='')
@click.option('--ckpt_path', required=True, metavar='DIR', help='Path to the checkpoint')
@click.option('--onnx_path', required=True, metavar='DIR', help='Path to the onnx')
@click.option('--quantize', required=False, default=False, help='quantize')
def export(ckpt_path, onnx_path, quantize):
    assert ckpt_path is not None, "Checkpoint directory (ckpt_dir) cannot be None"

    config_file = pathlib.Path(ckpt_path).with_name('config.yaml')

    assert os.path.exists(ckpt_path), f"Checkpoint path does not exist: {ckpt_path}"
    assert config_file.exists(), f"Config file does not exist: {config_file}"

    os.makedirs(pathlib.Path(onnx_path).parent, exist_ok=True)

    output_config = pathlib.Path(onnx_path).with_name('config.yaml')
    assert not os.path.exists(onnx_path), f"Error: The file '{onnx_path}' already exists."
    assert not output_config.exists(), f"Error: The file '{output_config}' already exists."

    config = get_config(config_file)

    model = FBLCLS(config)
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu')['model'])
    model.eval()

    mel = torch.randn(1, 100, 1024)

    if torch.cuda.is_available():
        model.cuda()
        mel = mel.cuda()

    with torch.no_grad():
        torch.onnx.export(
            model,
            mel,
            onnx_path,
            input_names=['mel'],
            output_names=['ap_probability'],
            dynamic_axes={
                'mel': {0: 'batch_size', 1: 'n_samples'},
                'ap_probability': {0: 'batch_size', 1: 'n_samples'}
            },
            opset_version=17
        )
        onnx_model, check = onnxsim.simplify(onnx_path, include_subgraph=True)
        assert check, 'Simplified ONNX model could not be validated'
        print(f'Model saved to: {onnx_path}')

    if quantize:
        quantized_model_path = onnx_path.replace(".onnx", "_quant.onnx")

        quantize_dynamic(
            onnx_path,
            quantized_model_path,
            weight_type=QuantType.QUInt8
        )

        print(f'Quantized model saved to: {quantized_model_path}')
    else:
        print(f'Model saved to: {onnx_path}')

    out_config = {
        'audio_sample_rate': config['audio_sample_rate'],
        'spec_win': config['spec_win'],
        'hop_size': config['hop_size']
    }
    with open(output_config, 'w') as file:
        yaml.dump(out_config, file, default_flow_style=False, allow_unicode=True)


if __name__ == '__main__':
    export()
