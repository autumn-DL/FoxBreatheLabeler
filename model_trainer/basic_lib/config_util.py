import datetime
import pathlib

import yaml
from typing import Optional, Union


def load_yaml(path: str) -> dict:
    with open(path, encoding='utf8') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data


def save_config(path, config: dict) -> None:
    with open(path, 'w') as f:
        yaml.dump(config, f)


def backup_config(config: dict, workdir: pathlib.Path, time_now):
    config_dir = workdir / 'config'
    config_dir.mkdir(parents=True, exist_ok=True)
    save_config(path=config_dir / (str(time_now) + '.yaml'), config=config)


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
