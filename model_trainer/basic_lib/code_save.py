import datetime
import pathlib
import shutil

from model_trainer.basic_lib.config_util import backup_config


def code_saver(save_list: list, work_dir, config=None):
    time_local = str(datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    work_dir = pathlib.Path(str(work_dir))
    if config is not None:
        backup_config(workdir=work_dir, config=config, time_now=time_local)

    code_dir = work_dir / 'codes' / time_local
    code_dir.mkdir(exist_ok=True, parents=True)
    for c in save_list:
        c = pathlib.Path(c)
        if c.is_file():
            shutil.copy(c, code_dir / c, )

        if c.is_dir():
            shutil.copytree(c, code_dir / c, dirs_exist_ok=True)
    print(f'| Copied codes to {code_dir}.')
