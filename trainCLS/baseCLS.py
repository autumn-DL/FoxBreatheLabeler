import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as PL

# from libs.build_model import build_optimizer, build_scheduler
import torch.utils.data

from model_trainer.basic_lib.build_model import build_optimizer, build_scheduler


class BasicCLS(PL.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.opt_step = 0
        self.config = config
        self.forward_step = 0
        self.global_epoch = 0
        self.train_dataset = None
        self.val_dataset = None
        self.grad_norm=0
        self.lrs=None
    def before_opt(self):
        pass
    def sync_step(self, global_step: int, forward_step: int, global_epoch: int):
        self.opt_step = global_step
        self.forward_step = forward_step
        self.global_epoch = global_epoch

    def training_step(self, batch, batch_idx: int, ):
        raise RuntimeError("")

    def configure_optimizers(self):
        optm = build_optimizer(self, config=self.config)
        scheduler = build_scheduler(optm, config=self.config)
        if scheduler is None:
            return optm
        self.lrs=scheduler
        return {
            "optimizer": optm,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }

    def train_dataloader(self):
        prefetch_factor = self.config['train_dataloader_prefetch_factor']
        persistent_workers = True
        if self.config['num_train_dataloader_workers'] == 0:
            prefetch_factor = None
            persistent_workers = False
        return torch.utils.data.DataLoader(
            self.train_dataset,
            collate_fn=self.train_dataset.collater(),
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_train_dataloader_workers'],
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            shuffle=True
        )

    def val_dataloader(self):
        prefetch_factor = self.config['val_dataloader_prefetch_factor']
        persistent_workers = True
        if self.config['num_val_dataloader_workers'] == 0:
            prefetch_factor = None
            persistent_workers = False
        return torch.utils.data.DataLoader(
            self.val_dataset,
            collate_fn=self.val_dataset.collater(),
            batch_size=1,
            num_workers=self.config['num_val_dataloader_workers'],
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers
        )

    def validation_step(self, *args, **kwargs):
        raise RuntimeError("")
