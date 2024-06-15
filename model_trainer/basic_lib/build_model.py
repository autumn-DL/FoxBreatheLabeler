import torch

from model_trainer.basic_lib.obj_loader import build_object_from_class_name


def build_lr_scheduler_from_config(optimizer, scheduler_args):
    try:
        # PyTorch 2.0+
        from torch.optim.lr_scheduler import LRScheduler as LRScheduler
    except ImportError:
        # PyTorch 1.X
        from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

    def helper(params):
        if isinstance(params, list):
            return [helper(s) for s in params]
        elif isinstance(params, dict):
            resolved = {k: helper(v) for k, v in params.items()}
            if 'cls' in resolved:
                if (
                        resolved["cls"] == "torch.optim.lr_scheduler.ChainedScheduler"
                        and scheduler_args["scheduler_cls"] == "torch.optim.lr_scheduler.SequentialLR"
                ):
                    raise ValueError(f"ChainedScheduler cannot be part of a SequentialLR.")
                resolved['optimizer'] = optimizer
                obj = build_object_from_class_name(
                    resolved['cls'],
                    LRScheduler,
                    **resolved
                )
                return obj
            return resolved
        else:
            return params

    resolved = helper(scheduler_args)
    resolved['optimizer'] = optimizer
    return build_object_from_class_name(
        scheduler_args['scheduler_cls'],
        LRScheduler,
        False,
        **resolved
    )


def build_scheduler(optimizer, config):
    scheduler_args = config['lr_scheduler_args']
    assert scheduler_args['scheduler_cls'] != ''
    scheduler = build_lr_scheduler_from_config(optimizer, scheduler_args)
    return scheduler


def build_optimizer(model, config):
    optimizer_args = config['optimizer_args']
    assert optimizer_args['optimizer_cls'] != ''
    if 'beta1' in optimizer_args and 'beta2' in optimizer_args and 'betas' not in optimizer_args:
        optimizer_args['betas'] = (optimizer_args['beta1'], optimizer_args['beta2'])
    optimizer = build_object_from_class_name(
        optimizer_args['optimizer_cls'],
        torch.optim.Optimizer,
        False,
        model.parameters(),
        **optimizer_args
    )
    return optimizer
