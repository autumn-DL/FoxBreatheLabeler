from functools import partial
from typing import Union, Any, Optional, Mapping

import torch
import lightning as PL
from lightning_utilities import apply_to_collection


def generator_output_warp(outputs: any):
    return apply_to_collection(outputs, dtype=torch.Tensor,
                               function=lambda x: x.detach())


class NormGanTrainStrategy:
    def __init__(self):
        pass

    def train_strategy(self, prent_obj,
                       generator_model: Union[PL.LightningModule, torch.nn.Module],
                       discriminator_model: Union[PL.LightningModule, torch.nn.Module],
                       generator_optimizers: torch.optim.Optimizer,
                       discriminator_optimizers: torch.optim.Optimizer,
                       generator_schedulers: Optional[
                           Mapping[str, Union[PL.fabric.utilities.types.LRScheduler, bool, str, int]]],
                       discriminator_schedulers: Optional[
                           Mapping[str, Union[PL.fabric.utilities.types.LRScheduler, bool, str, int]]],
                       batch: Any,
                       batch_idx: int):
        generator_output = generator_model.training_step(batch=batch, batch_idx=batch_idx)
        loges = {}

        def discriminator_step(prent_obj_,
                               discriminator_model_,
                               generator_output_,
                               batch_,
                               batch_idx_,
                               discriminator_optimizers_):
            discriminator_optimizers_.zero_grad()
            discriminator_fake_output = discriminator_model_.training_step(
                batch=generator_output_warp(generator_output_),
                batch_idx=batch_idx_)
            discriminator_true_output = discriminator_model_.training_step(batch=batch_, batch_idx=batch_idx_)
            discriminator_losses, discriminator_logs = discriminator_model_.model_loss.discriminator_loss_fn(
                discriminator_fake=discriminator_fake_output,
                discriminator_true=discriminator_true_output)
            loges.update(apply_to_collection(discriminator_logs, dtype=torch.Tensor,
                                             function=lambda x: x.detach().cpu().item()))
            prent_obj_.fabric.backward(discriminator_losses)
            return discriminator_losses

        discriminator_optimizers.zero_grad()
        discriminator_optimizers.step(
            partial(discriminator_step,
                    prent_obj_=prent_obj,
                    discriminator_model_=discriminator_model,
                    generator_output_=generator_output,
                    batch_=batch,
                    batch_idx_=batch_idx,
                    discriminator_optimizers_=discriminator_optimizers
                    ))
        discriminator_optimizers.zero_grad()
        prent_obj.step_scheduler(discriminator_model, discriminator_schedulers, level="step",
                                 current_value=prent_obj.get_state_step())
        prent_obj.global_step += 1

        def generator_step(prent_obj_,
                           discriminator_model_,
                           generator_optimizers_,
                           generator_output_,
                           batch_,
                           batch_idx_, ):
            generator_optimizers_.zero_grad()
            generator_discriminator_fake_output = discriminator_model_.training_step(batch=generator_output_,
                                                                                     batch_idx=batch_idx_)
            generator_discriminator_true_output = discriminator_model_.training_step(batch=batch_, batch_idx=batch_idx_)
            generator_discriminator_losses, generator_discriminator_logs = discriminator_model_.model_loss.generator_discriminator_loss_fn(
                generator_discriminator_fake=generator_discriminator_fake_output,
                generator_discriminator_true=generator_discriminator_true_output)
            loges.update(apply_to_collection(generator_discriminator_logs, dtype=torch.Tensor,
                                             function=lambda x: x.detach().cpu().item()))
            prent_obj_.fabric.backward(generator_discriminator_losses)
            return generator_discriminator_losses

        generator_optimizers.zero_grad()
        generator_optimizers.step(
            partial(generator_step,
                    prent_obj_=prent_obj,
                    discriminator_model_=discriminator_model,
                    generator_optimizers_=generator_optimizers,
                    generator_output_=generator_output,
                    batch_=batch,
                    batch_idx_=batch_idx,
                    ))
        generator_optimizers.zero_grad()
        prent_obj.step_scheduler(generator_model, generator_schedulers, level="step",
                                 current_value=prent_obj.get_state_step())
        prent_obj.global_step += 1
        prent_obj.generator_opt_step += 1
        prent_obj.train_log = loges


def abs_log(log_dict: list):
    temp_dict = {}
    for i in log_dict:
        for j in i:
            if temp_dict.get(j) is None:
                temp_dict[j] = [i[j]]
            else:
                temp_dict[j].append(i[j])
    for i in temp_dict:
        temp_dict[i] = sum(temp_dict[i]) / len(temp_dict[i])
    return temp_dict


class GradAccumGanTrainStrategy:
    def __init__(self, grad_accum_steps):
        self.grad_accum_steps = grad_accum_steps
        self.batchs = []
        pass

    def train_strategy(self, prent_obj,
                       generator_model: Union[PL.LightningModule, torch.nn.Module],
                       discriminator_model: Union[PL.LightningModule, torch.nn.Module],
                       generator_optimizers: torch.optim.Optimizer,
                       discriminator_optimizers: torch.optim.Optimizer,
                       generator_schedulers: Optional[
                           Mapping[str, Union[PL.fabric.utilities.types.LRScheduler, bool, str, int]]],
                       discriminator_schedulers: Optional[
                           Mapping[str, Union[PL.fabric.utilities.types.LRScheduler, bool, str, int]]],
                       batch: Any,
                       batch_idx: int):
        self.batchs.append([batch, batch_idx])
        loges = {}
        if prent_obj.forward_step % self.grad_accum_steps == 0 and prent_obj.forward_step != 0:

            generator_cache = []
            grad_accum_number = len(self.batchs)
            batchs_cache=self.batchs
            for i in batchs_cache:
                generator_output = generator_model.training_step(batch=i[0], batch_idx=i[1])
                generator_cache.append(generator_output)

            discriminator_optimizers.zero_grad()

            log_cache = []
            for idx,i in enumerate(batchs_cache):
                discriminator_fake_output = discriminator_model.training_step(
                    batch=generator_output_warp(generator_cache[idx]),
                    batch_idx=i[1])
                discriminator_true_output = discriminator_model.training_step(batch=i[0],
                                                                              batch_idx=i[1])
                discriminator_losses, discriminator_logs = discriminator_model.model_loss.discriminator_loss_fn(
                    discriminator_fake=discriminator_fake_output,
                    discriminator_true=discriminator_true_output)
                log_cache.append(discriminator_logs)
                prent_obj.fabric.backward(discriminator_losses / grad_accum_number)

            loges.update(abs_log(log_cache))
            # loges.update(log_cache[0])

            discriminator_optimizers.step()
            discriminator_optimizers.zero_grad()
            prent_obj.step_scheduler(discriminator_model, discriminator_schedulers, level="step",
                                     current_value=prent_obj.get_state_step())
            prent_obj.global_step += 1
            generator_optimizers.zero_grad()
            log_cache = []
            for idx,i in enumerate(batchs_cache):
                generator_discriminator_fake_output = discriminator_model.training_step(batch=generator_cache[idx],
                                                                                        batch_idx=i[1])
                generator_discriminator_true_output = discriminator_model.training_step(batch=i[0],
                                                                                        batch_idx=i[1])
                generator_discriminator_losses, generator_discriminator_logs = discriminator_model.model_loss.generator_discriminator_loss_fn(
                    generator_discriminator_fake=generator_discriminator_fake_output,
                    generator_discriminator_true=generator_discriminator_true_output)
                log_cache.append(generator_discriminator_logs)

                prent_obj.fabric.backward(generator_discriminator_losses / grad_accum_number)

            loges.update(abs_log(log_cache))
            # loges.update(log_cache[0])

            generator_optimizers.step()
            generator_optimizers.zero_grad()
            prent_obj.step_scheduler(generator_model, generator_schedulers, level="step",
                                     current_value=prent_obj.get_state_step())
            prent_obj.global_step += 1
            prent_obj.generator_opt_step += 1
            prent_obj.train_log = apply_to_collection(loges, dtype=torch.Tensor,
                                                      function=lambda x: x.detach().cpu().item())
            self.batchs=[]

        else:
            pass
