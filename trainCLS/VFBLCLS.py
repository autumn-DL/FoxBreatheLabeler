import torch
import torch.nn as nn

from Datasets.FBLCD import BL_DataSet
from Models.CVNT import CVNT
from Models.model_ada import MixModel
from libs.c_grad_norm import clip_grad_value_s
from trainCLS.baseCLS import BasicCLS
from libs.plot import spec_probs_to_figure

class FBLCLS(BasicCLS):
    def __init__(self, config):
        super().__init__(config)
        self.model=CVNT(config,output_size=1)
        self.loss=nn.BCEWithLogitsLoss()

        self.train_dataset = BL_DataSet(config=config,infer=False)
        self.val_dataset = BL_DataSet(config=config,infer=True)


        self.gn = 0

    def before_opt(self):
        pass

        if self.opt_step % 50 == 0:
            self.gn = clip_grad_value_s(self.parameters(), )


    def forward(self, x, mask=None):
        return self.model(x, mask=mask)

    def training_step(self, batch, batch_idx: int):
        spec, masks,loss_weight,target = batch['spec'], batch['mask'],batch['loss_weight'],batch['target']
        if masks is not None:
            masks = masks == 0

        logits = self(spec, masks)
        loss = self.loss(logits.squeeze(1), target)
        # loss = (loss * loss_weight).mean()

        lr = self.lrs.get_last_lr()[0]
        if self.opt_step % 50 == 0:
            tb_log = {}
            tb_log['training/loss'] = loss
            # tb_log['training/accuracy_train'] = accuracy_train
            tb_log['grad_norm/all'] = self.gn

            tb_log['training/lr'] = lr
            self.logger.log_metrics(tb_log, step=self.opt_step)
        logss = {'Tloss': loss, 'lr': lr}
        # logss.update(L)
        return {'loss': loss, 'logges': logss}

    def on_training_start(self):
        self.lrs.set_step(self.opt_step)


    def on_sum_validation_logs(self, logs):
        # self.lrs.set_step(self.opt_step)
        # tb_log = {'val/loss': logs['val_loss']}
        tb_log = {}
        for i in logs:
            tb_log[f'val/{i}'] = logs[i]
        self.logger.log_metrics(tb_log, step=self.opt_step)
    def validation_step(self, batch, batch_idx: int):
        spec, masks, loss_weight, target,mel = batch['spec'], batch['mask'], batch['loss_weight'], batch['target'],batch['mel']
        logits = self(spec, masks)
        loss = self.loss(logits.squeeze(1), target)
        # loss = (loss * loss_weight).mean()
        # g=loss * loss_weight
        logits=logits.transpose(1,2)
        logits=logits.sigmoid()
        P=(logits>self.config['prob_breathe']).long()[0].cpu().squeeze(-1)
        fig = spec_probs_to_figure(spec=batch['mel'][0].cpu(), prob_gt=target.long()[0].cpu(),
                                   prob_pred=P, prob_predP=logits[0].cpu().squeeze(-1), linewidth=2)
        self.logger.experiment.add_figure(f'spec_{batch_idx}', fig, global_step=self.opt_step)

        return {'vloss': loss}



