import torch.nn as nn

from Models.CVNT import CVNT

from trainCLS.baseCLS import BasicCLS


class FBLCLS(BasicCLS):
    def __init__(self, config):
        super().__init__(config)
        self.model = CVNT(config, output_size=1)
        self.loss = nn.BCEWithLogitsLoss()
        self.gn = 0

    def forward(self, x, mask=None):
        return self.model(x, mask=mask)
