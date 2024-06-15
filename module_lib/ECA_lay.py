
import torch
from torch import nn
from torch.nn.parameter import Parameter

class eca_layer1D(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self,  k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        '''

        :param x: B T C
        :return: B T C
        '''



        # feature descriptor on the global spatial information
        y = self.avg_pool(x.transpose(1, 2))

        # Two different branches of ECA module
        y = self.conv(y.transpose(1, 2))

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y


class eca_layer2D(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self,  k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        '''

        :param x: B C H W
        :return: B C H W
        '''
        # x: input features with shape [b, c, h, w]
        # b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

if __name__ == "__main__":
    eca=eca_layer1D()
    op=eca(torch.randn(5,10,80))

    eca2=eca_layer2D()
    op2=eca2(torch.randn(5,2,10,10))
    pass