import torch
import torch.nn as nn
import torch.nn.functional as F

from vistem.modeling.layers import Conv2d, get_norm
from vistem.utils import weight_init

__all__ = ['BasicStem']

class BasicStem(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, norm="BN"):
        super().__init__()
        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
            norm=get_norm(norm, out_channels),
        )
        weight_init.c2_msra_fill(self.conv1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu_(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x

    @property
    def out_channels(self):
        return self.conv1.out_channels

    @property
    def stride(self):
        return 4
