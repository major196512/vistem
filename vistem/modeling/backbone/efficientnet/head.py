import torch
import torch.nn as nn

from vistem.modeling.layers import Conv2d, get_norm
from vistem.modeling.layers import Swish, MemoryEfficientSwish
from vistem.modeling.layers.norm.frozen_bn import FrozenBatchNorm2d

__all__ = ['HeadBlock']

class HeadBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        norm="BN",
        memory_efficient=True,
    ):
        super().__init__()

        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
            norm=get_norm(norm, out_channels),
            activation=MemoryEfficientSwish() if memory_efficient else Swish(),
        )

    def forward(self, x):
        return self.conv1(x)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        FrozenBatchNorm2d.convert_frozen_batchnorm(self)
        return self

    @property
    def out_channels(self):
        return self.conv1.out_channels

    @property
    def stride(self):
        return 1