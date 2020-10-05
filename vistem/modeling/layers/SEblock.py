import torch
import torch.nn as nn
import torch.nn.functional as F

from vistem.modeling.layers import Conv2d, Swish, MemoryEfficientSwish

__all__ = ['SqueezeExcitation2d']

class SqueezeExcitation2d(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        memory_efficient=True,
    ):
        super().__init__()

        self.reduce = Conv2d(
            in_channels, 
            hidden_channels,
            kernel_size=1,
            stride=1,
            bias=True,
            activation=MemoryEfficientSwish() if memory_efficient else Swish(),
        )

        self.expand = Conv2d(
            hidden_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            bias=True,
            activation=MemoryEfficientSwish() if memory_efficient else Swish(),
        )

    def forward(self, x):
        out = F.adaptive_avg_pool2d(x, 1)
        out = self.reduce(out)
        out = self.expand(out)
        out = torch.sigmoid(out) * x

        return out