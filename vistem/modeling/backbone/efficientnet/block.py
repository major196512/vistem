import torch
import torch.nn as nn
import torch.nn.functional as F

from vistem.modeling.layers import Conv2d, SqueezeExcitation2d, get_norm, drop_connect
from vistem.modeling.layers import Swish, MemoryEfficientSwish
from vistem.modeling.layers.norm.frozen_bn import FrozenBatchNorm2d
from vistem.utils import weight_init

__all__ = ['MBConvBlock', 'HeadBlock']

class BlockBase(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        FrozenBatchNorm2d.convert_frozen_batchnorm(self)
        return self

class MBConvBlock(BlockBase):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        *,
        stride=1,
        expand_dim=1,
        num_groups=1,
        norm="BN",
        dilation=1,
        se_ratio=0.25,
        drop_connect_prob = 0.2,
        memory_efficient=True,
        is_skip=True,
    ):
        super().__init__(in_channels, out_channels, stride)

        self.is_skip = (is_skip and stride == 1 and in_channels != out_channels)
        self.drop_connect_prob = drop_connect_prob

        if expand_dim > 1:
            self.expand_conv = Conv2d(
                in_channels,
                in_channels * expand_dim,
                kernel_size=1,
                stride=1,
                bias=False,
                norm=get_norm(norm, in_channels * expand_dim),
                activation=MemoryEfficientSwish() if memory_efficient else Swish(),
            )

        self.depthwise_conv = Conv2d(
            in_channels * expand_dim,
            in_channels * expand_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=int((kernel_size-1)/2) * dilation,
            bias=False,
            groups=num_groups,
            dilation=dilation,
            norm=get_norm(norm, in_channels * expand_dim),
            activation=MemoryEfficientSwish() if memory_efficient else Swish(),
        )

        if se_ratio > 0:
            self.SEblock = SqueezeExcitation2d(
                in_channels * expand_dim,
                int(in_channels * se_ratio),
                in_channels * expand_dim,
                memory_efficient=memory_efficient,
            )

        self.project_conv = Conv2d(
            in_channels * expand_dim,
            out_channels,
            kernel_size=1,
            stride=1,
            bias=False,
            norm=get_norm(norm, out_channels),
            activation=None,
        )

        # for layer in [self.conv1, self.conv2, self.conv3, self.shortcut]:
        #     if layer is not None: 
        #         weight_init.c2_msra_fill(layer)

        # nn.init.constant_(self.conv3.norm.weight, 0)

    def forward(self, x):
        out = self.expand_conv(x) if hasattr(self, 'expand_conv') else x
        out = self.depthwise_conv(out)
        out = self.SEblock(out) if hasattr(self, 'SEblock') else out
        out = self.project_conv(out)

        if self.is_skip:
            if self.training : out = drop_connect(out, p=self.drop_connect_prob)
            out += x

        return out

class HeadBlock(BlockBase):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        *,
        stride=1,
        norm="BN",
        memory_efficient=True,
    ):
        super().__init__(in_channels, out_channels, stride)

        self.block = Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=int((kernel_size-1)/2),
            bias=False,
            norm=get_norm(norm, out_channels),
            activation=MemoryEfficientSwish() if memory_efficient else Swish(),
        )

    def forward(self, x):
        return self.block(x)
