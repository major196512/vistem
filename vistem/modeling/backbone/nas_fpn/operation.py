import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List

from vistem.utils import weight_init
from vistem.modeling.layers import Conv2d, get_norm
from vistem.structures import ShapeSpec

__all__ = ['binary_operation', 'RCB']

def binary_operation(op):
    if op == 'GP' : return GP
    elif op == 'SUM' : return SUM
    elif op == 'GP_SUM' : return GP_SUM
    else : ValueError(f'Invalid NAS-FPN binary operation : {op}')

class RCB(nn.Module):
    def __init__(self, input_shape : ShapeSpec, norm='BN', use_bias=True):
        super().__init__()
        in_channel = input_shape.channels

        self.conv = Conv2d(
                in_channel,
                in_channel,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm=get_norm(norm, in_channel),
            )
        weight_init.c2_xavier_fill(self.conv)

    def forward(self, x):
        return self.conv(F.relu(x))

class GP(nn.Module):
    def forward(self, x : List[torch.Tensor], out_shape : torch.Size):
        x0, x1 = x
        if x1.shape != out_shape : x1 = F.interpolate(x1, out_shape[2:])

        global_ctx = torch.mean(x0, [2, 3], keepdim=True)
        global_ctx = torch.sigmoid(global_ctx)
        result = (global_ctx * x1) + F.interpolate(x0, out_shape[2:])

        return result

class SUM(nn.Module):
    def forward(self, x : List[torch.Tensor], out_shape : torch.Size):
        x0, x1 = x
        if x1.shape != out_shape : x1 = F.interpolate(x1, out_shape[2:])

        result = x1 + F.interpolate(x0, out_shape[2:])

        return result

class GP_SUM(nn.Module):
    def __init__(self):
        super().__init__()
        self.gp = GP()
        self.sum = SUM()

    def forward(self, x : List[torch.Tensor], out_shape : torch.Size):
        x0, x1, x2 = x
        gp = self.gp([x0, x1], x1.shape)
        result = self.sum([gp, x2], out_shape)

        return result