import torch
import torch.nn as nn
import math
import re

from .base import PLANBase
from vistem.modeling.layers import Conv2d

__all__ = ['PLANBottomUP']

class PLANBottomUP(PLANBase):
    def __init__(self, top_channels, bot_channels, out_channels, plan_cfg):
        super().__init__()

        self.out_channels = out_channels
        self.decode_cfg(plan_cfg)

        self.conv_keys = Conv2d(top_channels, self.num_heads * self.hidden_channels, 1, stride=1, bias=False)
        self.conv_queries = Conv2d(bot_channels, self.num_heads * self.hidden_channels, 1, stride=1, bias=False)
        self.conv_values = Conv2d(bot_channels, self.num_heads * self.hidden_channels, 1, stride=1, bias=False)

        self.conv_fuse = []
        prev_channels = self.num_heads * self.hidden_channels
        for _ in range(self.num_fuse):
            self.conv_fuse.append(Conv2d(prev_channels, self.out_channels, 1, stride=1, bias=False, activation=nn.ReLU()))
            prev_channels = self.out_channels
        self.conv_fuse = nn.Sequential(*self.conv_fuse)

        for layer in self.modules():
            if isinstance(layer, Conv2d):
                torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                if layer.bias is not None : torch.nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        top_feat, bot_feat = x
        B, _, H, W = top_feat.shape
        results = []

        queries = self.conv_queries(bot_feat).permute(1, 0, 2, 3).split(self.hidden_channels)
        values = self.conv_values(bot_feat).permute(1, 0, 2, 3).split(self.hidden_channels)
        keys = self.conv_keys(top_feat).permute(1, 0, 2, 3).split(self.hidden_channels)

        for head_idx in range(self.num_heads):
            query = queries[head_idx].permute(1, 0, 2, 3)
            value = values[head_idx].permute(1, 0, 2, 3)
            key = keys[head_idx].permute(1, 0, 2, 3)
            
            query = nn.functional.unfold(query, kernel_size=self.erf, padding=(self.erf-1)//2, stride=2)
            query = query.reshape(B, -1, self.erf * self.erf, H, W)
            weight = (query * key.unsqueeze(2)).sum(dim=1) / math.sqrt(self.hidden_channels)
            weight = weight.softmax(dim=1)

            value = nn.functional.unfold(value, kernel_size=self.erf, padding=(self.erf-1)//2, stride=2)
            value = value.reshape(B, -1, self.erf * self.erf, H, W)

            ret = (value * weight.unsqueeze(1)).sum(dim=2)
            results.append(ret)

        results = torch.cat(results, dim=1)
        results = self.conv_fuse(results)

        return results
