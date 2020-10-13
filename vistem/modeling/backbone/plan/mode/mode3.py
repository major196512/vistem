import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import re

from . import PLANDecode, PLAN_REGISTRY
from vistem.modeling.layers import Conv2d, unfold_2d, fold_2d

__all__ = ['PLAN_mode3']

@PLAN_REGISTRY.register()
class PLAN_mode3(PLANDecode):
    def __init__(self, top_feat, bot_feat, out_channels, plan_cfg):
        super().__init__()

        self.top_feat = top_feat
        self.bot_feat = bot_feat

        self.out_channels = out_channels
        self.decode_cfg(plan_cfg)

        self.top_query = Conv2d(out_channels, self.num_heads * self.key_dim, 1, stride=1, bias=False)
        self.bot_key = Conv2d(out_channels, self.num_heads * self.key_dim, 1, stride=1, bias=False)
        self.top_value = Conv2d(out_channels, self.num_heads * self.value_dim, 1, stride=1, bias=False)
        self.bot_value = Conv2d(out_channels, self.num_heads * self.value_dim, 1, stride=1, bias=False)
        self.top_fuse = Conv2d(self.num_heads * self.value_dim, out_channels, 1, stride=1, bias=False)

        for layer in self.modules():
            if isinstance(layer, Conv2d):
                torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                if layer.bias is not None : torch.nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        top_feat, bot_feat = x
        B, _, H, W = top_feat.shape
        results = []

        top_query       = self.top_query(top_feat).split(self.key_dim, dim=1)
        bot_key         = self.bot_key(bot_feat).split(self.key_dim, dim=1)
        top_value       = self.top_value(top_feat).split(self.value_dim, dim=1)
        bot_value       = self.bot_value(bot_feat).split(self.value_dim, dim=1)

        top_results = []
        for head_idx in range(self.num_heads):
            top_Q = top_query[head_idx].reshape(B, 1, -1, H, W) # (B, 1, C, H, W)
            top_V = top_value[head_idx] # (B, C, H, W)
            bot_K = unfold_2d(bot_key[head_idx], kernel_size=self.erf, padding=(self.erf-1)//2, stride=2) # (B, 9, C, H, W)
            bot_V = unfold_2d(bot_value[head_idx], kernel_size=self.erf, padding=(self.erf-1)//2, stride=2)

            weight = (bot_K.unsqueeze(2) * top_Q.unsqueeze(1)).sum(dim=3) / math.sqrt(self.key_dim) # (B, 9, 1, H, W)
            weight = weight.softmax(dim=1)
            
            out = torch.sum(weight.unsqueeze(dim=3)*bot_V.unsqueeze(dim=2), dim=1).squeeze(dim=1) # (B, C, H, W)
            top_ret = top_V + out
            top_results.append(top_ret)

        top_results = torch.cat(top_results, dim=1)
        top_results = self.top_fuse(top_results)

        return top_results, bot_feat