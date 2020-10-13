import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import re

from . import PLANDecode, PLAN_REGISTRY
from vistem.modeling.layers import Conv2d, unfold_2d, fold_2d

__all__ = ['PLAN_mode7']

@PLAN_REGISTRY.register()
class PLAN_mode7(PLANDecode):
    def __init__(self, top_feat, bot_feat, out_channels, plan_cfg):
        super().__init__()

        self.top_feat = top_feat
        self.bot_feat = bot_feat

        self.out_channels = out_channels
        self.decode_cfg(plan_cfg)

        self.top_key = Conv2d(out_channels, self.num_heads * self.key_dim, 1, stride=1, bias=False)
        self.top_query = Conv2d(out_channels, self.num_heads * self.key_dim, 1, stride=1, bias=False)
        self.top_value = Conv2d(out_channels, self.num_heads * self.value_dim, 1, stride=1, bias=False)

        self.bot_query = Conv2d(out_channels, self.num_heads * self.key_dim, 1, stride=1, bias=False)
        self.bot_key = Conv2d(out_channels, self.num_heads * self.key_dim, 1, stride=1, bias=False)
        self.bot_value = Conv2d(out_channels, self.num_heads * self.value_dim, 1, stride=1, bias=False)

        for layer in self.modules():
            if isinstance(layer, Conv2d):
                torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                if layer.bias is not None : torch.nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        top_feat, bot_feat = x
        B, _, H, W = top_feat.shape
        results = []

        top_query       = self.top_query(top_feat).split(self.key_dim, dim=1)
        top_key         = self.top_key(top_feat).split(self.key_dim, dim=1)
        top_value       = self.top_value(top_feat).split(self.value_dim, dim=1)

        bot_query       = self.bot_query(bot_feat).split(self.key_dim, dim=1)
        bot_key         = self.bot_key(bot_feat).split(self.key_dim, dim=1)
        bot_value       = self.bot_value(bot_feat).split(self.value_dim, dim=1)

        top_results = []
        bot_results = []
        for head_idx in range(self.num_heads):
            top_Q = top_query[head_idx] # (B, C, H, W)
            top_V = top_value[head_idx] # (B, C, H, W)
            bot_K = unfold_2d(bot_key[head_idx], kernel_size=self.erf, padding=(self.erf-1)//2, stride=2) # (B, 9, C, H, W)

            bot_weight = (bot_K * top_Q.unsqueeze(1)).sum(dim=2) / math.sqrt(self.key_dim) # (B, 9, H, W)
            bot_weight = bot_weight.softmax(dim=1)
            
            bot_out = (bot_weight.unsqueeze(dim=2)*top_V.unsqueeze(dim=1)) # (B, 9, C, H, W)
            bot_out = fold_2d(bot_out, out_shape=bot_feat.shape[2:], kernel_size=self.erf, padding=(self.erf-1)//2, stride=2)
            bot_results.append(bot_out)

            bot_Q = unfold_2d(bot_query[head_idx], kernel_size=self.erf, padding=(self.erf-1)//2, stride=2) # (B, 9, C, H, W)
            bot_V = unfold_2d(bot_value[head_idx], kernel_size=self.erf, padding=(self.erf-1)//2, stride=2) # (B, 9, C, H, W)
            top_K = top_key[head_idx] # (B, C, H, W)

            top_weight = (top_K.unsqueeze(dim=1) * bot_Q).sum(dim=2) / math.sqrt(self.key_dim)
            top_weight = top_weight.softmax(dim=1) # (B, 9, H, W)

            top_out = (top_weight.unsqueeze(dim=2) * bot_V) # (B, 9, C, H, W)
            top_out = torch.sum(top_out, dim=1)
            top_results.append(top_out)

        return top_results, bot_results