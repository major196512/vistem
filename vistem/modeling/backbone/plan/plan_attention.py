import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import re

from vistem.modeling.layers import Conv2d, unfold_2d, fold_2d

__all__ = ['PLANAttention']

class PLANAttention(nn.Module):
    def __init__(self, top_feat, bot_feat, out_channels, plan_cfg):
        super().__init__()

        self.top_feat = top_feat
        self.bot_feat = bot_feat

        self.out_channels = out_channels
        self.decode_cfg(plan_cfg)

        self.top_key = Conv2d(out_channels, self.num_heads * self.key_dim, 1, stride=1, bias=False)
        self.top_query = Conv2d(out_channels, self.num_heads * self.key_dim, 1, stride=1, bias=False)
        self.top_value = Conv2d(out_channels, self.num_heads * self.value_dim, 1, stride=1, bias=False)

        self.bot_key = Conv2d(out_channels, self.num_heads * self.key_dim, 1, stride=1, bias=False)
        self.bot_query = Conv2d(out_channels, self.num_heads * self.key_dim, 1, stride=1, bias=False)
        self.bot_value = Conv2d(out_channels, self.num_heads * self.value_dim, 1, stride=1, bias=False)

        self.top_fuse = Conv2d(self.num_heads * self.value_dim, out_channels, 1, stride=1, bias=False)
        self.bot_fuse = Conv2d(self.num_heads * self.value_dim, out_channels, 1, stride=1, bias=False)

        for layer in self.modules():
            if isinstance(layer, Conv2d):
                torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                if layer.bias is not None : torch.nn.init.constant_(layer.bias, 0)

    def decode_cfg(self, cfg):
        self.num_heads = int(re.compile('h[0-9]+').findall(cfg)[0][1:])
        self.key_dim = int(re.compile('k[0-9]+').findall(cfg)[0][1:])
        self.value_dim = int(re.compile('v[0-9]+').findall(cfg)[0][1:])
        self.erf = int(re.compile('e[0-9]+').findall(cfg)[0][1:])

    def forward(self, x):
        top_feat, bot_feat = x
        B, _, H, W = top_feat.shape
        results = []

        top_key         = self.top_key(top_feat).split(self.key_dim, dim=1)
        top_query       = self.top_query(top_feat).split(self.key_dim, dim=1)
        top_value       = self.top_value(top_feat).split(self.value_dim, dim=1)

        bot_key         = self.bot_key(bot_feat).split(self.key_dim, dim=1)
        bot_query       = self.bot_query(bot_feat).split(self.key_dim, dim=1)
        bot_value       = self.bot_value(bot_feat).split(self.value_dim, dim=1)

        top_results = []
        bot_results = []
        for head_idx in range(self.num_heads):
            top_K = top_key[head_idx].reshape(B, 1, -1, H, W) # (B, 1, C, H, W)
            top_Q = top_query[head_idx].reshape(B, 1, -1, H, W)
            top_V = top_value[head_idx].reshape(B, 1, -1, H, W)

            bot_K = unfold_2d(bot_key[head_idx], kernel_size=self.erf, padding=(self.erf-1)//2, stride=2) # (B, 9, C, H, W)
            bot_Q = unfold_2d(bot_query[head_idx], kernel_size=self.erf, padding=(self.erf-1)//2, stride=2)
            bot_V = unfold_2d(bot_value[head_idx], kernel_size=self.erf, padding=(self.erf-1)//2, stride=2)

            key = torch.cat([top_K, bot_K], dim=1) # (B, 10, C, H, W)
            query = torch.cat([top_Q, bot_Q], dim=1)
            value = torch.cat([top_V, bot_V], dim=1)

            weight = (key.unsqueeze(2) * query.unsqueeze(1)).sum(dim=3) / math.sqrt(self.key_dim) # (B, 10, 10, H, W)
            out = (weight.unsqueeze(dim=3) * value.unsqueeze(dim=2)).sum(dim=2) # (B, 10, C, H, W)

            top_ret = out[:, 0, :, :, :]
            bot_ret = out[:, 1:, :, :, :]
            bot_ret = fold_2d(bot_ret, out_shape=bot_feat.shape[2:], kernel_size=self.erf, padding=(self.erf-1)//2, stride=2)

            top_results.append(top_ret)
            bot_results.append(bot_ret)

        top_results = torch.cat(top_results, dim=1)
        bot_results = torch.cat(bot_results, dim=1)
        
        top_results = self.top_fuse(top_results)
        bot_results = self.bot_fuse(bot_results)

        return top_results, bot_results