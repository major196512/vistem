import torch
import math

from vistem.modeling.layers import Conv2d, unfold_2d, fold_2d
from . import PLANDecode

__all__ = ['PLANBottomUp']

class PLANBottomUp(PLANDecode):
    def __init__(self, top_feat, bot_feat, in_channels, plan_cfg):
        super().__init__()
        self.decode_cfg(plan_cfg)
        self.out_channels = in_channels

        self.top_key = Conv2d(in_channels, self.num_heads * self.key_dim, 1, stride=1, bias=False)
        self.bot_query = Conv2d(in_channels, self.num_heads * self.key_dim, 1, stride=1, bias=False)
        self.bot_value = Conv2d(in_channels, self.num_heads * self.value_dim, 1, stride=1, bias=False)
        
        self.top_fuse = Conv2d(self.num_heads * self.value_dim, self.out_channels, 1, stride=1, bias=False)

        for layer in self.modules():
            if isinstance(layer, Conv2d):
                torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                if layer.bias is not None : torch.nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        top_feat, bot_feat = x
        top_key         = self.top_key(top_feat).split(self.key_dim, dim=1)
        bot_query       = self.bot_query(bot_feat).split(self.key_dim, dim=1)
        bot_value       = self.bot_value(bot_feat).split(self.value_dim, dim=1)

        top_results = []
        for head_idx in range(self.num_heads):
            bot_Q = unfold_2d(bot_query[head_idx], kernel_size=self.erf, padding=(self.erf-1)//2, stride=2) # (B, 9, C, H, W)
            bot_V = unfold_2d(bot_value[head_idx], kernel_size=self.erf, padding=(self.erf-1)//2, stride=2) # (B, 9, C, H, W)
            top_K = top_key[head_idx] # (B, C, H, W)

            top_weight = (top_K.unsqueeze(dim=1) * bot_Q).sum(dim=2) / math.sqrt(self.key_dim)
            top_weight = top_weight.softmax(dim=1) # (B, 9, H, W)

            top_out = (top_weight.unsqueeze(dim=2) * bot_V) # (B, 9, C, H, W)
            top_out = torch.sum(top_out, dim=1)
            top_results.append(top_out)

        top_results = torch.cat(top_results, dim=1)
        top_results = self.top_fuse(top_results)

        return top_results