import torch
import math

from vistem.modeling.layers import Conv2d, unfold_2d, fold_2d
from . import PLANDecode

__all__ = ['PLANTopDown']

class PLANTopDown(PLANDecode):
    def __init__(self, top_feat, bot_feat, in_channels, plan_cfg):
        super().__init__()
        self.decode_cfg(plan_cfg)
        self.out_channels = in_channels

        self.top_query = Conv2d(in_channels, self.num_heads * self.key_dim, 1, stride=1, bias=False)
        self.top_value = Conv2d(in_channels, self.num_heads * self.value_dim, 1, stride=1, bias=False)
        self.bot_key = Conv2d(in_channels, self.num_heads * self.key_dim, 1, stride=1, bias=False)
        
        self.bot_fuse = Conv2d(self.num_heads * self.value_dim, self.out_channels, 1, stride=1, bias=False)

        for layer in self.modules():
            if isinstance(layer, Conv2d):
                torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                if layer.bias is not None : torch.nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        top_feat, bot_feat = x
        top_query       = self.top_query(top_feat).split(self.key_dim, dim=1)
        top_value       = self.top_value(top_feat).split(self.value_dim, dim=1)
        bot_key         = self.bot_key(bot_feat).split(self.key_dim, dim=1)

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

        bot_results = torch.cat(bot_results, dim=1)
        bot_results = self.bot_fuse(bot_results)

        return bot_results