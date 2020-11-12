import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import defaultdict

from vistem.modeling.layers import Conv2d
from vistem.utils import weight_init

from .inter_layer import InterLayer
from .fuse import PLANFuse

__all__ = ['PLANBlock']

class PLANBlock(nn.Module):
    def __init__(self, in_features, out_channels, interlayer_mode, fuse_mode, plan_cfg):
        super().__init__()

        self.inter_layers = []
        fuse_channels = defaultdict(int)
        for top_feat, bot_feat in zip(in_features[1:], in_features[:-1]):
            assert out_channels[top_feat] == out_channels[bot_feat]
            out_channel = out_channels[bot_feat]
            
            inter_layer = InterLayer(interlayer_mode, top_feat, bot_feat, out_channel, plan_cfg)
            self.inter_layers.append(inter_layer)
            self.add_module(f'InterLayer_{top_feat}_{bot_feat}', inter_layer)

            fuse_channels[top_feat] += inter_layer.top_out_channels
            fuse_channels[bot_feat] += inter_layer.bot_out_channels

        self.plan_fuses = []
        self.convs = []
        for feat in in_features:
            in_channels = fuse_channels[feat]
            out_channel = out_channels[feat]
            
            plan_fuse = PLANFuse(fuse_mode, feat, in_channels, out_channel)

            self.plan_fuses.append(plan_fuse)
            self.add_module(f'PLANFuse_{feat}', plan_fuse)

    def forward(self, x):
        inter_layer_results = defaultdict(list)
        for inter_layer in self.inter_layers:
            top_feat = inter_layer.top_feat
            bot_feat = inter_layer.bot_feat
            plan_top, plan_bot = inter_layer([x[top_feat], x[bot_feat]])

            if plan_bot is not None : inter_layer_results[bot_feat].append(plan_bot)
            if plan_top is not None : inter_layer_results[top_feat].append(plan_top)

        results = defaultdict(list)
        for plan_fuse in self.plan_fuses:
            feat = plan_fuse.feat
            results[feat] = plan_fuse([x[feat], inter_layer_results[feat]])

        return results

