import torch
import torch.nn as nn
from typing import List
import math

from vistem.modeling.backbone import Backbone
from vistem.structures import ShapeSpec

# from .attn_top_down import AttnTopDown
# from .attn_bottom_up import AttnBottomUp
# from .attn_concat import AttnFuse
from .plan_attention import PLANAttention

__all__ = ['PLANBase']

class PLANBase(Backbone):
    def __init__(self, pyramid, out_channels, plan_cfg):
        super().__init__()

        in_features                 = pyramid.out_features
        pyramid_shape               = pyramid.output_shape()

        self._out_features          = in_features
        self._out_feature_strides   = {name : s.stride for name, s in pyramid_shape.items()}
        self._out_feature_channels  = {name : s.channels for name, s in pyramid_shape.items()}
        self._size_divisibility     = max([p.stride for p in pyramid_shape.values()])

        in_channels = self._out_feature_channels
        assert (len(in_channels) == len(in_features)) and (all(k in in_channels for k in in_features))

        # assert out_channels % num_heads == 0

        self.pyramid = pyramid
        self.plan_top_down = []
        for top_feat, bot_feat in zip(in_features[1:], in_features[:-1]):
            assert self._out_feature_channels[top_feat] == self._out_feature_channels[bot_feat]
            out_channels = self._out_feature_channels[bot_feat]
            
            attn = PLANAttention(top_feat, bot_feat, out_channels, plan_cfg)
            self.plan_top_down.append(attn)
            self.add_module(f'TopDown_{top_feat}_{bot_feat}', attn)

        self.plan_top_down = self.plan_top_down[::-1]

        # self.plan_top_down = AttnTopDown(in_features, in_channels, out_channels, num_heads, num_convs, erf)
        # self.plan_bottom_up = AttnBottomUp(in_features, in_channels, out_channels, num_heads, num_convs, erf)
        # self.plan_fuse = AttnFuse(in_features, out_channels, num_weights)
        

    def forward(self, x):
        pyramid_features = self.pyramid(x)

        results = []
        prev_top = self.plan_top_down[0].top_feat
        prev_top = pyramid_features[prev_top]

        for plan_top_down in self.plan_top_down:
            bot_feat = plan_top_down.bot_feat
            plan_top, plan_bot = plan_top_down([prev_top, pyramid_features[bot_feat]])

            results.append(plan_top)
            prev_top = plan_bot

        results.append(prev_top)

        return dict(zip(self._out_features, results))

    @property
    def size_divisibility(self):
        return self._size_divisibility

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

