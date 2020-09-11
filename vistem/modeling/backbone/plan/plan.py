import torch
import torch.nn as nn
from typing import List
import math

from vistem.modeling.backbone import Backbone
from vistem.structures import ShapeSpec

from .attn_top_down import AttnTopDown
from .attn_bottom_up import AttnBottomUp

__all__ = ['PLANBase']

class PLANBase(Backbone):
    def __init__(self, fpn, out_channels, num_heads, num_convs, num_weights, erf):
        super().__init__()

        in_features                 = fpn.out_features
        fpn_shape                   = fpn.output_shape()

        self._out_features          = in_features
        self._out_feature_strides   = {name : s.stride for name, s in fpn_shape.items()}
        self._out_feature_channels  = {name : s.channels for name, s in fpn_shape.items()}
        self._size_divisibility     = fpn.size_divisibility

        in_channels = self._out_feature_channels
        assert (len(in_channels) == len(in_features)) and (all(k in in_channels for k in in_features))

        assert out_channels % num_heads == 0

        self.fpn = fpn
        self.top_down = AttnTopDown(in_features, in_channels, out_channels, num_heads, num_convs, erf)
        

    def forward(self, x):
        fpn_features = self.fpn(x)
        top_down_features = self.top_down(fpn_features)

        results = top_down_features
        return results

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

