import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .block import TopDownBlock, BottomUpBlock

from vistem.modeling.backbone import Backbone
from vistem.modeling.layers import Conv2d, get_norm
from vistem.utils import weight_init
from vistem.structures import ShapeSpec

__all__ = ['BiFPNBase']

class BiFPNBase(Backbone):
    def __init__(self, bottom_up, top_block, out_channels, num_layers, norm='BN', fuse_type='fast_norm'):
        super().__init__()

        self.bottom_up = bottom_up
        assert isinstance(bottom_up, Backbone)

        self.top_block = top_block
        assert top_block, 'Top-Block in BiFPN should be callable functions.'

        in_channels = top_block.out_channels
        out_features = top_block.out_features
        
        td_blocks = []
        bu_blocks = []
        for _ in range(num_layers):
            td_blocks.append(TopDownBlock(in_channels, out_channels, norm=norm, fuse_type=fuse_type))
            bu_blocks.append(BottomUpBlock(in_channels, out_channels, norm=norm, fuse_type=fuse_type))
            in_channels = {k:out_channels for k in in_channels.keys()}

        self.num_layers = num_layers
        self.td_blocks = nn.ModuleList(td_blocks)
        self.bu_blocks = nn.ModuleList(bu_blocks)

        # weight_init.c2_xavier_fill(lateral_conv)
        # weight_init.c2_xavier_fill(output_conv)

        in_features = top_block.in_features
        out_feature_strides = [bottom_up.out_feature_strides[f] for f in in_features]
        for _ in range(top_block.num_block):
            s = out_feature_strides[-1]
            out_feature_strides.append(2*s)
        _assert_strides_are_log2_contiguous(out_feature_strides)

        self._out_features = top_block.out_features
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self._out_feature_strides = dict(zip(self._out_features, out_feature_strides))
        self._size_divisibility = out_feature_strides[-1]

    @property
    def size_divisibility(self):
        return self._size_divisibility

    def forward(self, x):
        bottom_up_features = self.bottom_up(x)
        x = self.top_block(bottom_up_features)
        
        for td_block, bu_block in zip(self.td_blocks, self.bu_blocks):
            td_nodes = td_block(x)
            x = bu_block([x, td_nodes])
        
        return x

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


def _assert_strides_are_log2_contiguous(strides):
    for i, stride in enumerate(strides[1:], 1):
        assert stride == 2 * strides[i - 1], f"Strides {stride} {strides[i - 1]} are not log2 contiguous"
