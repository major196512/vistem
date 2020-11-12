import torch
import re

from vistem.modeling.backbone import Backbone
from vistem.structures import ShapeSpec

from .block import PLANBlock

__all__ = ['PLANBase']

class PLANBase(Backbone):
    def __init__(self, pyramidal, interlayer_mode, fuse_mode, plan_cfg):
        super().__init__()

        in_features                 = pyramidal.out_features
        pyramidal_shape               = pyramidal.output_shape()

        self._out_features          = in_features
        self._out_feature_strides   = {name : s.stride for name, s in pyramidal_shape.items()}
        self._out_feature_channels  = {name : s.channels for name, s in pyramidal_shape.items()}
        self._size_divisibility     = max([p.stride for p in pyramidal_shape.values()])

        in_channels = self._out_feature_channels
        assert (len(in_channels) == len(in_features)) and (all(k in in_channels for k in in_features))

        self.pyramidal = pyramidal

        repeat = int(re.compile('r[0-9]+').findall(plan_cfg)[0][1:])
        out_channels = self._out_feature_channels
        self.plan_blocks = []
        for idx in range(repeat):
            plan_block = PLANBlock(in_features, out_channels, interlayer_mode, fuse_mode, plan_cfg)
            self.plan_blocks.append(plan_block)
            self.add_module(f'PLANBlock{idx}', plan_block)

    def forward(self, x):
        results = self.pyramidal(x)

        for plan_block in self.plan_blocks:
            results = plan_block(results)

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

