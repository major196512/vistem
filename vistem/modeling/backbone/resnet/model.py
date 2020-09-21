import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from .block import ResNetBlockBase

from vistem.modeling.backbone import Backbone
from vistem.modeling.layers import Conv2d, get_norm
from vistem.structures import ShapeSpec

__all__ = ['ResNetBase']

class ResNetBase(Backbone):
    def __init__(self, stem, stages, out_features=None):
        super(ResNetBase, self).__init__()
        self.stem = stem
        current_stride = self.stem.stride
        self._out_feature_strides = {"stem": self.stem.stride}
        self._out_feature_channels = {"stem": self.stem.out_channels}

        self.stages_and_names = []
        for i, blocks in enumerate(stages):
            for block in blocks:
                assert isinstance(block, ResNetBlockBase), block
                curr_channels = block.out_channels

            stage = nn.Sequential(*blocks)
            name = "res" + str(i + 2)
            self.add_module(name, stage)
            self.stages_and_names.append((stage, name))

            current_stride = int(current_stride * np.prod([k.stride for k in blocks]))
            self._out_feature_strides[name] = current_stride
            self._out_feature_channels[name] = blocks[-1].out_channels

        if out_features is None:
            out_features = [name]
        self._out_features = out_features
        assert len(self._out_features)

        children = [x[0] for x in self.named_children()]
        for out_feature in self._out_features:
            assert out_feature in children, f"Available children: {', '.join(children)}"

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        if "stem" in self._out_features:
            outputs["stem"] = x

        for stage, name in self.stages_and_names:
            x = stage(x)
            if name in self._out_features:
                outputs[name] = x

        return outputs

    def freeze(self, freeze_at=0):
        if freeze_at >= 1:
            self.stem.freeze()
        for idx, (stage, _) in enumerate(self.stages_and_names, start=2):
            if freeze_at >= idx:
                for block in stage.children():
                    block.freeze()
        return self

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }
