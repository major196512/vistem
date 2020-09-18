import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from .block import BlockBase

from vistem.modeling.backbone import Backbone
from vistem.modeling.layers import Conv2d, get_norm
from vistem.structures import ShapeSpec

__all__ = ['EfficientNetBase']

class EfficientNetBase(Backbone):
    def __init__(self, stem, stages, num_classes=0, dropout_prob=0.5, out_features=None):
        super().__init__()
        self.stage1 = stem
        self._out_feature_strides = {"stage1": self.stage1.stride}
        self._out_feature_channels = {"stage1": self.stage1.out_channels}

        current_stride = self.stage1.stride
        self.stages_and_names = []
        for i, blocks in enumerate(stages, start=2):
            for block in blocks:
                assert isinstance(block, BlockBase), block
                curr_channels = block.out_channels
                current_stride *= block.stride

            stage = nn.Sequential(*blocks)
            name = f"stage{str(i)}"
            self.add_module(name, stage)
            self.stages_and_names.append((stage, name))

            self._out_feature_strides[name] = current_stride
            self._out_feature_channels[name] = blocks[-1].out_channels

        if num_classes > 0:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.dropout = nn.Dropout(dropout_prob)
            self.linear = nn.Linear(curr_channels, num_classes)

            # Sec 5.1 in "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour":
            # "The 1000-way fully-connected layer is initialized by
            # drawing weights from a zero-mean Gaussian with standard deviation of 0.01."
            # nn.init.normal_(self.linear.weight, stddev=0.01)
            name = "linear"

        if out_features is None : out_features = [name]
        self._out_features = out_features
        assert len(self._out_features)

        children = [x[0] for x in self.named_children()]
        for out_feature in self._out_features:
            assert out_feature in children, f"Available children: {', '.join(children)}"

    def forward(self, x):
        outputs = {}
        x = self.stage1(x)
        if "stage1" in self._out_features:
            outputs["stage1"] = x

        for stage, name in self.stages_and_names:
            x = stage(x)
            if name in self._out_features:
                outputs[name] = x

        if hasattr(self, 'avgpool'):
            x = self.dropout(self.avgpool(x))
            x = self.linear(x.reshape(x.shape[0], -1))
            if "linear" in self._out_features:
                outputs["linear"] = x

        return outputs

    def freeze(self, freeze_at=0):
        if freeze_at >= 1:
            self.stage1.freeze()
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
