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
    def __init__(self, stem, stages, num_classes=None, out_features=None):
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

        self.num_classes = num_classes
        if num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.linear = nn.Linear(curr_channels, num_classes)

            # Sec 5.1 in "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour":
            # "The 1000-way fully-connected layer is initialized by
            # drawing weights from a zero-mean Gaussian with standard deviation of 0.01."
            nn.init.normal_(self.linear.weight)#, stddev=0.01)
            name = "linear"

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

        if self.num_classes is not None:
            x = self.avgpool(x)
            x = self.linear(x.reshape(x.shape[0], -1))
            if "linear" in self._out_features:
                outputs["linear"] = x

        return outputs

    def output_shape(self):
        print(self._out_features)
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }
