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
    def __init__(self, stem, stages, head, out_features=None):
        super(EfficientNetBase, self).__init__()

        self.stem = stem
        current_stride = self.stem.stride

        self._stage_strides = {"stage1": stem.stride}
        self._stage_channels = {"stage1": stem.out_channels}

        self.blocks = []
        self.stage_to_block = dict()
        self.block_to_stage = dict()

        for i, blocks in enumerate(stages, start=2):
            for block in blocks:
                assert isinstance(block, BlockBase), block
                curr_channels = block.out_channels
                current_stride *= block.stride

                block_name = f"block{str(len(self.blocks))}"
                self.add_module(block_name, block)
                self.blocks.append(block)

            stage_name = f'stage{str(i)}'
            block_idx = len(self.blocks) - 1

            self.stage_to_block[stage_name] = block_idx
            self.block_to_stage[block_idx] = stage_name
            
            self._stage_strides[stage_name] = current_stride
            self._stage_channels[stage_name] = blocks[-1].out_channels

        self.head = head
        current_stride *= head.stride

        stage_name = 'stage9'
        self._stage_strides[stage_name] = current_stride
        self._stage_channels[stage_name] = head.out_channels

        if out_features is None : out_features = [stage_name]
        self._out_features = out_features
        assert len(self._out_features)

        for out_feature in self._out_features:
            assert out_feature in self._stage_strides.keys(), f"Available stages: {', '.join(self._stage_strides.keys())}"

        self._out_feature_strides = self._stage_strides
        self._out_feature_channels = self._stage_channels

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        if 'stage1' in self._out_features :
            outputs['stage1'] = x

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.block_to_stage.keys():
                stage_name = self.block_to_stage[idx]
                outputs[stage_name] = x

        x = self.head(x)
        if 'stage9' in self._out_features :
            outputs['stage9'] = x

        return outputs

    def freeze(self, freeze_at=0):
        if freeze_at == 0 : return self
        self.stem.freeze()

        freeze_stage_idx = min(8, freeze_at)
        freeze_stage_name = f'stage{str(freeze_stage_idx)}'
        freeze_block_idx = self.stage_to_block[freeze_stage_name]

        for idx, block in enumerate(self.blocks):
            if freeze_block_idx >= idx:
                block.freeze()

        if freeze_at >= 9 : self.head.freeze()
        return self

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._stage_channels[name], stride=self._stage_strides[name]
            )
            for name in self._out_features
        }
        