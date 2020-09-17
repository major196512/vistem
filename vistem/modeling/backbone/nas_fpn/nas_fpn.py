import torch
import torch.nn.functional as F
import math

from typing import Dict
from .operation import binary_operation, RCB

from vistem.modeling.backbone import Backbone
from vistem.modeling.layers import Conv2d, get_norm
from vistem.utils import weight_init
from vistem.structures import ShapeSpec

__all__ = ['NASFPNBase']

class NASFPNBase(Backbone):
    def __init__(self, fpn, cell_inputs, cell_outputs, cell_ops, nas_outputs):
        super(NASFPNBase, self).__init__()
        assert isinstance(fpn, Backbone)

        in_features = fpn.out_features
        fpn_shapes = fpn.output_shape()

        self.cell_ops = []
        self.rcb_ops = []
        for idx, cell_op in enumerate(cell_ops):
            cell = binary_operation(cell_op)()
            rcb = RCB(fpn_shapes[cell_outputs[idx]])

            self.add_module(f"cell_op{idx}", cell)
            self.add_module(f"rcb{idx}", rcb)

            self.cell_ops.append(cell)
            self.rcb_ops.append(rcb)

        self.fpn = fpn
        self.in_features = fpn.out_features
        self.cell_inputs = cell_inputs
        self.cell_outputs = cell_outputs
        self.nas_outputs = nas_outputs

        self._out_features = fpn.out_features
        self._out_feature_channels = fpn._out_feature_channels
        self._out_feature_strides = fpn._out_feature_strides
        self._size_divisibility = fpn.size_divisibility

    @property
    def size_divisibility(self):
        return self._size_divisibility

    def forward(self, x) -> Dict[str, torch.Tensor]:
        fpn_features = self.fpn(x)

        for idx, (cell_op, rcb_op) in enumerate(zip(self.cell_ops, self.rcb_ops)):
            in_features = self.cell_inputs[idx]
            out_features = self.cell_outputs[idx]

            cell_feature = cell_op([fpn_features[k] for k in in_features], fpn_features[out_features].shape)
            rcb_feature = rcb_op(cell_feature)

            fpn_features[f'cell{idx+1}'] = cell_feature
            fpn_features[f'rcb{idx+1}'] = rcb_feature

        results = [fpn_features[k] for k in self.nas_outputs]
        return dict(zip(self._out_features, results))
