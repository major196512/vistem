import torch
import torch.nn.functional as F
import math

from typing import Dict

from vistem.modeling.backbone import Backbone
from vistem.modeling.layers import Conv2d, get_norm
from vistem.utils import weight_init
from vistem.structures import ShapeSpec

__all__ = ['PANetBase']

class PANetBase(Backbone):
    def __init__(self, fpn, in_features, out_channels, use_bias=True, norm=""):
        super(PANetBase, self).__init__()
        assert isinstance(fpn, Backbone)

        self.fpn = fpn
        self.in_features = in_features

        fpn_shape = fpn.output_shape()
        in_channels = [fpn_shape[k].channels for k in self.in_features]
        in_strides = [fpn_shape[k].stride for k in self.in_features]

        self.lateral_convs = []
        self.output_convs = []
        for idx, in_channels in enumerate(in_channels[:-1], start=1):
            t = get_norm(norm, out_channels)
            lateral_conv = Conv2d(
                in_channels, out_channels, 
                kernel_size=3,
                stride=2,
                padding=1,
                bias=use_bias, 
                norm=get_norm(norm, out_channels)
            )
            output_conv = Conv2d(
                out_channels, out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm=get_norm(norm, out_channels)
            )
            weight_init.c2_xavier_fill(lateral_conv)
            weight_init.c2_xavier_fill(output_conv)

            stage = int(math.log2(in_strides[idx]))
            self.add_module(f"panet_lateral{stage}", lateral_conv)
            self.add_module(f"panet_output{stage}", output_conv)

            self.lateral_convs.append(lateral_conv)
            self.output_convs.append(output_conv)

        self._out_feature_strides = {f"n{int(math.log2(s))}": s for s in in_strides}
        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self._size_divisibility = in_strides[-1]

    @property
    def size_divisibility(self):
        return self._size_divisibility

    def forward(self, x) -> Dict[str, torch.Tensor]:
        results = list()

        fpn_features = self.fpn(x)
        results.append(fpn_features[self.in_features[0]])
        for idx, fpn_feat in enumerate(self.in_features[1:]):
            lateral_feat = self.lateral_convs[idx](results[-1])
            prev_feat = fpn_features[fpn_feat] + lateral_feat
            results.append(self.output_convs[idx](prev_feat))

        return dict(zip(self._out_features, results))
        
