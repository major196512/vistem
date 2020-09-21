import torch
import torch.nn as nn

import math
from typing import List

from . import META_ARCH_REGISTRY, DefaultMetaArch
from vistem.modeling import Box2BoxTransform, Matcher, detector_postprocess
from vistem.modeling.backbone import build_backbone
from vistem.modeling.layers import Conv2d, batched_nms
from vistem.modeling.model_utils import permute_to_N_HWA_K, pairwise_iou

from vistem.structures import ImageList, ShapeSpec, Boxes, Instances

__all__ = ['EfficientNet']

@META_ARCH_REGISTRY.register()
class EfficientNet(DefaultMetaArch):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.in_features                = cfg.META_ARCH.EFFICIENTNET.IN_FEATURES
        self.num_classes                = cfg.META_ARCH.NUM_CLASSES

        dropout_prob                    = cfg.META_ARCH.EFFICIENTNET.DROPOUT_PROB

        assert type(self.in_features) == str

        # Backbone Network
        self.backbone = build_backbone(cfg)
        assert self.in_features in self.backbone.out_features, f"'{self.in_features}' is not in backbone({self.backbone.out_features})"

        backbone_shape = self.backbone.output_shape()
        feature_shapes = backbone_shape[self.in_features]

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_prob)
        self.linear = Conv2d(feature_shapes.channels, self.num_classes, 1, 1)

        # Sec 5.1 in "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour":
        # "The 1000-way fully-connected layer is initialized by
        # drawing weights from a zero-mean Gaussian with standard deviation of 0.01."
        # nn.init.normal_(self.linear.weight, stddev=0.01)


    def forward(self, batched_inputs):
        images, gt_annotations, _, _ = self.preprocess_image(batched_inputs)

        features = self.backbone(images.tensor)
        features = self.avgpool(features[self.in_features])
        features = self.dropout(features)
        features = self.linear(features).reshape(features.shape[0], -1)
        features = torch.softmax(features, dim=1)

        if self.training:
            pass

        else:
            processed_results = []
            for results_per_image in features:
                processed_results.append({"annotations": results_per_image})

            return processed_results

    def losses(self):
        pass
