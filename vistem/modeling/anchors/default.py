import torch
import torch.nn as nn
import copy
import math
from typing import List

from . import ANCHOR_GENERATOR_REGISTRY
from .base import BufferList, create_grid_offsets
from vistem.structures import Boxes, ShapeSpec

@ANCHOR_GENERATOR_REGISTRY.register()
class DefaultAnchorGenerator(nn.Module):
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()
        # fmt: off
        sizes         = cfg.ANCHOR_GENERATOR.SIZES
        aspect_ratios = cfg.ANCHOR_GENERATOR.ASPECT_RATIOS
        self.strides  = [x.stride for x in input_shape]
        # fmt: on

        self.num_features = len(self.strides)
        self.cell_anchors = self._calculate_anchors(sizes, aspect_ratios)

    def _calculate_anchors(self, sizes, aspect_ratios):
        if len(sizes) == 1:
            sizes *= self.num_features
        if len(aspect_ratios) == 1:
            aspect_ratios *= self.num_features
        assert self.num_features == len(sizes)
        assert self.num_features == len(aspect_ratios)

        cell_anchors = [
            self.generate_cell_anchors(s, a).float() for s, a in zip(sizes, aspect_ratios)
        ]

        return BufferList(cell_anchors)

    @property
    def box_dim(self):
        return 4

    @property
    def num_cell_anchors(self):
        return [len(cell_anchors) for cell_anchors in self.cell_anchors]

    def grid_anchors(self, grid_sizes):
        anchors = []
        for size, stride, base_anchors in zip(grid_sizes, self.strides, self.cell_anchors):
            shift_x, shift_y = create_grid_offsets(size, stride, base_anchors.device)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

            anchors.append((shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4))

        return anchors

    def generate_cell_anchors(self, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2)):
        anchors = []
        for size in sizes:
            area = size ** 2.0
            for aspect_ratio in aspect_ratios:
                # s * s = w * h
                # a = h / w
                # ... some algebra ...
                # w = sqrt(s * s / a)
                # h = a * w
                w = math.sqrt(area / aspect_ratio)
                h = aspect_ratio * w
                x0, y0, x1, y1 = -w / 2.0, -h / 2.0, w / 2.0, h / 2.0
                anchors.append([x0, y0, x1, y1])
        return torch.tensor(anchors)

    def forward(self, features):
        num_images = len(features[0])
        grid_sizes = [feature_map.shape[-2:] for feature_map in features]
        anchors_over_all_feature_maps = self.grid_anchors(grid_sizes)

        anchors_in_image = []
        for anchors_per_feature_map in anchors_over_all_feature_maps:
            boxes = Boxes(anchors_per_feature_map)
            anchors_in_image.append(boxes)

        anchors = [copy.deepcopy(anchors_in_image) for _ in range(num_images)]
        return anchors
