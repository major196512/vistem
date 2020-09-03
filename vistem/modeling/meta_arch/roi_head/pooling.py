import torch
import torch.nn as nn

import math
from typing import List, Tuple, Union

from vistem.structures import Boxes

from torchvision.ops import RoIPool
from .roi_align import ROIAlign

class ROIPooler(nn.Module):
    def __init__(
        self,
        output_size : Union[int, Tuple[int, int]],
        scales : Tuple[float],
        sampling_ratio : int,
        pooler_type : str,
        canonical_box_size : int = 224,
        canonical_level : int = 4,
    ):
        super().__init__()

        if isinstance(output_size, int) : output_size = (output_size, output_size)
        self.output_size = output_size

        # Select Pooling Type
        if pooler_type == "ROIAlign":
            self.level_poolers = nn.ModuleList(
                ROIAlign(
                    output_size, spatial_scale=scale, sampling_ratio=sampling_ratio, aligned=False
                )
                for scale in scales
            )
        elif pooler_type == "ROIAlignV2":
            self.level_poolers = nn.ModuleList(
                ROIAlign(
                    output_size, spatial_scale=scale, sampling_ratio=sampling_ratio, aligned=True
                )
                for scale in scales
            )
        elif pooler_type == "ROIPool":
            self.level_poolers = nn.ModuleList(
                RoIPool(output_size, spatial_scale=scale) for scale in scales
            )
        else:
            raise ValueError(f"Unknown pooler type: {pooler_type}")

        # Map scale (defined as 1 / stride) to its feature map level under the assumption that stride is a power of 2.
        min_level = -(math.log2(scales[0]))
        max_level = -(math.log2(scales[-1]))
        assert math.isclose(min_level, int(min_level)) and math.isclose(max_level, int(max_level)), \
            "Featuremap stride is not power of 2!"

        self.min_level = int(min_level)
        self.max_level = int(max_level)
        assert len(scales) == self.max_level - self.min_level + 1, \
            "[ROIPooler] Sizes of input featuremaps do not form a pyramid!"
        assert 0 <= self.min_level and self.min_level <= self.max_level

        self.canonical_level = canonical_level
        assert canonical_box_size > 0
        self.canonical_box_size = canonical_box_size

    def forward(
        self, 
        x : List[torch.Tensor], 
        box_lists: List[Boxes]
    ) -> torch.Tensor: # (#boxes, channels, output_size, output_size)
    
        num_level_assignments = len(self.level_poolers)

        assert len(x) == num_level_assignments, f"unequal value, num_level_assignments={num_level_assignments}, but x is list of {len(x)} Tensors"
        assert len(box_lists) == x[0].size(0), f"unequal value, x[0] batch dim 0 is {x[0].size(0)}, but box_list has length {len(box_lists)}"
        
        if len(box_lists) == 0:
            return torch.zeros((0, x[0].shape[1]) + self.output_size, device=x[0].device, dtype=x[0].dtype)

        # convert_boxes_to_pooler_format(0col : img_id, 1~4col : box)
        def _fmt_box_list(box_tensor: torch.Tensor, batch_index: int):
            repeated_index = torch.full(
                (len(box_tensor), 1), batch_index, dtype=box_tensor.dtype, device=box_tensor.device
            )
            return torch.cat((repeated_index, box_tensor), dim=1)
        pooler_fmt_boxes = torch.cat([_fmt_box_list(box_list.tensor, i) for i, box_list in enumerate(box_lists)], dim=0)

        level_assignments = self.assign_boxes_to_levels(
            box_lists, self.min_level, self.max_level, self.canonical_box_size, self.canonical_level
        )

        num_boxes = len(pooler_fmt_boxes)
        num_channels = x[0].shape[1]
        output_size = self.output_size[0]

        dtype, device = x[0].dtype, x[0].device
        output = torch.zeros(
            (num_boxes, num_channels, output_size, output_size), dtype=dtype, device=device
        )

        for level, pooler in enumerate(self.level_poolers):
            inds = torch.nonzero(level_assignments == level, as_tuple=True)[0]
            pooler_fmt_boxes_level = pooler_fmt_boxes[inds]
            output[inds] = pooler(x[level], pooler_fmt_boxes_level)

        return output

    def assign_boxes_to_levels(
        self,
        box_lists: List[Boxes],
        min_level: int,
        max_level: int,
        canonical_box_size: int,
        canonical_level: int,
    ):
        box_sizes = torch.sqrt(torch.cat([boxes.area() for boxes in box_lists]))
        # Eqn.(1) in FPN paper
        level_assignments = torch.floor(
            canonical_level + torch.log2(box_sizes / canonical_box_size + 1e-8)
        )
        level_assignments = torch.clamp(level_assignments, min=min_level, max=max_level)
        return level_assignments.to(torch.int64) - min_level