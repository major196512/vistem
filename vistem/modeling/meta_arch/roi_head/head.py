import torch
import torch.nn as nn

import numpy as np
from typing import List, Dict, Tuple, Optional

from . import ROI_REGISTRY
from vistem.modeling import Matcher, subsample_labels
from vistem.structures import ImageList, Instances, ShapeSpec, Boxes
from vistem.modeling.model_utils import pairwise_iou
from vistem.modeling.meta_arch.proposal.proposal_utils import add_ground_truth_to_proposals

from vistem.utils import get_event_storage

@ROI_REGISTRY.register()
class StandardROIHeads(nn.Module):
    def __init__(self, cfg, input_shape):
        super().__init__()
        self.in_features                    = cfg.MODEL.ROI_HEAD.IN_FEATURES
        self.num_classes                    = cfg.MODEL.ROI_HEAD.NUM_CLASSES
        for feat in self.in_features:
            assert feat in input_shape.keys(), f"'{feat}' is not in backbone({input_shape.keys()})"

        self.proposal_append_gt             = cfg.MODEL.ROI_HEAD.PROPOSAL_APPEND_GT
        self.batch_size_per_image           = cfg.MODEL.ROI_HEAD.BATCH_SIZE_PER_IMAGE
        self.positive_fraction              = cfg.MODEL.ROI_HEAD.POSITIVE_FRACTION

        self.proposal_matcher = Matcher(
                cfg.MODEL.ROI_HEAD.IOU_THRESHOLDS,
                cfg.MODEL.ROI_HEAD.IOU_LABELS,
                allow_low_quality_matches=False,
            )






        # self.train_on_pred_boxes = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES

        # self.in_features = self.box_in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        # in_channels = [input_shape[f].channels for f in in_features]
        # # Check all channel counts are equal
        # assert len(set(in_channels)) == 1, in_channels
        # in_channels = in_channels[0]

        # self.box_pooler = ROIPooler(
        #     output_size=cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION,
        #     scales=tuple(1.0 / input_shape[k].stride for k in in_features),
        #     sampling_ratio=cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO,
        #     pooler_type=cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE,
        # )
        # self.box_head = build_box_head(
        #     cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        # )
        # self.box_predictor = FastRCNNOutputLayers(cfg, box_head.output_shape)

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:

        del images
        if self.training:
            assert targets
            proposals = self.get_ground_truth(proposals, targets)
        del targets

        # if self.training:
        #     losses = self._forward_box(features, proposals)
        #     return proposals, losses
        # else:
        #     pred_instances = self._forward_box(features, proposals)
        #     pred_instances = self.forward_with_given_boxes(features, pred_instances)
        #     return pred_instances, {}

        return [], {}

    
    @torch.no_grad()
    def get_ground_truth(
        self, 
        proposals: List[Instances], 
        targets: List[Instances]
    ) -> List[Instances]:

        gt_boxes = [x.gt_boxes for x in targets]
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []
        num_fg_samples = []
        num_bg_samples = []

        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(targets_per_image.gt_boxes, proposals_per_image.proposal_boxes) # (#gt, #proposal)
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            del match_quality_matrix

            has_gt = targets_per_image.gt_classes.numel() > 0
            if has_gt:
                gt_classes = targets_per_image.gt_classes[matched_idxs]
                gt_classes[matched_labels == 0] = self.num_classes
                gt_classes[matched_labels == -1] = -1
            else:
                gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

            # Sampling
            sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
                gt_classes, self.batch_size_per_image, self.positive_fraction, self.num_classes
            )
            sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)

            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = targets_per_image.gt_classes[sampled_idxs]

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                proposals_per_image.gt_boxes = targets_per_image.gt_boxes[sampled_targets]
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            else:
                proposals_per_image.gt_boxes = Boxes(targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4)))

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt
