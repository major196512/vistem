import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from typing import List, Dict, Tuple, Union, Optional

from . import ROI_REGISTRY
from .pooling import ROIPooler

from vistem.modeling import Box2BoxTransform, Matcher, subsample_labels
from vistem.modeling.model_utils import pairwise_iou
from vistem.modeling.layers import Conv2d, Linear
from vistem.modeling.layers.norm import get_norm
from vistem.modeling.meta_arch.proposal.proposal_utils import add_ground_truth_to_proposals

from vistem.structures import ImageList, Instances, ShapeSpec, Boxes
from vistem.utils.losses import smooth_l1_loss
from vistem.utils import get_event_storage
from vistem.utils import weight_init

@ROI_REGISTRY.register()
class StandardROIHeads(nn.Module):
    def __init__(self, cfg, input_shape):
        super().__init__()
        self.in_features                    = cfg.MODEL.ROI_HEAD.IN_FEATURES
        self.num_classes                    = cfg.MODEL.ROI_HEAD.NUM_CLASSES
        in_channels                         = [input_shape[f].channels for f in self.in_features]
        assert len(set(in_channels)) == 1, in_channels
        for feat in self.in_features:
            assert feat in input_shape.keys(), f"'{feat}' is not in backbone({input_shape.keys()})"

        self.proposal_append_gt             = cfg.MODEL.ROI_HEAD.PROPOSAL_APPEND_GT
        self.batch_size_per_image           = cfg.MODEL.ROI_HEAD.BATCH_SIZE_PER_IMAGE
        self.positive_fraction              = cfg.MODEL.ROI_HEAD.POSITIVE_FRACTION

        pooler_resolution = cfg.MODEL.ROI_HEAD.BOX_POOLER_RESOLUTION

        self.proposal_matcher = Matcher(
                cfg.MODEL.ROI_HEAD.IOU_THRESHOLDS,
                cfg.MODEL.ROI_HEAD.IOU_LABELS,
                allow_low_quality_matches=False,
            )

        self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=tuple(1.0 / input_shape[k].stride for k in self.in_features),
            sampling_ratio=cfg.MODEL.ROI_HEAD.BOX_POOLER_SAMPLING_RATIO,
            pooler_type=cfg.MODEL.ROI_HEAD.BOX_POOLER_TYPE,
        )

        self.box_head = BoxHead(
            cfg, ShapeSpec(channels=in_channels[0], height=pooler_resolution, width=pooler_resolution)
        )
        # self.box_predictor = FastRCNNOutputLayers(cfg, self.box_head.output_shape)
        
        self.train_on_pred_boxes = cfg.MODEL.ROI_HEAD.TRAIN_ON_PRED_BOXES


        self.box2box_transform = Box2BoxTransform(weights=cfg.MODEL.ROI_HEAD.BBOX_REG_WEIGHTS)

        input_shape = self.box_head.output_shape
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        self.cls_score = Linear(input_size, self.num_classes + 1)
        num_bbox_reg_classes = 1 if cfg.MODEL.ROI_HEAD.CLS_AGNOSTIC_BBOX_REG else self.num_classes
        box_dim = len(self.box2box_transform.weights)
        self.bbox_pred = Linear(input_size, num_bbox_reg_classes * box_dim)

        loss_weight = cfg.MODEL.ROI_HEAD.BBOX_REG_LOSS_WEIGHT
        self.loss_weight = {"loss_cls": loss_weight, "loss_box_reg": loss_weight}
        self.smooth_l1_beta = cfg.MODEL.ROI_HEAD.BBOX_SMOOTH_L1_BETA

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
        
        features = [features[f] for f in self.in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        
        # predictions = self.box_predictor(box_features)
        if box_features.dim() > 2:
            box_features = torch.flatten(box_features, start_dim=1)
        scores = self.cls_score(box_features)
        proposal_deltas = self.bbox_pred(box_features)
        del box_features

        if self.training:
            losses = self.losses(scores, proposal_deltas, proposals)

            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.predict_boxes_for_gt_classes(scores, proposal_deltas, proposals)
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)

            return proposals, losses

        else:
            pred_instances, _ = self.inference(scores, proposal_deltas, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def losses(
        self, 
        pred_scores : torch.Tensor, 
        pred_deltas : torch.Tensor, 
        proposals : List[Instances]
    ) -> Dict[torch.Tensor, torch.Tensor]:

        gt_boxes = Boxes.cat([p.gt_boxes for p in proposals])
        gt_classes = torch.cat([p.gt_classes for p in proposals], dim=0)
        proposal_boxes = Boxes.cat([p.proposal_boxes for p in proposals])
        num_instances = gt_classes.numel()

        valid_idxs = gt_classes >= 0
        foreground_idxs = torch.nonzero((gt_classes >= 0) & (gt_classes != self.num_classes), as_tuple=True)[0]

        pred_classes = pred_scores.argmax(dim=1)
        fg_gt_classes = gt_classes[foreground_idxs]
        fg_pred_classes = pred_classes[foreground_idxs]

        num_foreground = foreground_idxs.sum()
        num_false_negative = (fg_pred_classes == self.num_classes).nonzero().numel()
        num_accurate = (pred_classes == gt_classes).nonzero().numel()
        fg_num_accurate = (fg_pred_classes == fg_gt_classes).nonzero().numel()

        storage = get_event_storage()
        if num_instances > 0:
            storage.put_scalar("fast_rcnn/cls_accuracy", num_accurate / num_instances)
            if num_foreground > 0:
                storage.put_scalar("fast_rcnn/fg_cls_accuracy", fg_num_accurate / num_foreground)
                storage.put_scalar("fast_rcnn/false_negative", num_false_negative / num_foreground)

        if len(proposals) == 0:
            loss_cls = 0.0 * pred_scores.sum()
            loss_reg = 0.0 * pred_deltas.sum()
        else:
            loss_cls = F.cross_entropy(pred_scores, gt_classes, reduction="mean")
            gt_proposal_deltas = self.box2box_transform.get_deltas(proposal_boxes.tensor, gt_boxes.tensor)

            box_dim = gt_boxes.tensor.size(1)
            fg_gt_classes = gt_classes[foreground_idxs]
            gt_class_cols = box_dim * fg_gt_classes[:, None] + torch.arange(box_dim, device=pred_deltas.device)
            
            print(pred_deltas[foreground_idxs[:, None], gt_class_cols])
            print(gt_proposal_deltas[foreground_idxs])
            loss_reg = smooth_l1_loss(
                pred_deltas[foreground_idxs[:, None], gt_class_cols],
                gt_proposal_deltas[foreground_idxs],
                self.smooth_l1_beta,
                reduction="sum",
            )

        loss_cls *= self.loss_weight.get('loss_cls', 1.0)
        loss_reg *= self.loss_weight.get('loss_reg', 1.0)
        return {"loss_cls": loss_cls, "loss_reg": loss_reg}

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
            match_quality_matrix = pairwise_iou(targets_per_image.gt_boxes, proposals_per_image.proposal_boxes) # (#gt, #proposal)
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            del match_quality_matrix

            # ground truth classes
            has_gt = len(targets_per_image) > 0
            if has_gt:
                gt_classes_i = targets_per_image.gt_classes[matched_idxs]
                # Proposals with label 0 are treated as background.
                gt_classes_i[matched_labels == 0] = self.num_classes
                # Proposals with label -1 are ignored.
                gt_classes_i[matched_labels == -1] = -1
            else:
                gt_classes_i = torch.zeros_like(matched_idxs) + self.num_classes

            # Sampling
            sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
                gt_classes_i, self.batch_size_per_image, self.positive_fraction, self.num_classes
            )
            sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
            proposals_per_image = proposals_per_image[sampled_idxs]

            # ground truth classes with sampling
            proposals_per_image.gt_classes = gt_classes_i[sampled_idxs]

            # ground truth box
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                proposals_per_image.gt_boxes = targets_per_image.gt_boxes[sampled_targets]
            else:
                proposals_per_image.gt_boxes = Boxes(targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4)))

            proposals_with_gt.append(proposals_per_image)

            # Storage with foreground and backgrond samples
            num_bg_samples.append((gt_classes_i == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes_i.numel() - num_bg_samples[-1])

        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt

class BoxHead(nn.Module):
    def __init__(
        self, cfg,
        input_shape : ShapeSpec
    ):

        super().__init__()

        num_conv = cfg.MODEL.ROI_HEAD.BOX_HEAD_NUM_CONV
        conv_dim = cfg.MODEL.ROI_HEAD.BOX_HEAD_CONV_DIM
        num_fc = cfg.MODEL.ROI_HEAD.BOX_HEAD_NUM_FC
        fc_dim = cfg.MODEL.ROI_HEAD.BOX_HEAD_FC_DIM

        conv_dims = [conv_dim] * num_conv
        fc_dims = [fc_dim] * num_fc
        assert len(conv_dims) + len(fc_dims) > 0

        conv_norm = cfg.MODEL.ROI_HEAD.BOX_HEAD_NORM

        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)

        # Conv Subnet
        self.conv_subnet = []
        for k, conv_dim in enumerate(conv_dims):
            conv = Conv2d(
                self._output_size[0],
                conv_dim,
                kernel_size=3,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=F.relu,
            )
            self.add_module("conv{}".format(k + 1), conv)
            self.conv_subnet.append(conv)
            self._output_size = (conv_dim, self._output_size[1], self._output_size[2])

        # FC Subnet
        self.fc_subnet = []
        for k, fc_dim in enumerate(fc_dims):
            fc = Linear(np.prod(self._output_size), fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.fc_subnet.append(fc)
            self._output_size = fc_dim

        # Initialization
        for layer in self.conv_subnet:
            weight_init.c2_msra_fill(layer)
        for layer in self.fc_subnet:
            weight_init.c2_xavier_fill(layer)

    def forward(self, x):
        for layer in self.conv_subnet:
            x = layer(x)
        if len(self.fc_subnet):
            if x.dim() > 2:
                x = torch.flatten(x, start_dim=1)
            for layer in self.fc_subnet:
                x = F.relu(layer(x))
        return x

    @property
    def output_shape(self) -> ShapeSpec:
        o = self._output_size
        if isinstance(o, int):
            return ShapeSpec(channels=o)
        else:
            return ShapeSpec(channels=o[0], height=o[1], width=o[2])
