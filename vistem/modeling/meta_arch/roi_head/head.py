import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from typing import List, Dict, Tuple, Union, Optional

from . import ROI_REGISTRY
from .pooling import ROIPooler

from vistem.modeling import Box2BoxTransform, Matcher, subsample_labels
from vistem.modeling.model_utils import pairwise_iou
from vistem.modeling.layers import Conv2d, ConvTranspose2d, Linear, batched_nms
from vistem.modeling.layers.norm import get_norm
from vistem.modeling.meta_arch import DefaultMetaArch
from vistem.modeling.meta_arch.proposal.proposal_utils import add_ground_truth_to_proposals

from vistem.structures import ImageList, Instances, ShapeSpec, Boxes
from vistem.utils.losses import smooth_l1_loss
from vistem.utils import get_event_storage
from vistem.utils import weight_init

@ROI_REGISTRY.register()
class StandardROIHeads(DefaultMetaArch):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg)

        self.seg_on                         = cfg.INPUT.SEG_ON

        self.in_features                    = cfg.META_ARCH.ROI.IN_FEATURES
        self.num_classes                    = cfg.META_ARCH.NUM_CLASSES
        in_channels                         = [input_shape[f].channels for f in self.in_features]

        assert len(set(in_channels)) == 1, in_channels
        for feat in self.in_features:
            assert feat in input_shape.keys(), f"'{feat}' is not in backbone({input_shape.keys()})"

        # Matcher
        iou_thres                           = cfg.META_ARCH.ROI.MATCHER.IOU_THRESHOLDS
        iou_labels                          = cfg.META_ARCH.ROI.MATCHER.IOU_LABELS
        allow_low_quality_matches           = cfg.META_ARCH.ROI.MATCHER.LOW_QUALITY_MATCHES
        self.proposal_matcher               = Matcher(iou_thres, iou_labels, allow_low_quality_matches=allow_low_quality_matches)

        # Sampling
        self.proposal_append_gt             = cfg.META_ARCH.ROI.SAMPLING.PROPOSAL_APPEND_GT
        self.batch_size_per_image           = cfg.META_ARCH.ROI.SAMPLING.BATCH_SIZE_PER_IMAGE
        self.positive_fraction              = cfg.META_ARCH.ROI.SAMPLING.POSITIVE_FRACTION

        # Pooling Parameters and Module
        pooler_type                     = cfg.META_ARCH.ROI.POOLING.TYPE
        pooler_resolution               = cfg.META_ARCH.ROI.POOLING.RESOLUTION
        pooler_sampling_ratio           = cfg.META_ARCH.ROI.POOLING.SAMPLING_RATIO

        # ROI Box Head
        self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=tuple(1.0 / input_shape[k].stride for k in self.in_features),
            sampling_ratio=pooler_sampling_ratio,
            pooler_type=pooler_type,
        )
        self.box_head = BoxHead(
            cfg, ShapeSpec(channels=in_channels[0], height=pooler_resolution, width=pooler_resolution)
        )
        # self.train_on_pred_boxes            = cfg.MODEL.ROI_HEAD.TRAIN_ON_PRED_BOXES

        if self.seg_on:
            self.mask_pooler = ROIPooler(
                output_size=pooler_resolution,
                scales=tuple(1.0 / input_shape[k].stride for k in self.in_features),
                sampling_ratio=pooler_sampling_ratio,
                pooler_type=pooler_type,
            )
            self.mask_head = MaskHead(
                cfg, ShapeSpec(channels=in_channels[0])
            )

        # Loss parameters
        self.loss_weight                    = cfg.META_ARCH.ROI.BOX_LOSS.LOSS_WEIGHT
        self.smooth_l1_beta                 = cfg.META_ARCH.ROI.BOX_LOSS.SMOOTH_L1_BETA

        if isinstance(self.loss_weight, float):
            self.loss_weight = {"loss_cls": self.loss_weight, "loss_loc": self.loss_weight}
        assert 'loss_cls' in self.loss_weight
        assert 'loss_loc' in self.loss_weight

        # Inference parameters
        bbox_reg_weights                    = cfg.META_ARCH.ROI.TEST.BBOX_REG_WEIGHTS
        self.box2box_transform              = Box2BoxTransform(weights=bbox_reg_weights)

        self.test_nms_thresh                = cfg.META_ARCH.ROI.TEST.NMS_THRESH
        self.score_threshhold               = cfg.TEST.SCORE_THRESH
        self.max_detections_per_image       = cfg.TEST.DETECTIONS_PER_IMAGE

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:

        if self.training:
            assert targets
            proposals = self.get_ground_truth(proposals, targets)
            del targets
        
        features = [features[f] for f in self.in_features]

        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        scores, proposal_deltas = self.box_head(box_features)
        del box_features

        if self.seg_on:
            mask_features = self.mask_pooler(features, [x.proposal_boxes for x in proposals])
            mask_features = self.mask_head(mask_features)

        if self.training:
            losses = self.losses_box(scores, proposal_deltas, proposals)
            if self.seg_on : losses.update(self.losses())

            # proposals is modified in-place below, so losses must be computed first.
            # if self.train_on_pred_boxes:
            #     with torch.no_grad():
            #         pred_boxes = self.predict_boxes_for_gt_classes(scores, proposal_deltas, proposals)
            #         for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
            #             proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            # return proposals, losses

            if self.vis_period > 0:
                results = self.inference(scores, proposal_deltas, proposals)
                return results, losses
            else : return None, losses

        else:
            results = self.inference(scores, proposal_deltas, proposals)
            # pred_instances = self.forward_with_given_boxes(features, pred_instances)

            return results, {}

    def losses_box(
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

        num_foreground = foreground_idxs.numel()
        num_false_negative = (fg_pred_classes == self.num_classes).nonzero().numel()
        num_accurate = (pred_classes == gt_classes).nonzero().numel()
        fg_num_accurate = (fg_pred_classes == fg_gt_classes).nonzero().numel()

        storage = get_event_storage()
        if num_instances > 0:
            storage.put_scalar("roi_head/cls_accuracy", num_accurate / num_instances)
            if num_foreground > 0:
                storage.put_scalar("roi_head/fg_cls_accuracy", fg_num_accurate / num_foreground)
                storage.put_scalar("roi_head/false_negative", num_false_negative / num_foreground)

        if len(proposals) == 0:
            loss_cls = 0.0 * pred_scores.sum()
            loss_loc = 0.0 * pred_deltas.sum()
        else:
            loss_cls = F.cross_entropy(pred_scores, gt_classes, reduction="mean")
            gt_proposal_deltas = self.box2box_transform.get_deltas(proposal_boxes.tensor, gt_boxes.tensor)

            box_dim = gt_boxes.tensor.size(1)
            fg_gt_classes = gt_classes[foreground_idxs]
            gt_class_cols = box_dim * fg_gt_classes[:, None] + torch.arange(box_dim, device=pred_deltas.device)
            
            loss_loc = smooth_l1_loss(
                pred_deltas[foreground_idxs[:, None], gt_class_cols],
                gt_proposal_deltas[foreground_idxs],
                self.smooth_l1_beta,
                reduction="sum",
            ) / gt_classes.numel()

        loss_cls *= self.loss_weight.get('loss_cls', 1.0)
        loss_loc *= self.loss_weight.get('loss_loc', 1.0)
        return {"loss_cls": loss_cls, "loss_loc": loss_loc}

    def losses_mask(
        self, 
        pred : torch.Tensor,
        proposals : List[Instances],
    ):
        num_masks = pred.size(0)
        mask_side_len = pred.size(2)
        assert pred.size(2) == pred.size(3), "Mask prediction must be square!"

        gt_masks = []
        gt_classes = []
        for proposal_per_image in proposals:
            if len(proposal_per_image) == 0 : continue
            
            gt_class = proposal_per_image.gt_classes.to(dtype=torch.int64)
            gt_mask = proposal_per_image.gt_mask.crop_and_resize(
                proposal_per_image.proposal_boxes.tensor, mask_side_len
            ).to(device=pred.device)

            gt_classes.append(gt_class)
            gt_masks.append(gt_mask)

        if len(gt_masks) == 0 : return pred.sum() * 0

        gt_classes = torch.cat(gt_classes, dim=0)
        gt_masks = torch.cat(gt_masks, dim=0)
        gt_masks = gt_masks.to(dtype=torch.float32)
        gt_masks_bool = (gt_masks > 0.5)

        pred = pred[torch.arange(num_masks), gt_classes]

        mask_loss = F.binary_cross_entropy_with_logits(pred, gt_masks, reduction='mean')

        return mask_loss

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
                if hasattr(targets_per_image, 'gt_mask') : 
                    proposals_per_image.gt_mask = targets_per_image.gt_mask[sampled_targets]
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

    def inference(
        self,
        pred_scores : torch.Tensor, 
        pred_deltas : torch.Tensor, 
        proposals : List[Instances]
    ) -> List[Instances]:

        results = []
        if not len(proposals) : return results
        num_inst_per_image = [len(p) for p in proposals]

        proposal_boxes = [p.proposal_boxes for p in proposals]
        proposal_boxes = proposal_boxes[0].cat(proposal_boxes).tensor
        pred_deltas_per_image = self.box2box_transform.apply_deltas(pred_deltas, proposal_boxes)
        pred_deltas_per_image = pred_deltas_per_image.split(num_inst_per_image)

        pred_scores_per_image = F.softmax(pred_scores, dim=-1)
        pred_scores_per_image = pred_scores_per_image.split(num_inst_per_image)

        image_sizes = [x.image_size for x in proposals]
        for img_idx, image_size in enumerate(image_sizes):
            results_per_image = self.inference_single_image(
                pred_scores_per_image[img_idx], pred_deltas_per_image[img_idx], image_size
            )
            results.append(results_per_image)

        return results

    def inference_single_image(
        self,
        box_cls: torch.Tensor,
        box_delta: torch.Tensor,
        image_size: List[Tuple[int, int]]
    ) -> Instances:
    
        valid_mask = torch.isfinite(box_delta).all(dim=1) & torch.isfinite(box_cls).all(dim=1)
        if not valid_mask.all():
            box_delta = box_delta[valid_mask]
            box_cls = box_cls[valid_mask]

        box_cls = box_cls[:, :-1]
        keep_idxs = box_cls > self.score_threshhold  # R x K
        box_cls = box_cls[keep_idxs]

        num_bbox_reg_classes = box_delta.shape[1] // 4
        # Convert to Boxes to use the `clip` function ...
        box_delta = Boxes(box_delta.reshape(-1, 4))
        box_delta.clip(image_size)
        box_delta = box_delta.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

        filter_inds = keep_idxs.nonzero()
        proposal_idxs = filter_inds[:, 0]
        classes_idxs = filter_inds[:, 1]

        if num_bbox_reg_classes == 1:
            box_delta = box_delta[proposal_idxs, 0]
        else:
            box_delta = box_delta[keep_idxs]
        
        keep = batched_nms(box_delta, box_cls, classes_idxs, self.test_nms_thresh)
        if self.max_detections_per_image >= 0:
            keep = keep[:self.max_detections_per_image]

        result = Instances(image_size)
        result.pred_boxes = Boxes(box_delta[keep])
        result.pred_scores = box_cls[keep]
        result.pred_classes = classes_idxs[keep]
        return result

class BoxHead(nn.Module):
    def __init__(
        self, cfg,
        input_shape : ShapeSpec
    ):

        super().__init__()

        num_classes     = cfg.META_ARCH.NUM_CLASSES

        num_conv        = cfg.META_ARCH.ROI.BOX_HEAD.NUM_CONV
        conv_dim        = cfg.META_ARCH.ROI.BOX_HEAD.CONV_DIM
        conv_norm       = cfg.META_ARCH.ROI.BOX_HEAD.CONV_NORM

        num_fc          = cfg.META_ARCH.ROI.BOX_HEAD.NUM_FC
        fc_dim          = cfg.META_ARCH.ROI.BOX_HEAD.FC_DIM

        conv_dims = [conv_dim] * num_conv
        fc_dims = [fc_dim] * num_fc
        assert len(conv_dims) + len(fc_dims) > 0

        output_size = (input_shape.channels, input_shape.height, input_shape.width)

        # Conv Subnet
        self.conv_subnet = []
        for k, conv_dim in enumerate(conv_dims):
            conv = Conv2d(
                output_size[0],
                conv_dim,
                kernel_size=3,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=F.relu,
            )
            self.add_module("conv{}".format(k + 1), conv)
            self.conv_subnet.append(conv)
            output_size = (conv_dim, output_size[1], output_size[2])

        # FC Subnet
        self.fc_subnet = []
        for k, fc_dim in enumerate(fc_dims):
            fc = Linear(np.prod(output_size), fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.fc_subnet.append(fc)
            output_size = fc_dim

        # Classification and Localization
        if isinstance(output_size, int) : input_size = output_size
        else : input_size = output_size[0] * output_size[1] * output_size[2]
        box_dim = len(cfg.META_ARCH.ROI.TEST.BBOX_REG_WEIGHTS)

        self.cls_score = Linear(input_size, num_classes + 1)
        self.bbox_pred = Linear(input_size, num_classes * box_dim)

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

        scores = self.cls_score(x)
        proposal_deltas = self.bbox_pred(x)

        return scores, proposal_deltas

class MaskHead(nn.Module):
    def __init__(
        self, cfg,
        input_shape : ShapeSpec
    ):

        super().__init__()

        num_classes     = cfg.META_ARCH.NUM_CLASSES

        num_conv        = cfg.META_ARCH.ROI.MASK_HEAD.NUM_CONV
        conv_dim        = cfg.META_ARCH.ROI.MASK_HEAD.CONV_DIM
        conv_norm       = cfg.META_ARCH.ROI.MASK_HEAD.CONV_NORM

        conv_dims = [conv_dim] * num_conv

        output_size = input_shape.channels

        self.conv_subnet = []
        for k, conv_dim in enumerate(conv_dims):
            conv = Conv2d(
                output_size,
                conv_dim,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=F.relu,
            )
            self.add_module(f'conv{k+1}', conv)
            self.conv_subnet.append(conv)
            output_size = conv_dim

        self.deconv = ConvTranspose2d(
            output_size,
            conv_dim,
            kernel_size=2,
            stride=2,
            padding=0,
            activation=F.relu,
        )
        output_size = conv_dim

        self.predictor = Conv2d(
            output_size,
            num_classes,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        # Initialization
        for layer in self.conv_subnet:
            weight_init.c2_msra_fill(layer)
        weight_init.c2_xavier_fill(self.deconv)
        nn.init.normal_(self.predictor.weight, std=0.001)
        if self.predictor.bias is not None:
            nn.init.normal_(self.predictor.bias, 0)

    def forward(self, x):
        for layer in self.conv_subnet:
            x = layer(x)
        x = self.deconv(x)
        x = self.predictor(x)

        return x
