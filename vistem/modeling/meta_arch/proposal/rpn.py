import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Tuple

from . import PROPOSAL_REGISTRY
from vistem.modeling.meta_arch import DefaultMetaArch

from vistem.modeling import Box2BoxTransform, Matcher, subsample_labels
from vistem.modeling.anchors import build_anchor_generator
from vistem.modeling.layers import Conv2d, batched_nms
from vistem.modeling.model_utils import permute_to_N_HWA_K, pairwise_iou

from vistem.structures import ImageList, Boxes, Instances
from vistem.utils.losses import smooth_l1_loss
from vistem.utils.event import get_event_storage

@PROPOSAL_REGISTRY.register()
class RPN(DefaultMetaArch):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg)
        self.in_features                = cfg.MODEL.RPN.IN_FEATURES
        self.batch_size_per_image       = cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE
        self.positive_fraction          = cfg.MODEL.RPN.POSITIVE_FRACTION
        
        self.smooth_l1_loss_beta        = cfg.MODEL.RPN.SMOOTH_L1_BETA
        self.box_reg_loss_type          = cfg.MODEL.RPN.BBOX_REG_LOSS_TYPE
        self.loss_weight = {
            'loss_rpn_cls' : cfg.MODEL.RPN.LOSS_WEIGHT,
            'loss_rpn_loc' : cfg.MODEL.RPN.BBOX_REG_LOSS_WEIGHT
        }

        self.nms_threshold              = cfg.MODEL.RPN.NMS_THRESH
        self.pre_nms_topk               = {True : cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN, False : cfg.MODEL.RPN.PRE_NMS_TOPK_TEST}
        self.post_nms_topk              = {True : cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN, False : cfg.MODEL.RPN.POST_NMS_TOPK_TEST}
        self.min_box_size               = cfg.MODEL.RPN.MIN_SIZE

        if cfg.MODEL.RPN.HEAD_NAME == 'StandardRPNHead' :
            self.rpn_head = StandardRPNHead(cfg, [input_shape[f] for f in self.in_features])
        else:
            raise ValueError(f"Invalid rpn head class '{cfg.MODEL.RPN.HEAD_NAME}'")

        self.anchor_generator = build_anchor_generator(cfg, [input_shape[f] for f in self.in_features])
        self.box2box_transform = Box2BoxTransform(weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS)
        self.matcher = Matcher(
            cfg.MODEL.RPN.IOU_THRESHOLDS,
            cfg.MODEL.RPN.IOU_LABELS,
            allow_low_quality_matches=True,
        )

    def forward(self, images, features, gt_instances):
        features = [features[f] for f in self.in_features]
        anchors = self.anchor_generator(features)
        box_cls, box_delta = self.rpn_head(features)

        box_cls = [permute_to_N_HWA_K(x, 1) for x in box_cls]
        box_delta = [permute_to_N_HWA_K(x, 4) for x in box_delta]

        if self.training:
            gt_classes, gt_anchors_reg_deltas = self.get_ground_truth(anchors, gt_instances)
            losses =  self.losses(gt_classes, gt_anchors_reg_deltas, box_cls, box_delta)
        else:
            losses = {}

        proposals = self.inference(anchors, box_cls, box_delta, images.image_sizes)
        return proposals, losses

    @torch.jit.unused
    def losses(self, gt_classes, gt_anchors_deltas, pred_class_logits, pred_anchor_deltas):
        num_images = len(gt_classes)

        pred_class_logits = torch.cat(pred_class_logits, dim=1).view(-1)
        pred_anchor_deltas = torch.cat(pred_anchor_deltas, dim=1).view(-1, 4)
        gt_classes = gt_classes.flatten()
        gt_anchors_deltas = gt_anchors_deltas.view(-1, 4)

        valid_idxs = gt_classes >= 0
        foreground_idxs = gt_classes == 1

        # storage
        num_valid = valid_idxs.sum().item()
        num_foreground = foreground_idxs.sum().item()
        num_pred_fg = (pred_class_logits[valid_idxs] > 0).sum().item()
        
        storage = get_event_storage()
        storage.put_scalar("rpn/num_pos_anchors", num_foreground / num_images)
        storage.put_scalar("rpn/num_neg_anchors", (num_valid - num_foreground) / num_images)
        if num_foreground > 0 : storage.put_scalar("rpn/recall", (pred_class_logits[foreground_idxs] > 0).nonzero().numel() / num_foreground)
        if num_pred_fg > 0 : storage.put_scalar("rpn/precision", (pred_class_logits[foreground_idxs] > 0).nonzero().numel() / num_pred_fg)
            

        # logits loss
        loss_cls = F.binary_cross_entropy_with_logits(
            pred_class_logits[valid_idxs],
            gt_classes[valid_idxs].to(torch.float32),
            reduction="sum",
        )

        # regression loss
        if self.box_reg_loss_type == 'smooth_l1':
            loss_box_reg = smooth_l1_loss(
                pred_anchor_deltas[foreground_idxs],
                gt_anchors_deltas[foreground_idxs],
                beta=self.smooth_l1_loss_beta,
                reduction="sum",
            )
        else:
            raise ValueError(f"Invalid rpn box reg loss type '{self.box_reg_loss_type}'")

        normalizer = self.batch_size_per_image * num_images
        losses = {
            "loss_rpn_cls": loss_cls / normalizer,
            "loss_rpn_loc": loss_box_reg / normalizer,
        }
        losses = {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}
        return losses

    @torch.jit.unused
    @torch.no_grad()
    def get_ground_truth(self, anchors, targets):
        gt_classes = []
        gt_anchors_deltas = []
        anchors = [Boxes.cat(anchors_i) for anchors_i in anchors]
        # list[Tensor(R, 4)], one for each image

        for anchors_per_image, targets_per_image in zip(anchors, targets):
            match_quality_matrix = pairwise_iou(targets_per_image.gt_boxes, anchors_per_image)
            gt_matched_idxs, gt_classes_i = self.matcher(match_quality_matrix)
            # gt_classes_i = gt_classes_i.to(device=targets.gt_boxes.device)
            del match_quality_matrix

            # ground truth box regression
            matched_gt_boxes = targets_per_image[gt_matched_idxs].gt_boxes
            gt_anchors_reg_deltas_i = self.box2box_transform.get_deltas(
                anchors_per_image.tensor, matched_gt_boxes.tensor
            )

            # if self.anchor_boundary_thresh >= 0:
            #     # Discard anchors that go out of the boundaries of the image
            #     # NOTE: This is legacy functionality that is turned off by default in Detectron2
            #     anchors_inside_image = anchors_per_image.inside_box(image_size_i, self.anchor_boundary_thresh)
            #     gt_classes_i[~anchors_inside_image] = -1

            pos_idx, neg_idx = subsample_labels(gt_classes_i, self.batch_size_per_image, self.positive_fraction, 0)
            gt_classes_i.fill_(-1)
            gt_classes_i.scatter_(0, pos_idx, 1)
            gt_classes_i.scatter_(0, neg_idx, 0)

            gt_classes.append(gt_classes_i)
            gt_anchors_deltas.append(gt_anchors_reg_deltas_i)

        return torch.stack(gt_classes), torch.stack(gt_anchors_deltas)

    # TODO: use torch.no_grad when torchscript supports it.
    # https://github.com/pytorch/pytorch/pull/41371
    def inference(self, anchors, box_cls, box_delta, image_sizes):
        results : List[Instances] = []
        for img_idx, image_size in enumerate(image_sizes):
            pred_cls_per_image = [x[img_idx].detach() for x in box_cls]
            pred_delta_per_image = [x[img_idx].detach() for x in box_delta]
            results_per_image = self.inference_single_image(
                anchors[img_idx], pred_cls_per_image, pred_delta_per_image, tuple(image_size)
            )
            results.append(results_per_image)

        return results

            
    def inference_single_image(self, anchors, box_cls, box_delta, image_size):
        boxes_all = []
        scores_all = []
        feat_lvl_all = []

        # Iterate over every feature level
        for lvl_id, (box_cls_i, box_reg_i, anchors_i) in enumerate(zip(box_cls, box_delta, anchors)):
            box_cls_i = box_cls_i.flatten()

            # Keep top k top scoring indices only.
            num_topk = min(self.pre_nms_topk[int(self.training)], box_reg_i.size(0))
            # torch.sort is actually faster than .topk (at least on GPUs)
            predicted_prob, topk_idxs = box_cls_i.sort(descending=True)
            predicted_prob = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]

            feat_lvl = torch.full((num_topk,), lvl_id, dtype=torch.int64, device=self.device)

            # predict boxes
            box_reg_i = box_reg_i[topk_idxs]
            anchors_i = anchors_i[topk_idxs]
            boxes = self.box2box_transform.apply_deltas(box_reg_i, anchors_i.tensor)
            predicted_boxes = Boxes(boxes)

            valid_mask = torch.isfinite(predicted_boxes.tensor).all(dim=1) & torch.isfinite(predicted_prob)
            if not valid_mask.all():
                if self.training:
                    raise FloatingPointError(
                        "Predicted boxes or scores contain Inf/NaN. Training has diverged."
                    )
                predicted_boxes = predicted_boxes[valid_mask]
                predicted_prob = predicted_prob[valid_mask]
                feat_lvl = feat_lvl[valid_mask]
            predicted_boxes.clip(image_size)

            # filter empty boxes
            keep = predicted_boxes.nonempty(threshold=self.min_box_size)
            if keep.sum().item() != len(boxes):
                predicted_boxes, predicted_prob, feat_lvl = predicted_boxes[keep], predicted_prob[keep], feat_lvl[keep]

            boxes_all.append(predicted_boxes)
            scores_all.append(predicted_prob)
            feat_lvl_all.append(feat_lvl)

        scores_all, feat_lvl_all = [
            torch.cat(x) for x in [scores_all, feat_lvl_all]
        ]
        boxes_all = Boxes.cat(boxes_all)

        keep = batched_nms(boxes_all.tensor, scores_all, feat_lvl_all, self.nms_threshold)
        keep = keep[:self.post_nms_topk[int(self.training)]]

        result = Instances(image_size)
        result.proposal_boxes = boxes_all[keep]
        result.proposal_scores = scores_all[keep]

        return result

class StandardRPNHead(nn.Module):
    def __init__(self, cfg, input_shape):
        """
        NOTE: this interface is experimental.
        """
        super().__init__()

        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]

        anchor_generator    = build_anchor_generator(cfg, input_shape)
        num_anchors         = anchor_generator.num_cell_anchors
        assert (
            len(set(num_anchors)) == 1
        ), "Each level must have the same number of anchors per spatial position"

        num_anchors         = num_anchors[0]
        box_dim             = anchor_generator.box_dim
        
        self.conv = Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, activation=F.relu)
        self.cls_subnet = Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_subnet = Conv2d(in_channels, num_anchors * box_dim, kernel_size=1, stride=1)

        for l in [self.conv, self.cls_subnet, self.bbox_subnet]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)

    def forward(self, features: List[torch.Tensor]):
        logits = []
        bbox_reg = []
        for x in features:
            t = self.conv(x)
            logits.append(self.cls_subnet(t))
            bbox_reg.append(self.bbox_subnet(t))
        return logits, bbox_reg
