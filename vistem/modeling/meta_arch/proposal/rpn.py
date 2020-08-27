import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List

from . import PROPOSAL_REGISTRY
from vistem.modeling.meta_arch import DefaultMetaArch
from vistem.modeling import Box2BoxTransform, Matcher
from vistem.modeling.anchors import build_anchor_generator
from vistem.modeling.layers import Conv2d
from vistem.modeling.model_utils import permute_to_N_HWA_K, pairwise_iou

from vistem.structures import ImageList, Boxes
from vistem.utils.losses import smooth_l1_loss
from vistem.utils.event import get_event_storage

@PROPOSAL_REGISTRY.register()
class RPN(DefaultMetaArch):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg)
        self.device = torch.device(cfg.DEVICE)

        self.in_features                = cfg.MODEL.RPN.IN_FEATURES
        self.batch_size_per_image       = cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE
        self.positive_fraction          = cfg.MODEL.RPN.POSITIVE_FRACTION
        
        self.smooth_l1_loss_beta        = cfg.MODEL.RPN.SMOOTH_L1_BETA
        self.box_reg_loss_type          = cfg.MODEL.RPN.BBOX_REG_LOSS_TYPE
        self.loss_weight = {
            'loss_rpn_cls' : cfg.MODEL.RPN.LOSS_WEIGHT,
            'loss_rpn_loc' : cfg.MODEL.RPN.BBOX_REG_LOSS_WEIGHT
        }

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

        # proposals = self.predict_proposals(
        #     anchors, pred_objectness_logits, pred_anchor_deltas, images.image_sizes
        # )
        return losses

    @torch.jit.unused
    def losses(self, gt_classes, gt_anchors_deltas, pred_class_logits, pred_anchor_deltas):
        num_images = len(gt_classes)

        pred_class_logits = torch.cat(pred_class_logits, dim=1).view(-1)
        pred_anchor_deltas = torch.cat(pred_anchor_deltas, dim=1).view(-1, 4)
        gt_classes = gt_classes.flatten()
        gt_anchors_deltas = gt_anchors_deltas.view(-1, 4)

        valid_idxs = gt_classes >= 0
        foreground_idxs = gt_classes == 1
        num_valid = valid_idxs.sum().item()
        num_fg = foreground_idxs.sum().item()

        # storage
        storage = get_event_storage()
        storage.put_scalar("rpn/num_pos_anchors", num_fg / num_images)
        storage.put_scalar("rpn/num_neg_anchors", (num_valid - num_fg) / num_images)

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

            pos_idx, neg_idx = self.subsample_labels(gt_classes_i, self.batch_size_per_image, self.positive_fraction, 0)
            gt_classes_i.fill_(-1)
            gt_classes_i.scatter_(0, pos_idx, 1)
            gt_classes_i.scatter_(0, neg_idx, 0)

            gt_classes.append(gt_classes_i)
            gt_anchors_deltas.append(gt_anchors_reg_deltas_i)

        return torch.stack(gt_classes), torch.stack(gt_anchors_deltas)
        
    def subsample_labels(self, labels: torch.Tensor, num_samples: int, positive_fraction: float, bg_label: int):
        pos = ((labels != -1) & (labels != bg_label))
        if pos.dim()==0 : pos = pos.unsqueeze(0).nonzero().unbind(1)[0]
        else : pos = pos.nonzero().unbind(1)[0]

        neg = (labels == bg_label)
        if neg.dim()==0 : neg = neg.unsqueeze(0).nonzero().unbind(1)[0]
        else : neg = neg.nonzero().unbind(1)[0]

        num_pos = int(num_samples * positive_fraction)
        num_pos = min(pos.numel(), num_pos)

        num_neg = num_samples - num_pos
        num_neg = min(neg.numel(), num_neg)

        perm1 = torch.randperm(pos.numel(), device=pos.device)[:num_pos]
        perm2 = torch.randperm(neg.numel(), device=neg.device)[:num_neg]

        pos_idx = pos[perm1]
        neg_idx = neg[perm2]

        return pos_idx, neg_idx


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
