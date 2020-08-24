import torch
import torch.nn as nn

import numpy as np
import math
from typing import List

from . import META_ARCH_REGISTRY
from vistem.modeling import Box2BoxTransform, Matcher, detector_postprocess
from vistem.modeling.backbone import build_backbone
from vistem.modeling.anchors import build_anchor_generator
from vistem.modeling.layers import Conv2d, batched_nms

from vistem.structures import ImageList, ShapeSpec, Boxes, Instances
from vistem.utils.losses import sigmoid_focal_loss_jit, smooth_l1_loss

from vistem.utils.event import get_event_storage
from vistem.utils.visualizer import Visualizer, convert_image_to_rgb

__all__ = ['Retinanet']

def pairwise_iou(boxes1: Boxes, boxes2: Boxes) -> torch.Tensor:
    area1 = boxes1.area()
    area2 = boxes2.area()

    boxes1, boxes2 = boxes1.tensor, boxes2.tensor

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    # handle empty boxes
    iou = torch.where(
        inter > 0,
        inter / (area1[:, None] + area2 - inter),
        torch.zeros(1, dtype=inter.dtype, device=inter.device),
    )
    return iou

def permute_to_N_HWA_K(tensor, K):
    """
    Transpose/reshape a tensor from (N, (A x K), H, W) to (N, (HxWxA), K)
    """
    assert tensor.dim() == 4, tensor.shape
    N, _, H, W = tensor.shape
    tensor = tensor.view(N, -1, K, H, W)
    tensor = tensor.permute(0, 3, 4, 1, 2)
    tensor = tensor.reshape(N, -1, K)  # Size=(N,HWA,K)
    return tensor

def permute_all_cls_and_box_to_N_HWA_K_and_concat(box_cls, box_delta, num_classes=80):
    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness and the box_delta
    box_cls_flattened = [permute_to_N_HWA_K(x, num_classes) for x in box_cls]
    box_delta_flattened = [permute_to_N_HWA_K(x, 4) for x in box_delta]
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    box_cls = torch.cat(box_cls_flattened, dim=1).view(-1, num_classes)
    box_delta = torch.cat(box_delta_flattened, dim=1).view(-1, 4)
    return box_cls, box_delta


@META_ARCH_REGISTRY.register()
class RetinaNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.DEVICE)

        self.num_classes              = cfg.MODEL.RETINANET.NUM_CLASSES
        self.in_features              = cfg.MODEL.RETINANET.IN_FEATURES
        # Loss parameters:
        self.focal_loss_alpha         = cfg.MODEL.RETINANET.FOCAL_LOSS_ALPHA
        self.focal_loss_gamma         = cfg.MODEL.RETINANET.FOCAL_LOSS_GAMMA
        self.smooth_l1_loss_beta      = cfg.MODEL.RETINANET.SMOOTH_L1_LOSS_BETA
        # Inference parameters:
        self.score_threshold          = cfg.MODEL.RETINANET.SCORE_THRESH_TEST
        self.topk_candidates          = cfg.MODEL.RETINANET.TOPK_CANDIDATES_TEST
        self.nms_threshold            = cfg.MODEL.RETINANET.NMS_THRESH_TEST
        self.max_detections_per_image = cfg.TEST.DETECTIONS_PER_IMAGE
        # Vis parameters
        self.vis_period               = cfg.VIS_PERIOD
        self.input_format             = cfg.INPUT.FORMAT
        
        self.backbone = build_backbone(cfg)
        for feat in self.in_features:
            assert feat in self.backbone.out_features, f"'{feat}' is not in backbone({self.backbone.out_features})"

        backbone_shape = self.backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in self.in_features]
        self.head = RetinaNetHead(cfg, feature_shapes)
        self.anchor_generator = build_anchor_generator(cfg, feature_shapes)

        # Matching and loss
        self.box2box_transform = Box2BoxTransform(weights=cfg.MODEL.RETINANET.BBOX_REG_WEIGHTS)
        self.matcher = Matcher(
            cfg.MODEL.RETINANET.IOU_THRESHOLDS,
            cfg.MODEL.RETINANET.IOU_LABELS,
            allow_low_quality_matches=True,
        )
        
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        self.loss_normalizer = 100  # initialize with any reasonable #fg that's not too small
        self.loss_normalizer_momentum = 0.9
        self.to(self.device)

    def visualize_training(self, batched_inputs, results):
        assert len(batched_inputs) == len(results)
        storage = get_event_storage()
        max_boxes = 20

        img = batched_inputs[0]["image"]
        img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
        v_gt = Visualizer(img)
        v_gt = v_gt.overlay_instances(boxes=batched_inputs[0]["instances"].gt_boxes)
        anno_img = v_gt.get_image()
        processed_results = detector_postprocess(results[0], img.shape[0], img.shape[1])
        predicted_boxes = processed_results.pred_boxes.tensor.detach().cpu().numpy()

        v_pred = Visualizer(img)
        v_pred = v_pred.overlay_instances(boxes=predicted_boxes[0:max_boxes])
        prop_img = v_pred.get_image()
        vis_img = np.vstack((anno_img, prop_img))
        vis_img = vis_img.transpose(2, 0, 1)
        vis_name = f"Top: GT bounding boxes; Bottom: {max_boxes} Highest Scoring Results"
        storage.put_image(vis_name, vis_img)

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

        features = self.backbone(images.tensor)
        features = [features[f] for f in self.in_features]
        anchors = self.anchor_generator(features)
        box_cls, box_delta = self.head(features)

        if self.training:
            gt_classes, gt_anchors_reg_deltas = self.get_ground_truth(anchors, gt_instances)
            losses =  self.losses(gt_classes, gt_anchors_reg_deltas, box_cls, box_delta)

            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    box_cls = [permute_to_N_HWA_K(x, self.num_classes) for x in box_cls]
                    box_delta = [permute_to_N_HWA_K(x, 4) for x in box_delta]

                    results = self.inference(anchors, box_cls, box_delta, images.image_sizes)
                    self.visualize_training(batched_inputs, results)

            return losses

        else:
            box_cls = [permute_to_N_HWA_K(x, self.num_classes) for x in box_cls]
            box_delta = [permute_to_N_HWA_K(x, 4) for x in box_delta]

            results = self.inference(anchors, box_cls, box_delta, images.image_sizes)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})

            return processed_results

    def losses(self, gt_classes, gt_anchors_deltas, pred_class_logits, pred_anchor_deltas):
        pred_class_logits, pred_anchor_deltas = permute_all_cls_and_box_to_N_HWA_K_and_concat(
            pred_class_logits, pred_anchor_deltas, self.num_classes
        )  # Shapes: (N x R, K) and (N x R, 4), respectively.

        gt_classes = gt_classes.flatten()
        gt_anchors_deltas = gt_anchors_deltas.view(-1, 4)

        valid_idxs = gt_classes >= 0
        foreground_idxs = (gt_classes >= 0) & (gt_classes != self.num_classes)
        num_foreground = foreground_idxs.sum()

        self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (
            1 - self.loss_normalizer_momentum
        ) * max(num_foreground, 1)

        gt_classes_target = torch.zeros_like(pred_class_logits)
        gt_classes_target[foreground_idxs, gt_classes[foreground_idxs]] = 1

        # logits loss
        loss_cls = sigmoid_focal_loss_jit(
            pred_class_logits[valid_idxs],
            gt_classes_target[valid_idxs],
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        ) / self.loss_normalizer

        # regression loss
        loss_box_reg = smooth_l1_loss(
            pred_anchor_deltas[foreground_idxs],
            gt_anchors_deltas[foreground_idxs],
            beta=self.smooth_l1_loss_beta,
            reduction="sum",
        ) / self.loss_normalizer

        return {"loss_cls": loss_cls, "loss_box_reg": loss_box_reg}

    @torch.no_grad()
    def get_ground_truth(self, anchors, targets):
        gt_classes = []
        gt_anchors_deltas = []
        anchors = [Boxes.cat(anchors_i) for anchors_i in anchors]
        # list[Tensor(R, 4)], one for each image

        for anchors_per_image, targets_per_image in zip(anchors, targets):
            match_quality_matrix = pairwise_iou(targets_per_image.gt_boxes, anchors_per_image)
            gt_matched_idxs, anchor_labels = self.matcher(match_quality_matrix)

            # ground truth box regression
            matched_gt_boxes = targets_per_image[gt_matched_idxs].gt_boxes
            gt_anchors_reg_deltas_i = self.box2box_transform.get_deltas(
                anchors_per_image.tensor, matched_gt_boxes.tensor
            )

            # ground truth classes
            has_gt = len(targets_per_image) > 0
            if has_gt:
                gt_classes_i = targets_per_image.gt_classes[gt_matched_idxs]
                # Anchors with label 0 are treated as background.
                gt_classes_i[anchor_labels == 0] = self.num_classes
                # Anchors with label -1 are ignored.
                gt_classes_i[anchor_labels == -1] = -1
            else:
                gt_classes_i = torch.zeros_like(gt_matched_idxs) + self.num_classes

            gt_classes.append(gt_classes_i)
            gt_anchors_deltas.append(gt_anchors_reg_deltas_i)

        return torch.stack(gt_classes), torch.stack(gt_anchors_deltas)

    def inference(self, anchors, pred_logits, pred_anchor_deltas, image_sizes):
        results = []
        for img_idx, image_size in enumerate(image_sizes):
            pred_logits_per_image = [x[img_idx] for x in pred_logits]
            deltas_per_image = [x[img_idx] for x in pred_anchor_deltas]
            results_per_image = self.inference_single_image(
                anchors[img_idx], pred_logits_per_image, deltas_per_image, tuple(image_size)
            )
            results.append(results_per_image)
        return results

    def inference_single_image(self, anchors, box_cls, box_delta, image_size):
        boxes_all = []
        scores_all = []
        class_idxs_all = []

        # Iterate over every feature level
        for box_cls_i, box_reg_i, anchors_i in zip(box_cls, box_delta, anchors):
            # (HxWxAxK,)
            box_cls_i = box_cls_i.flatten().sigmoid_()

            # Keep top k top scoring indices only.
            num_topk = min(self.topk_candidates, box_reg_i.size(0))
            # torch.sort is actually faster than .topk (at least on GPUs)
            predicted_prob, topk_idxs = box_cls_i.sort(descending=True)
            predicted_prob = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]

            # filter out the proposals with low confidence score
            keep_idxs = predicted_prob > self.score_threshold
            predicted_prob = predicted_prob[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]

            anchor_idxs = topk_idxs // self.num_classes
            classes_idxs = topk_idxs % self.num_classes

            box_reg_i = box_reg_i[anchor_idxs]
            anchors_i = anchors_i[anchor_idxs]
            # predict boxes
            predicted_boxes = self.box2box_transform.apply_deltas(box_reg_i, anchors_i.tensor)

            boxes_all.append(predicted_boxes)
            scores_all.append(predicted_prob)
            class_idxs_all.append(classes_idxs)

        boxes_all, scores_all, class_idxs_all = [
            torch.cat(x) for x in [boxes_all, scores_all, class_idxs_all]
        ]
        keep = batched_nms(boxes_all, scores_all, class_idxs_all, self.nms_threshold)
        keep = keep[: self.max_detections_per_image]

        result = Instances(image_size)
        result.pred_boxes = Boxes(boxes_all[keep])
        result.scores = scores_all[keep]
        result.pred_classes = class_idxs_all[keep]
        return result

    def preprocess_image(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

class RetinaNetHead(nn.Module):
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()
        # fmt: off
        in_channels      = input_shape[0].channels
        num_classes      = cfg.MODEL.RETINANET.NUM_CLASSES
        num_convs        = cfg.MODEL.RETINANET.NUM_CONVS
        prior_prob       = cfg.MODEL.RETINANET.PRIOR_PROB
        num_anchors      = build_anchor_generator(cfg, input_shape).num_cell_anchors
        # fmt: on
        assert (
            len(set(num_anchors)) == 1
        ), "Using different number of anchors between levels is not currently supported!"
        num_anchors = num_anchors[0]

        cls_subnet = []
        bbox_subnet = []
        for _ in range(num_convs):
            cls_subnet.append(
                Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            cls_subnet.append(nn.ReLU())
            bbox_subnet.append(
                Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            bbox_subnet.append(nn.ReLU())

        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.bbox_subnet = nn.Sequential(*bbox_subnet)
        self.cls_score = Conv2d(
            in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1
        )
        self.bbox_pred = Conv2d(in_channels, num_anchors * 4, kernel_size=3, stride=1, padding=1)

        # Initialization
        for modules in [self.cls_subnet, self.bbox_subnet, self.cls_score, self.bbox_pred]:
            for layer in modules.modules():
                if isinstance(layer, Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)

        # Use prior in model initialization to improve stability
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_score.bias, bias_value)

    def forward(self, features):
        logits = []
        bbox_reg = []
        for feature in features:
            logits.append(self.cls_score(self.cls_subnet(feature)))
            bbox_reg.append(self.bbox_pred(self.bbox_subnet(feature)))
        return logits, bbox_reg