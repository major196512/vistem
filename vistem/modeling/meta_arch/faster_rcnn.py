import torch
import torch.nn as nn

from . import META_ARCH_REGISTRY, DefaultMetaArch
from vistem.modeling import detector_postprocess
from vistem.modeling.backbone import build_backbone
from vistem.modeling.meta_arch.proposal import build_proposal_generator
from vistem.modeling.meta_arch.roi_head import build_roi_head

from vistem.utils.event import get_event_storage

__all__ = ['FasterRCNN']

@META_ARCH_REGISTRY.register()
class FasterRCNN(DefaultMetaArch):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.in_features              = cfg.META_ARCH.RPN.IN_FEATURES

        # Backbone Network
        self.backbone = build_backbone(cfg)
        # for feat in self.in_features:
        #     assert feat in self.backbone.out_features, f"'{feat}' is not in backbone({self.backbone.out_features})"

        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
        self.roi_heads = build_roi_head(cfg, self.backbone.output_shape())

    def forward(self, batched_inputs):
        images, gt_instances, proposals = self.preprocess_image(batched_inputs)

        features = self.backbone(images.tensor)
        if proposals is None :
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            proposal_losses = {}

        results, detector_losses = self.roi_heads(images, features, proposals, gt_instances)

        if self.training:
            losses = {}
            losses.update(proposal_losses)
            losses.update(detector_losses)

            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    self.visualize_training(batched_inputs, results)

            return losses

        else:
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})

            return processed_results
