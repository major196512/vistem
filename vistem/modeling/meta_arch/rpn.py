import torch
import torch.nn as nn

from . import META_ARCH_REGISTRY, DefaultMetaArch
from vistem.modeling import detector_postprocess
from vistem.modeling.backbone import build_backbone
from vistem.modeling.meta_arch.proposal import build_proposal_generator
from vistem.structures import ImageList

@META_ARCH_REGISTRY.register()
class ProposalNetwork(DefaultMetaArch):
    def __init__(self, cfg):
        super().__init__(cfg)

        # Backbone Network
        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())

    def forward(self, batched_inputs):
        images, gt_instances, _ = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        
        # In training, the proposals are not useful at all but we generate them anyway.
        # This makes RPN-only models about 5% slower.
        if self.training:
            return proposal_losses

        else:
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(proposals, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"proposals": r})

            return processed_results
