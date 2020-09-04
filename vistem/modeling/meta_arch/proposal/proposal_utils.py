import torch
import math
from typing import List

from vistem.structures import Instances, Boxes

__all__ = ['add_ground_truth_to_proposals']

def add_ground_truth_to_proposals(
    gt_boxes : List[Boxes], 
    proposals : List[Instances]
    ) -> List[Instances]:

    assert gt_boxes is not None

    assert len(proposals) == len(gt_boxes)
    if len(proposals) == 0:
        return proposals

    ret = []
    for gt_boxes_i, proposals_i in zip(gt_boxes, proposals):
        ret.append(add_ground_truth_to_proposals_single_image(gt_boxes_i, proposals_i))
    
    return ret

def add_ground_truth_to_proposals_single_image(
    gt_boxes : Boxes, 
    proposals : Instances
    ) -> Instances:

    device = proposals.proposal_scores.device

    # Assign all ground-truth boxes an objectness logit corresponding to
    # P(object) = sigmoid(logit) =~ 1.
    gt_logit_value = math.log((1.0 - 1e-10) / (1 - (1.0 - 1e-10)))
    gt_logits = gt_logit_value * torch.ones(len(gt_boxes), device=device)

    gt_proposal = Instances(proposals.image_size)
    gt_proposal.proposal_boxes = gt_boxes
    gt_proposal.proposal_scores = gt_logits
    new_proposals = Instances.cat([proposals, gt_proposal])

    return new_proposals
