# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import re
import torch
from typing import List, Dict, Tuple

from vistem.utils.logger import setup_logger

__all__ = ['convert_efficientnet_names']

def convert_efficientnet_names(weights : Dict) -> Tuple[Dict[str, torch.Tensor], Dict[str, str]]:
    logger = setup_logger(__name__)

    logger.debug("Remapping EfficientNet weights ......")
    original_keys = sorted(weights.keys())

    layer_keys = copy.deepcopy(original_keys)
    layer_keys = [re.sub('^_', '', k) for k in layer_keys]

    layer_keys = [re.sub('^conv_stem', 'stem.conv1', k) for k in layer_keys]
    layer_keys = [re.sub('^bn0', 'stem.conv1.norm', k) for k in layer_keys]

    layer_keys = [re.sub('^conv_head', 'head.conv1', k) for k in layer_keys]
    layer_keys = [re.sub('^bn1', 'head.conv1.norm', k) for k in layer_keys]

    layer_keys = [re.sub('^blocks\\.', 'block', k) for k in layer_keys]

    # Expand Convolution
    layer_keys = [k.replace('_expand_conv', 'expand_conv') for k in layer_keys]
    layer_keys = [k.replace('_bn0', 'expand_conv.norm') for k in layer_keys]

    # Depthwise Convolution
    layer_keys = [k.replace('_depthwise_conv', 'depthwise_conv') for k in layer_keys]
    layer_keys = [k.replace('_bn1', 'depthwise_conv.norm') for k in layer_keys]

    # Project Convolution
    layer_keys = [k.replace('_project_conv', 'project_conv') for k in layer_keys]
    layer_keys = [k.replace('_bn2', 'project_conv.norm') for k in layer_keys]

    # Squeeze and Excitation
    layer_keys = [re.sub('_se_reduce', 'SEblock.reduce', k) for k in layer_keys]
    layer_keys = [re.sub('_se_expand', 'SEblock.expand', k) for k in layer_keys]

    layer_keys = [re.sub('^fc', 'linear', k) for k in layer_keys]
    # --------------------------------------------------------------------------
    # Done with replacements
    # --------------------------------------------------------------------------
    assert len(set(layer_keys)) == len(layer_keys)
    assert len(original_keys) == len(layer_keys)

    new_weights = {}
    new_keys_to_original_keys = {}
    for orig, renamed in zip(original_keys, layer_keys):
        new_keys_to_original_keys[renamed] = orig
        if renamed.startswith("bbox_pred.") or renamed.startswith("mask_head.predictor."):
            # remove the meaningless prediction weight for background class
            new_start_idx = 4 if renamed.startswith("bbox_pred.") else 1
            new_weights[renamed] = weights[orig][new_start_idx:]
            logger.debug(
                "Remove prediction weight for background class in {}. The shape changes from "
                "{} to {}.".format(
                    renamed, tuple(weights[orig].shape), tuple(new_weights[renamed].shape)
                )
            )
        elif renamed.startswith("cls_score."):
            # move weights of bg class from original index 0 to last index
            logger.debug(
                "Move classification weights for background class in {} from index 0 to "
                "index {}.".format(renamed, weights[orig].shape[0] - 1)
            )
            new_weights[renamed] = torch.cat([weights[orig][1:], weights[orig][:1]])
        else:
            new_weights[renamed] = weights[orig]

    return new_weights, new_keys_to_original_keys
