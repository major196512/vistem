import torch

from vistem.utils.logger import setup_logger
from vistem.checkpointer.missing import (
    get_missing_parameters_message,
    get_unexpected_parameters_message,
)

from .detectron_caffe import convert_c2_detectron_names
from .efficientnet_pytorch import convert_efficientnet_names

# Note the current matching is not symmetric.
# it assumes model_state_dict will have longer names.
def align_and_update_state_dicts(model_state_dict, ckpt_state_dict, c2_conversion):
    logger = setup_logger(__name__)
    
    model_keys = sorted(list(model_state_dict.keys()))
    if c2_conversion=='Caffe2':
        ckpt_state_dict, original_keys = convert_c2_detectron_names(ckpt_state_dict)
        # original_keys: the name in the original dict (before renaming)
    elif c2_conversion=='lukemelas':
        ckpt_state_dict, original_keys = convert_efficientnet_names(ckpt_state_dict)
    else:
        original_keys = {x: x for x in ckpt_state_dict.keys()}
    ckpt_keys = sorted(list(ckpt_state_dict.keys()))

    def match(a, b):
        # Matched ckpt_key should be a complete (starts with '.') suffix.
        # For example, roi_heads.mesh_head.whatever_conv1 does not match conv1,
        # but matches whatever_conv1 or mesh_head.whatever_conv1.
        return a == b or a.endswith("." + b)

    # get a matrix of string matches, where each (i, j) entry correspond to the size of the
    # ckpt_key string, if it matches
    match_matrix = [len(j) if match(i, j) else 0 for i in model_keys for j in ckpt_keys]
    match_matrix = torch.as_tensor(match_matrix).view(len(model_keys), len(ckpt_keys))
    # use the matched one with longest size in case of multiple matches
    max_match_size, idxs = match_matrix.max(1)
    # remove indices that correspond to no-match
    idxs[max_match_size == 0] = -1

    # used for logging
    max_len_model = max(len(key) for key in model_keys) if model_keys else 1
    max_len_ckpt = max(len(key) for key in ckpt_keys) if ckpt_keys else 1
    log_str_template = "{: <{}} loaded from {: <{}} of shape {}"
    # matched_pairs (matched checkpoint key --> matched model key)
    matched_keys = {}
    msg = ''
    for idx_model, idx_ckpt in enumerate(idxs.tolist()):
        if idx_ckpt == -1:
            continue
        key_model = model_keys[idx_model]
        key_ckpt = ckpt_keys[idx_ckpt]
        value_ckpt = ckpt_state_dict[key_ckpt]
        shape_in_model = model_state_dict[key_model].shape

        if shape_in_model != value_ckpt.shape:
            logger.warning(
                "Shape of {} in checkpoint is {}, while shape of {} in model is {}.".format(
                    key_ckpt, value_ckpt.shape, key_model, shape_in_model
                )
            )
            logger.warning(
                "{} will not be loaded. Please double check and see if this is desired.".format(
                    key_ckpt
                )
            )
            continue

        model_state_dict[key_model] = value_ckpt.clone()
        if key_ckpt in matched_keys:  # already added to matched_keys
            logger.error(
                "Ambiguity found for {} in checkpoint!"
                "It matches at least two keys in the model ({} and {}).".format(
                    key_ckpt, key_model, matched_keys[key_ckpt]
                )
            )
            raise ValueError("Cannot match one checkpoint key to multiple keys in the model.")

        matched_keys[key_ckpt] = key_model
        log_str = log_str_template.format(
                        key_model,
                        max_len_model,
                        original_keys[key_ckpt],
                        max_len_ckpt,
                        tuple(shape_in_model),
                    )
        msg += f'\n{log_str}'

    logger.debug(f'align and update state dicts{msg}')
    matched_model_keys = matched_keys.values()
    matched_ckpt_keys = matched_keys.keys()
    # print warnings about unmatched keys on both side
    unmatched_model_keys = [k for k in model_keys if k not in matched_model_keys]
    if len(unmatched_model_keys):
        logger.debug(get_missing_parameters_message(unmatched_model_keys))

    unmatched_ckpt_keys = [k for k in ckpt_keys if k not in matched_ckpt_keys]
    if len(unmatched_ckpt_keys):
        logger.debug(
            get_unexpected_parameters_message(original_keys[x] for x in unmatched_ckpt_keys)
        )
