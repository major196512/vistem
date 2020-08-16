from termcolor import colored
from collections import defaultdict
from typing import Dict, List

__all__ = ['get_missing_parameters_message', 'get_unexpected_parameters_message']

def get_missing_parameters_message(keys: List[str]) -> str:
    groups = _group_checkpoint_keys(keys)
    msg = "Some model parameters or buffers are not found in the checkpoint:\n"
    msg += "\n".join(f"  {colored(f'{k}{_group_to_str(v)}', 'cyan')}" for k, v in groups.items())
    return msg

def get_unexpected_parameters_message(keys: List[str]) -> str:
    groups = _group_checkpoint_keys(keys)
    msg = "The checkpoint state_dict contains keys that are not used by the model:\n"
    msg += "\n".join(f"  {colored(f'{k}{_group_to_str(v)}', 'magenta')}" for k, v in groups.items())
    return msg

def _group_checkpoint_keys(keys: List[str]) -> Dict[str, List[str]]:
    groups = defaultdict(list)
    for key in keys:
        pos = key.rfind(".")
        if pos >= 0:
            head, tail = key[:pos], [key[pos + 1 :]]
        else:
            head, tail = key, []
        groups[head].extend(tail)
    return groups

def _group_to_str(group: List[str]) -> str:
    if len(group) == 0:
        return ""

    if len(group) == 1:
        return "." + group[0]

    return ".{" + ", ".join(group) + "}"