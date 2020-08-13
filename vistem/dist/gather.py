import pickle
import torch
import torch.distributed as dist

from .comm_tensor import serialize_to_tensor, pad_to_largest_tensor
from . import get_world_size, get_global_gloo_group

__all__ = ['all_gather', 'gather']

def all_gather(data, group=None):
    if get_world_size() == 1 : return [data]
    if group is None : group = get_global_gloo_group()
    if dist.get_world_size(group) == 1 : return [data]

    tensor = serialize_to_tensor(data, group)
    size_list, tensor = pad_to_largest_tensor(tensor, group)
    max_size = max(size_list)

    tensor_list = [
        torch.empty((max_size,), dtype=torch.uint8, device=tensor.device) for _ in size_list
    ]
    dist.all_gather(tensor_list, tensor, group=group)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list

def gather(data, dst=0, group=None):
    if get_world_size() == 1 : return [data]
    if group is None : group = get_global_gloo_group()
    if dist.get_world_size(group=group) == 1 : return [data]

    rank = dist.get_rank(group=group)
    if rank == dst:
        tensor = serialize_to_tensor(data, group)
        size_list, tensor = pad_to_largest_tensor(tensor, group)
        max_size = max(size_list)

        tensor_list = [
            torch.empty((max_size,), dtype=torch.uint8, device=tensor.device) for _ in size_list
        ]
        dist.gather(tensor, tensor_list, dst=dst, group=group)

        data_list = []
        for size, tensor in zip(size_list, tensor_list):
            buffer = tensor.cpu().numpy().tobytes()[:size]
            data_list.append(pickle.loads(buffer))
        return data_list

    else:
        tensor = serialize_to_tensor(data, group)
        size_list, tensor = pad_to_largest_tensor(tensor, group)
        dist.gather(tensor, [], dst=dst, group=group)
        return []
