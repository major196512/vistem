import torch
import torch.distributed as dist

from . import get_world_size

def reduce_dict(input_dict, average=True):
    world_size = get_world_size()
    if world_size < 2 : return input_dict

    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])

        values = torch.stack(values, dim=0)
        dist.reduce(values, dst=0)

        if average :
            if dist.get_rank() == 0 : values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}

    return reduced_dict
