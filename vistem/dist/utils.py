import torch.distributed as dist
import numpy as np

from . import all_gather

def shared_random_seed():
    ints = np.random.randint(2 ** 31)
    all_ints = all_gather(ints)
    return all_ints[0]

def synchronize():
    if not dist.is_available() : return
    if not dist.is_initialized() : return
    world_size = dist.get_world_size()
    if world_size == 1 : return
    dist.barrier()
