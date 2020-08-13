import torch.distributed as dist
import functools

__all__ = ['get_global_gloo_group']

@functools.lru_cache()
def get_global_gloo_group():
    if dist.get_backend() == "nccl":
        return dist.new_group(backend="gloo")
    else:
        return dist.group.WORLD
