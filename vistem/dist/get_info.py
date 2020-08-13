import torch.distributed as dist

__all__ = ['get_world_size', 'get_rank', 'get_local_rank', 'get_local_size', 'is_main_process']

def get_world_size() -> int:
    if not dist.is_available() : return 1
    if not dist.is_initialized() : return 1
    return dist.get_world_size()

def get_rank() -> int:
    if not dist.is_available() : return 0
    if not dist.is_initialized() : return 0
    return dist.get_rank()

def get_local_rank() -> int:
    from . import _LOCAL_PROCESS_GROUP
    if not dist.is_available() : return 0
    if not dist.is_initialized() : return 0
    assert _LOCAL_PROCESS_GROUP is not None
    return dist.get_rank(group=_LOCAL_PROCESS_GROUP)

def get_local_size() -> int:
    from . import _LOCAL_PROCESS_GROUP
    if not dist.is_available() : return 1
    if not dist.is_initialized() : return 1
    assert _LOCAL_PROCESS_GROUP is not None
    return dist.get_world_size(group=_LOCAL_PROCESS_GROUP)

def is_main_process() -> bool:
    return get_rank() == 0
