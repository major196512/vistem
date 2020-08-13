import torch.distributed as dist
import datetime

__all__ = ['init_process_group', 'new_group']

def init_process_group(backend, init_method=None, timeout=datetime.timedelta(0, 1800), world_size=-1, rank=-1, store=None, group_name=''):
    return dist.init_process_group(backend, init_method, timeout, world_size, rank, store, group_name)

def new_group(ranks=None, timeout=datetime.timedelta(0, 1800), backend=None):
    return dist.new_group(ranks=ranks, timeout=timeout, backend=backend)
