import torch
from torch.utils.data.sampler import Sampler
import itertools

from vistem import dist

__all__ = ['IterSampler']

class IterSampler(Sampler):
    def __init__(self, cfg, dataset):
        self._size = len(dataset)
        assert self._size > 0
        self._shuffle = cfg.LOADER.TRAIN_SHUFFLE
        if cfg.SEED < 0 : self._seed = int(dist.shared_random_seed())
        else : self._seed = int(cfg.SEED)

        self._rank = dist.get_rank()
        self._world_size = dist.get_world_size()

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)

    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        while True:
            if self._shuffle:
                yield from torch.randperm(self._size, generator=g)
            else:
                yield from torch.arange(self._size)
