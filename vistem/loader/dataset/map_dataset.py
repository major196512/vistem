import torch
from torch.utils.data import Dataset

import random

from vistem.utils.logger import setup_logger

__all__ = ['MapDataset']

class MapDataset(Dataset):
    def __init__(self, dataset, map_func):
        self._dataset = dataset
        # self._map_func = PicklableWrapper(map_func)  # wrap so that a lambda will work
        self._map_func = map_func

        self._rng = random.Random(42)
        self._fallback_candidates = set(range(len(dataset)))

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        retry_count = 0
        cur_idx = int(idx)

        while True:
            data = self._map_func(self._dataset[cur_idx])
            if data is not None:
                self._fallback_candidates.add(cur_idx)
                return data

            # _map_func fails for this idx, use a random new index from the pool
            retry_count += 1
            self._fallback_candidates.discard(cur_idx)
            cur_idx = self._rng.sample(self._fallback_candidates, k=1)[0]

            if retry_count >= 3:
                logger = setup_logger()
                logger.warning(f"Failed to apply `_map_func` for idx: {idx}, retry count: {retry_count}")
