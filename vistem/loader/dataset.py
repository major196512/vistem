import random
import torch
from torch.utils.data import Dataset, IterableDataset

from vistem.utils.logger import setup_logger

__all__ = ['ListDataset', 'DictionaryDataset', 'MapDataset', 'AspectRatioGroupedDataset']

_logger = setup_logger(__name__)

class ListDataset(Dataset):
    def __init__(self, cfg, data):
        self._data = data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]


class DictionaryDataset(Dataset):
    def __init__(self, cfg, data):
        self._data = data
        self._key = list(data.keys())

    def __len__(self):
        return len(self._data[self._key[0]])

    def __getitem__(self, idx):
        ret_dict = dict()
        for key in self._key:
            ret_dict[key] = self._data[key][idx]
        return ret_dict

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
                _logger.warning(f"Failed to apply `_map_func` for idx: {idx}, retry count: {retry_count}")

class AspectRatioGroupedDataset(IterableDataset):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self._buckets = [[] for _ in range(2)]
        # Hard-coded two aspect ratio groups: w > h and w < h.
        # Can add support for more aspect ratio groups, but doesn't seem useful

    def __iter__(self):
        for d in self.dataset:
            w, h = d["width"], d["height"]
            bucket_id = 0 if w > h else 1
            bucket = self._buckets[bucket_id]
            bucket.append(d)
            if len(bucket) == self.batch_size:
                yield bucket[:]
                del bucket[:]
