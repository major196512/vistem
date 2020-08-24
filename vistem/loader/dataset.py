import torch
from torch.utils.data import Dataset, IterableDataset

import numpy as np
import random
import pickle

from vistem.utils.logger import setup_logger
from vistem.utils.serialize import PicklableWrapper

__all__ = ['ListDataset', 'DictionaryDataset', 'MapDataset', 'AspectRatioGroupedDataset']

_logger = setup_logger(__name__)

class ListDataset(Dataset):
    def __init__(self, cfg, data, copy: bool = True, serialize: bool = True):
        self._data = data
        self._copy = copy
        self._serialize = serialize

        def _serialize(data):
            buffer = pickle.dumps(data, protocol=-1)
            return np.frombuffer(buffer, dtype=np.uint8)

        if self._serialize:
            _logger.info(f"Serializing {len(self._data)} elements to byte tensors and concatenating them all ...")
            self._data = [_serialize(x) for x in self._data]
            self._addr = np.asarray([len(x) for x in self._data], dtype=np.int64)
            self._addr = np.cumsum(self._addr)
            self._data = np.concatenate(self._data)
            _logger.info(f"Serialized dataset takes {(len(self._data) / 1024 ** 2):.2f} MiB")

    def __len__(self):
        if self._serialize:
            return len(self._addr)
        else:
            return len(self._data)

    def __getitem__(self, idx):
        if self._serialize:
            start_addr = 0 if idx == 0 else self._addr[idx - 1].item()
            end_addr = self._addr[idx].item()
            bytes = memoryview(self._data[start_addr:end_addr])
            return pickle.loads(bytes)
        elif self._copy:
            return copy.deepcopy(self._data[idx])
        else:
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
        self._map_func = PicklableWrapper(map_func)  # wrap so that a lambda will work
        # self._map_func = map_func

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
