import torch
from torch.utils.data import Dataset

__all__ = ['DictionaryDataset']

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
