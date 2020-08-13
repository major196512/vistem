import torch
from torch.utils.data import Dataset

__all__ = ['ListDataset']

class ListDataset(Dataset):
    def __init__(self, cfg, data):
        self._data = data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]
