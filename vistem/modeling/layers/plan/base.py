import torch
import torch.nn as nn
import re

__all__ = ['PLANBase']

class PLANBase(nn.Module):
    def __init__(self):
        super().__init__()

    def decode_cfg(self, cfg):
        self.num_heads = int(re.compile('h[0-9]+').findall(cfg)[0][1:])
        self.hidden_channels = int(re.compile('c[0-9]+').findall(cfg)[0][1:])
        self.erf = int(re.compile('k[0-9]+').findall(cfg)[0][1:])
        self.num_fuse = int(re.compile('f[0-9]+').findall(cfg)[0][1:])