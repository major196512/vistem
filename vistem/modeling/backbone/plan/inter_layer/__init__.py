import torch

class PLANDecode(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def decode_cfg(self, cfg):
        import re
        self.num_heads = int(re.compile('h[0-9]+').findall(cfg)[0][1:])
        self.key_dim = int(re.compile('k[0-9]+').findall(cfg)[0][1:])
        self.value_dim = int(re.compile('v[0-9]+').findall(cfg)[0][1:])
        self.erf = int(re.compile('e[0-9]+').findall(cfg)[0][1:])

from .build import InterLayer