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

from vistem.utils.registry import Registry
PLAN_REGISTRY = Registry("PLAN")

from .mode1 import *
from .mode2 import *
from .mode3 import *
from .mode4 import *
from .mode5 import *
from .mode6 import *
from .mode7 import *

def plan_mode(mode, top_feat, bot_feat, out_channels, plan_cfg):
    return PLAN_REGISTRY.get(mode)(top_feat, bot_feat, out_channels, plan_cfg)
