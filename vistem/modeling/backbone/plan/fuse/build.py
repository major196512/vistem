import torch
import torch.nn as nn

from .SENet import Fuse_SENet
from .summation import Fuse_Summation

__all__ = ['PLANFuse']

def PLANFuse(fuse_mode, in_feat, in_channels, out_channels):
    if fuse_mode == 'SENet':
        return Fuse_SENet(in_feat, in_channels, out_channels)
    elif fuse_mode == 'Summation':
        return Fuse_Summation(in_feat, in_channels, out_channels)
    else:
        ValueError('Not Supported PLAN Fuse Mode')