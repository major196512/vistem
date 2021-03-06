import torch
import torch.nn as nn
import torch.nn.functional as F

from vistem.modeling.layers import Conv2d
from vistem.utils import weight_init

class Fuse_Summation(nn.Module):
    def __init__(self, feat, in_channels, out_channels):
        super().__init__()

        self.feat = feat
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        pyramid, inter_features = x

        ret = inter_features[0]
        for feat in inter_features[1:]:
            ret += feat
        ret += pyramid
        
        return ret