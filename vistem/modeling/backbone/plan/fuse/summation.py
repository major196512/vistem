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
        self.conv_block = Conv2d(out_channels, out_channels, stride=1, kernel_size=3, padding=1, bias=True)
        weight_init.c2_msra_fill(self.conv_block)

    def forward(self, x):
        pyramid, inter_features = x

        ret = inter_features[0]
        for feat in inter_features[1:]:
            ret += feat
        
        # return F.relu(self.conv_block(pyramid + ret))
        return self.conv_block(pyramid + ret)