import torch
import torch.nn as nn

from vistem.modeling.layers.SEblock import SqueezeExcitation2d
from vistem.modeling.layers import Conv2d

class Fuse_SENet(nn.Module):
    def __init__(self, feat, in_channels, out_channels):
        super().__init__()

        self.feat = feat
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.SE_block = SqueezeExcitation2d(in_channels, 32, in_channels)
        self.conv_block = conv_block = Conv2d(in_channels, out_channels, stride=1, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        pyramid, inter_features = x
        
        feat_list = torch.cat(inter_features, dim=1)
        feat_list = self.SE_block(feat_list)
        feat_list = self.conv_block(feat_list)
        
        return feat_list + pyramid