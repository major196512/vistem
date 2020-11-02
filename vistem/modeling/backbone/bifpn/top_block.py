import torch
import torch.nn as nn
import torch.nn.functional as F

from vistem.modeling.layers import Conv2d, get_norm
from vistem.utils import weight_init

__all__ = ['TopBlock']

class TopBlock(nn.Module):
    def __init__(self, in_features, out_features, in_channels, out_channels, norm='BN'):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        assert len(in_features) == len(in_channels)
        in_channel = in_channels[-1]
        self.out_channels = in_channels

        self.num_block = 0
        for out_feat in out_features[len(in_features):]:
            conv1x1 = Conv2d(
                in_channel, out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
                norm=get_norm(norm, out_channels),
                activation=nn.MaxPool2d(kernel_size=2),
            )
            self.add_module(out_feat, conv1x1)
            self.num_block += 1
            in_channel = out_channels
            self.out_channels.append(out_channels)

        self.out_channels = dict(zip(self.out_features, self.out_channels))

        # for module in [self.p6, self.p7]:
        #     weight_init.c2_xavier_fill(module)

    def forward(self, x):
        ret = list()
        for in_feat in self.in_features:
            ret.append(x[in_feat])

        for out_feat in self.out_features[-self.num_block:]:
            top_node = getattr(self, out_feat)(ret[-1])
            ret.append(top_node)

        return dict(zip(self.out_features, ret))
