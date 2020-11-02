import torch
import torch.nn as nn

from vistem.modeling.layers import Conv2d, get_norm, Swish, MemoryEfficientSwish
from .fuse import FuseBlock

__all__ = ['TopDownBlock', 'BottomUpBlock']

class TopDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm='BN', fuse_type='fast_norm'):
        super().__init__()
        self.in_features = list(in_channels.keys())[::-1]

        td_channels = in_channels[self.in_features[0]]
        for in_feat in self.in_features[1:]:
            td_conv = Conv2d(
                td_channels,
                out_channels,
                kernel_size=1,
                padding=0,
                bias=True,
                norm=get_norm(norm, out_channels),
                activation=nn.Upsample(scale_factor=2)
            )

            in_conv = Conv2d(
                in_channels[in_feat],
                out_channels,
                kernel_size=1,
                padding=0,
                bias=True,
                norm=get_norm(norm, out_channels),
            )

            fuse = FuseBlock(out_channels, num_weights=2, norm=norm, fuse_type=fuse_type)

            self.add_module(f'{in_feat}_td', td_conv)
            self.add_module(f'{in_feat}_in', in_conv)
            self.add_module(f'{in_feat}_fuse', fuse)

            td_channels = out_channels

    def forward(self, x):
        ret = []
        ret.append(x[self.in_features[0]])

        for in_feat in self.in_features[1:]:
            in_node = getattr(self, f'{in_feat}_in')(x[in_feat])
            td_node = getattr(self, f'{in_feat}_td')(ret[-1])

            out_node = torch.stack([in_node, td_node], dim=0)
            out_node = getattr(self, f'{in_feat}_fuse')(out_node)

            ret.append(out_node)

        return dict(zip(self.in_features, ret))

class BottomUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm='BN', fuse_type='fast_norm'):
        super().__init__()
        self.in_features = list(in_channels.keys())

        for in_feat in self.in_features[1:]:
            bu_conv = Conv2d(
                out_channels,
                out_channels,
                kernel_size=1,
                padding=0,
                bias=True,
                norm=get_norm(norm, out_channels),
                activation=nn.MaxPool2d(kernel_size=2)
            )

            in_conv = Conv2d(
                in_channels[in_feat],
                out_channels,
                kernel_size=1,
                padding=0,
                bias=True,
                norm=get_norm(norm, out_channels),
            )

            self.add_module(f'{in_feat}_bu', bu_conv)
            self.add_module(f'{in_feat}_in', in_conv)

        for in_feat in self.in_features[1:-1]:
            td_conv = Conv2d(
                out_channels,
                out_channels,
                kernel_size=1,
                padding=0,
                bias=True,
                norm=get_norm(norm, out_channels),
            )
            fuse = FuseBlock(out_channels, num_weights=3, norm=norm, fuse_type=fuse_type)

            self.add_module(f'{in_feat}_td', td_conv)
            self.add_module(f'{in_feat}_fuse', fuse)

        fuse = FuseBlock(out_channels, num_weights=2, norm=norm, fuse_type=fuse_type)
        self.add_module(f'{self.in_features[-1]}_fuse', fuse)

    def forward(self, x):
        in_nodes, td_nodes = x
        ret = []
        ret.append(td_nodes[self.in_features[0]])

        for in_feat in self.in_features[1:-1]:
            in_node = getattr(self, f'{in_feat}_in')(in_nodes[in_feat])
            td_node = getattr(self, f'{in_feat}_td')(td_nodes[in_feat])
            bu_node = getattr(self, f'{in_feat}_bu')(ret[-1])

            out_node = torch.stack([in_node, td_node, bu_node], dim=0)
            out_node = getattr(self, f'{in_feat}_fuse')(out_node)

            ret.append(out_node)

        in_feat = self.in_features[-1]
        in_node = getattr(self, f'{in_feat}_in')(in_nodes[in_feat])
        bu_node = getattr(self, f'{in_feat}_bu')(ret[-1])

        out_node = torch.stack([in_node, bu_node], dim=0)
        out_node = getattr(self, f'{in_feat}_fuse')(out_node)

        ret.append(out_node)

        return dict(zip(self.in_features, ret))
