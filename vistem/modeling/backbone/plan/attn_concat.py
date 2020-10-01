import torch
import torch.nn as nn

from vistem.modeling.layers import Conv2d

class AttnFuse(nn.Module):
    def __init__(self, in_features, out_channels, num_weights):
        super().__init__()

        self.in_features = in_features

        conv = []
        for _ in range(len(self.in_features)):
            fuse_conv = []
            fuse_conv.append(Conv2d(out_channels, out_channels, 1, stride=1, bias=True))
            for _ in range(num_weights-1):
                fuse_conv.append(nn.ReLU())
                fuse_conv.append(Conv2d(out_channels, out_channels, 1, stride=1, bias=True))
            conv.append(nn.Sequential(*fuse_conv))

        self.conv = nn.ModuleList(conv)

    def forward(self, fpn_feat, top_down, bot_up):
        results = []
        for idx, in_feat in enumerate(self.in_features):
            feat = fpn_feat[in_feat] + top_down[in_feat] + bot_up[in_feat]
            results.append(self.conv[idx](feat))

        return dict(zip(self.in_features, results))