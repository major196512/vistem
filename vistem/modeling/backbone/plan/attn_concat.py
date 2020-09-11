import torch
import torch.nn as nn

from vistem.modeling.layers import Conv2d

class AttnFuse(nn.Module):
    def init_concat(self, in_features, out_channels, num_weights):
        self.in_features = in_features

        conv = []
        for _ in range(len(self.in_feaures)):
            fuse_conv = []
            fuse_conv.append(Conv2d(out_channels, out_channels, 1, stride=1, bias=True))
            for _ in range(num_weights-1):
                fuse_conv.append(nn.ReLU())
                fuse_conv.append(Conv2d(out_channels, out_channels, 1, stride=1, bias=True))
            conv.append(nn.Sequential(*fuse_conv))

        self.conv = nn.ModuleList(conv)

    def forward(self, fpn_feat, top_down, bot_up):
        results = []
        for idx, in_feat in self.in_features.items():
            feat = fpn_feat[in_feat] + top_down[in_feat] + bot_up[in_feat]
            results.append(self.conv[idx](feat))

        return dict(zip(self.in_features, results))