import torch
import torch.nn as nn
import math

from vistem.modeling.layers import Conv2d

class AttnTopDown(nn.Module):
    def __init__(self, in_features, in_channels, out_channels, num_heads, num_convs, erf):
        
        super().__init__()

        self.in_features = in_features
        self.out_channels = out_channels
        self.num_layers = len(in_features)
        self.num_heads = num_heads
        self.erf = erf

        conv_keys = []
        conv_queries = []
        conv_values = []
        for top_idx, bot_idx in zip(in_features[1:], in_features[:-1]):
            conv_keys.append(Conv2d(in_channels[top_idx], out_channels, 1, stride=1, bias=False))
            conv_queries.append(Conv2d(in_channels[bot_idx], out_channels, 1, stride=1, bias=False))
            conv_values.append(Conv2d(in_channels[bot_idx], out_channels, 1, stride=1, bias=False))

        self.conv_keys = nn.ModuleList(conv_keys)
        self.conv_queries = nn.ModuleList(conv_queries)
        self.conv_values = nn.ModuleList(conv_values)

        conv_weights = []
        for bot_idx in in_features[:-1]:
            in_channel = in_channels[bot_idx]
            concat = []
            concat.append(Conv2d(in_channel, in_channel, 1, stride=1, bias=False))
            for _ in range(num_convs-1):
                concat.append(nn.ReLU())
                concat.append(Conv2d(in_channel, in_channel, 1, stride=1, bias=False))
            conv_weights.append(nn.Sequential(*concat))

        self.conv_weights = nn.ModuleList(conv_weights)

        for layer in self.modules():
            if isinstance(layer, Conv2d):
                torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                if layer.bias is not None : torch.nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        top_features = [x[k] for k in self.in_features[1:]]
        bot_features = [x[k] for k in self.in_features[:-1]]
        results = []

        for idx in range(self.num_layers-1):
            top_feat = top_features[idx]
            bot_feat = bot_features[idx]
            B, _, H, W = top_feat.shape
            weighted_features = []

            queries = self.conv_queries[idx](bot_feat).permute(1, 0, 2, 3).split(self.out_channels // self.num_heads)
            values = self.conv_values[idx](bot_feat).permute(1, 0, 2, 3).split(self.out_channels // self.num_heads)
            keys = self.conv_keys[idx](top_feat).permute(1, 0, 2, 3).split(self.out_channels // self.num_heads)

            for head_idx in range(self.num_heads):
                query = queries[head_idx].permute(1, 0, 2, 3)
                value = values[head_idx].permute(1, 0, 2, 3)
                key = keys[head_idx].permute(1, 0, 2, 3)
                
                query = nn.functional.unfold(query, kernel_size=self.erf, padding=(self.erf-1)//2, stride=2)
                query = query.reshape(B, -1, self.erf * self.erf, H, W)
                value = nn.functional.unfold(value, kernel_size=self.erf, padding=(self.erf-1)//2, stride=2)
                value = value.reshape(B, -1, self.erf * self.erf, H, W)

                weight = (query * key.unsqueeze(2)).sum(dim=1) / math.sqrt(self.out_channels // self.num_heads)
                weight = weight.softmax(dim=1)

                weighted_feature = (value * weight.unsqueeze(1))
                weighted_feature = weighted_feature.reshape(B, -1, H*W)
                weighted_feature = nn.functional.fold(weighted_feature, bot_feat.shape[-2:], kernel_size=self.erf, padding=(self.erf-1)//2, stride=2)
                weighted_features.append(weighted_feature)

            weighted_features = torch.cat(weighted_features, dim=1)
            weighted_features = self.conv_weights[idx](weighted_features)

            results.append(weighted_features)

        top_layer_idx = self.in_features[-1]
        results.append(x[top_layer_idx])
        return dict(zip(self.in_features, results))
