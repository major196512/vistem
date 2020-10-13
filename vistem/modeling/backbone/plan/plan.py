import torch
import torch.nn as nn
from typing import List
import math

from vistem.modeling.backbone import Backbone
from vistem.structures import ShapeSpec
from vistem.modeling.layers.SEblock import SqueezeExcitation2d
from vistem.modeling.layers import Conv2d

from .mode import plan_mode

__all__ = ['PLANBase']

class PLANBase(Backbone):
    def __init__(self, pyramid, out_channels, mode, plan_cfg):
        super().__init__()

        in_features                 = pyramid.out_features
        pyramid_shape               = pyramid.output_shape()

        self._out_features          = in_features
        self._out_feature_strides   = {name : s.stride for name, s in pyramid_shape.items()}
        self._out_feature_channels  = {name : s.channels for name, s in pyramid_shape.items()}
        self._size_divisibility     = max([p.stride for p in pyramid_shape.values()])

        in_channels = self._out_feature_channels
        assert (len(in_channels) == len(in_features)) and (all(k in in_channels for k in in_features))

        self.pyramid = pyramid
        self.init_fuse3(in_features, mode, plan_cfg)

    def init_fuse1(self, in_features, mode, plan_cfg):
        self.plan_top_down = []
        for top_feat, bot_feat in zip(in_features[1:], in_features[:-1]):
            assert self._out_feature_channels[top_feat] == self._out_feature_channels[bot_feat]
            out_channels = self._out_feature_channels[bot_feat]
            
            attn = plan_mode(mode, top_feat, bot_feat, out_channels, plan_cfg)
            self.plan_top_down.append(attn)
            self.add_module(f'TopDown_{top_feat}_{bot_feat}', attn)

        self.plan_top_down = self.plan_top_down[::-1]

    def init_fuse2(self, in_features, mode, plan_cfg):
        self.plan_bottom_up = []
        for top_feat, bot_feat in zip(in_features[1:], in_features[:-1]):
            assert self._out_feature_channels[top_feat] == self._out_feature_channels[bot_feat]
            out_channels = self._out_feature_channels[bot_feat]
            
            attn = plan_mode(mode, top_feat, bot_feat, out_channels, plan_cfg)
            self.plan_bottom_up.append(attn)
            self.add_module(f'BottomUp_{top_feat}_{bot_feat}', attn)

    def init_fuse3(self, in_features, mode, plan_cfg):
        self.plan_bottom_up = []
        for top_feat, bot_feat in zip(in_features[1:], in_features[:-1]):
            assert self._out_feature_channels[top_feat] == self._out_feature_channels[bot_feat]
            out_channels = self._out_feature_channels[bot_feat]
            
            attn = plan_mode(mode, top_feat, bot_feat, out_channels, plan_cfg)
            self.plan_bottom_up.append(attn)
            self.add_module(f'BottomUp_{top_feat}_{bot_feat}', attn)

        self.se = []
        self.conv = []
        for feat in self._out_features:
            out_channels = self._out_feature_channels[feat]
            in_channels = self.plan_bottom_up[0].num_heads * self.plan_bottom_up[0].value_dim
            if (feat != self._out_features[0]) and (feat != self._out_features[-1]) : in_channels = 2 * out_channels

            se_block = SqueezeExcitation2d(in_channels, 32, in_channels)
            conv_block = Conv2d(in_channels, out_channels, stride=1, kernel_size=1, padding=0, bias=False)

            self.se.append(se_block)
            self.conv.append(conv_block)
            self.add_module(f'SE_Block_{feat}', se_block)
            self.add_module(f'Conv_Block_{feat}', conv_block)

    def forward(self, x):
        pyramid_features = self.pyramid(x)
        return self.forward_fuse4(pyramid_features)

    def forward_fuse1(self, pyramid_features):
        results = []

        results.append(pyramid_features[self.plan_top_down[0].top_feat])
        for plan_top_down in self.plan_top_down:
            top_feat = plan_top_down.top_feat
            bot_feat = plan_top_down.bot_feat
            plan_top, plan_bot = plan_top_down([pyramid_features[top_feat], pyramid_features[bot_feat]])

            # results.append(plan_top)
            results.append(plan_bot)

        # results.append(prev_top)

        return dict(zip(self._out_features, results[::-1]))

    def forward_fuse2(self, pyramid_features):
        results = []
        # for out_feat in self._out_features:
        #     results[out_feat] = pyramid_features[out_feat]

        for plan_bottom_up in self.plan_bottom_up:
            top_feat = plan_bottom_up.top_feat
            bot_feat = plan_bottom_up.bot_feat
            plan_top, plan_bot = plan_bottom_up([pyramid_features[top_feat], pyramid_features[bot_feat]])

            if len(results) : results[-1] += plan_bot
            else : results.append(plan_bot + pyramid_features[bot_feat])
            results.append(plan_top + pyramid_features[top_feat])

        return dict(zip(self._out_features, results))

    def forward_fuse3(self, pyramid_features):
        from collections import defaultdict
        results = defaultdict(list)

        for plan_bottom_up in self.plan_bottom_up:
            top_feat = plan_bottom_up.top_feat
            bot_feat = plan_bottom_up.bot_feat
            plan_top, plan_bot = plan_bottom_up([pyramid_features[top_feat], pyramid_features[bot_feat]])

            results[bot_feat].append(plan_bot)
            results[top_feat].append(plan_top)

        for idx, feat in enumerate(self._out_features):
            feat_list = torch.cat(results[feat], dim=1)
            feat_list = self.se[idx](feat_list)
            feat_list = self.conv[idx](feat_list)
            results[feat] = feat_list + pyramid_features[feat]

        return results

    def forward_fuse4(self, pyramid_features):
        from collections import defaultdict
        results = defaultdict(list)

        for plan_bottom_up in self.plan_bottom_up:
            top_feat = plan_bottom_up.top_feat
            bot_feat = plan_bottom_up.bot_feat
            plan_top, plan_bot = plan_bottom_up([pyramid_features[top_feat], pyramid_features[bot_feat]])

            results[bot_feat].extend(plan_bot)
            results[top_feat].extend(plan_top)

        for idx, feat in enumerate(self._out_features):
            feat_list = torch.cat(results[feat], dim=1)
            feat_list = self.se[idx](feat_list)
            feat_list = self.conv[idx](feat_list)
            results[feat] = feat_list + pyramid_features[feat]

        return results

    @property
    def size_divisibility(self):
        return self._size_divisibility

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

