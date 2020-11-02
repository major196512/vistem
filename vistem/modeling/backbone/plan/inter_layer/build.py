import torch
import torch.nn as nn

from .bottom_up import PLANBottomUp
from .top_down import PLANTopDown
from .sharing import PLANSharing

__all__ = ['InterLayer']

class InterLayer(nn.Module):
    def __init__(self, mode, top_feat, bot_feat, in_channels, plan_cfg):
        super().__init__()
        self.top_feat = top_feat
        self.bot_feat = bot_feat
        self.in_channels = in_channels

        self.mode = mode
        if mode == 'DotProduct':
            pass
        elif mode == 'TopDownOnly':
            self.plan_bot = PLANTopDown(top_feat, bot_feat, in_channels, plan_cfg)
            self.top_out_channels = 0
            self.bot_out_channels = self.plan_bot.out_channels
        elif mode == 'BottomUpOnly':
            self.plan_top = PLANBottomUp(top_feat, bot_feat, in_channels, plan_cfg)
            self.top_out_channels = self.plan_top.out_channels
            self.bot_out_channels = 0
        elif mode == 'Sharing':
            self.plan = PLANSharing(top_feat, bot_feat, in_channels, plan_cfg)
            self.top_out_channels = self.plan.out_channels
            self.bot_out_channels = self.plan.out_channels
        elif mode == 'Default':
            self.plan_top = PLANBottomUp(top_feat, bot_feat, in_channels, plan_cfg)
            self.plan_bot = PLANTopDown(top_feat, bot_feat, in_channels, plan_cfg)
            self.top_out_channels = self.plan_top.out_channels
            self.bot_out_channels = self.plan_bot.out_channels
        else:
            raise ValueError('Not Supported InterLayer Mode')

    def forward(self, x):
        if self.mode == 'DotProduct':
            pass
        elif self.mode == 'TopDownOnly':
            plan_top = None
            plan_bot = self.plan_bot(x)
        elif self.mode == 'BottomUpOnly':
            plan_top = self.plan_top(x)
            plan_bot = None
        elif self.mode == 'Sharing':
            plan_top, plan_bot = self.plan(x)
        elif self.mode == 'Default':
            plan_top = self.plan_top(x)
            plan_bot = self.plan_bot(x)
        
        return plan_top, plan_bot