import torch
import torch.nn as nn

from vistem.modeling.layers import Conv2d, get_norm, Swish, MemoryEfficientSwish

class FuseBlock(nn.Module):
    def __init__(
        self, 
        in_channel, 
        num_weights, 
        *, 
        norm='BN', 
        fuse_type='fast_norm', 
        memory_efficient=True
    ):
        super().__init__()

        assert fuse_type in {'unbound', 'softmax_norm', 'fast_norm'}

        if fuse_type == 'fast_norm':
            self.fuse = FastNormFuse(num_weights)

        self.swish = MemoryEfficientSwish() if memory_efficient else Swish()
        self.conv = Conv2d(
            in_channel, in_channel,
            kernel_size=3,
            padding=1,
            bias=True,
            norm=get_norm(norm, in_channel),
        )

    def forward(self, x):
        ret = self.fuse(x)
        ret = self.swish(ret)
        ret = self.conv(ret)

        return ret

class FastNormFuse(nn.Module):
    def __init__(self, num_weights, eps=1e-4):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_weights, ))
        self.eps = eps

    def forward(self, x):
        input_size = x.shape
        assert input_size[0] == self.weight.shape[0]

        x = self.weight[:, None] * x.view(input_size[0], -1)
        x = x.sum(dim=0) / (self.weight.sum() + self.eps)
        
        return x.view(input_size[1:])