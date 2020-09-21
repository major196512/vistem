import math

from vistem.modeling.backbone import BACKBONE_REGISTRY
from vistem.modeling.layers import Conv2d, get_norm, Swish, MemoryEfficientSwish

from .stem import BasicStem
from .block import MBConvBlock, HeadBlock
from .model import EfficientNetBase

__all__ = ['EfficientNet']

def make_stage(block_class, num_blocks, first_stride, **kwargs):
    blocks = []
    for i in range(num_blocks):
        blocks.append(block_class(stride=first_stride if i == 0 else 1, **kwargs))
        kwargs["in_channels"] = kwargs["out_channels"]
    return blocks
    
@BACKBONE_REGISTRY.register()
def EfficientNet(cfg, input_shape, memory_efficient=True):
    # norm = cfg.MODEL.RESNETS.NORM
    norm = 'BN'
    pi = 1
    se_ratio = 0.25
    drop_connect_prob = 0.2

    freeze_at           = 0
    out_features        = ['stage3', 'stage4', 'stage5', 'stage7', 'stage9']
    num_classes         = cfg.MODEL.RESNETS.NUM_CLASSES
    
    depth_factor = [1.0, 1.1, 1.2, 1.4, 1.8, 2.2, 2.6, 3.1]
    width_factor = [1.0, 1.0, 1.1, 1.2, 1.4, 1.6, 1.8, 2.0]

    B0_layers_per_stage = [1, 1, 2, 2, 3, 3, 4, 1, 1] # B0
    num_layers_per_stage = [int(math.ceil(depth_factor[pi] * n)) for n in B0_layers_per_stage]
    num_layers_per_stage[0] = 1
    num_layers_per_stage[-1] = 1

    B0_channels_per_stage = [32, 16, 24, 40, 80, 112, 192, 320, 1280] # B0
    num_channels_per_stage = [int(8 * round(width_factor[pi] * n / 8)) for n in B0_channels_per_stage]

    kernel_sizes = [3, 3, 3, 5, 3, 5, 5, 3, 1]
    expand_dims = [0, 1, 6, 6, 6, 6, 6, 6, 0]
    first_strides = [2, 1, 2, 2, 2, 1, 2, 1, 1]

    stem = BasicStem(
        in_channels=input_shape.channels,
        out_channels=num_channels_per_stage[0],
        norm=norm,
    )

    out_stage_idx = {
        "stage1": 1, "stage2": 2, "stage3": 3, "stage4": 4, "stage5": 5, 
        "stage6": 6, "stage7": 7, "stage8": 8, "stage9": 9, "linear": 9, 
    }
    out_stage_idx = [out_stage_idx[f] for f in out_features]
    max_stage_idx = max(out_stage_idx)

    stages = []
    for idx in range(1, max_stage_idx):
        num_blocks = num_layers_per_stage[idx]
        in_channels = num_channels_per_stage[idx-1]
        out_channels = num_channels_per_stage[idx]

        stage_kargs = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "num_blocks": num_blocks,
            "kernel_size" : kernel_sizes[idx],
            "expand_dim" : expand_dims[idx],
            "first_stride": first_strides[idx],
            'se_ratio' : se_ratio,
            'drop_connect_prob' : drop_connect_prob,
            "norm": norm,
        }

        if idx < 8:
            stage_kargs["block_class"] = MBConvBlock

        else:
            stage_kargs["block_class"] = HeadBlock
            stage_kargs.pop('expand_dim')
            stage_kargs.pop('se_ratio')
            stage_kargs.pop('drop_connect_prob')
            
        blocks = make_stage(**stage_kargs)

        stages.append(blocks)

    dropout_probs = [0.2, 0.2, 0.3, 0.3, 0.4, 0.4, 0.5, 0.5]
    return EfficientNetBase(stem, stages, out_features=out_features, num_classes=num_classes, dropout_prob=dropout_probs[pi]).freeze(freeze_at)
