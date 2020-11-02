import math

from vistem.modeling.backbone import BACKBONE_REGISTRY
from vistem.modeling.layers import Conv2d, get_norm, Swish, MemoryEfficientSwish

from .stem import BasicStem
from .head import HeadBlock
from .block import MBConvBlock

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
    freeze_at               = cfg.BACKBONE.EFFICIENTNET.FREEZE_AT
    out_features            = cfg.BACKBONE.EFFICIENTNET.OUT_FEATURES
    num_classes             = cfg.META_ARCH.NUM_CLASSES
    norm                    = cfg.BACKBONE.EFFICIENTNET.NORM
    
    depth_factor            = cfg.BACKBONE.EFFICIENTNET.DEPTH.DEPTH_FACTOR
    width_factor            = cfg.BACKBONE.EFFICIENTNET.DEPTH.WIDTH_FACTOR
    se_ratio                = cfg.BACKBONE.EFFICIENTNET.DEPTH.SE_RATIO
    drop_connect_prob       = cfg.BACKBONE.EFFICIENTNET.DEPTH.DROP_CONNECT_PROB

    B0_layers_per_stage     = cfg.BACKBONE.EFFICIENTNET.STAGE_PARAM.NUM_LAYERS_PER_STAGE
    B0_channels_per_stage   = cfg.BACKBONE.EFFICIENTNET.STAGE_PARAM.OUT_CHANNELS_PER_STAGE

    kernel_sizes            = cfg.BACKBONE.EFFICIENTNET.STAGE_PARAM.KERNEL_SIZE_PER_STAGE
    expand_dims             = cfg.BACKBONE.EFFICIENTNET.STAGE_PARAM.EXPAND_DIMS_PER_STAGE
    first_strides           = cfg.BACKBONE.EFFICIENTNET.STAGE_PARAM.STRIDES_PER_STAGE

    stem = BasicStem(
        in_channels=input_shape.channels,
        out_channels=int(8 * round(width_factor * 4)),
        norm=norm,
    )

    num_layers_per_stage = [int(math.ceil(depth_factor * n)) for n in B0_layers_per_stage]
    num_channels_per_stage = [int(8 * round(width_factor * n / 8)) for n in B0_channels_per_stage]

    out_stage_idx = {
        "stage1": 0, "stage2": 1, "stage3": 2, "stage4": 3, "stage5": 4, 
        "stage6": 5, "stage7": 6, "stage8": 7, "stage9": 7, 
    }
    out_stage_idx = [out_stage_idx[f] for f in out_features]
    max_stage_idx = max(out_stage_idx)

    stages = []
    in_channels = stem.out_channels
    for idx in range(max_stage_idx):
        num_blocks = num_layers_per_stage[idx]
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

        stage_kargs["block_class"] = MBConvBlock
        blocks = make_stage(**stage_kargs)
        stages.append(blocks)

        in_channels = out_channels

    if 'stage9' in out_features:
        stage_kargs = {
            "in_channels": in_channels,
            "out_channels": int(8 * round(width_factor * 160)),
            "norm": norm,
        }
        head = HeadBlock(**stage_kargs)

    return EfficientNetBase(stem, stages, head, out_features=out_features).freeze(freeze_at)
