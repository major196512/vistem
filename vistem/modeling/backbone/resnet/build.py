from vistem.modeling.backbone import BACKBONE_REGISTRY

from .stem import *
from .block import *
from .model import ResNetBase

def make_stage(block_class, num_blocks, first_stride, **kwargs):
    blocks = []
    for i in range(num_blocks):
        blocks.append(block_class(stride=first_stride if i == 0 else 1, **kwargs))
        kwargs["in_channels"] = kwargs["out_channels"]
    return blocks
    
@BACKBONE_REGISTRY.register()
def ResNet(cfg, input_shape):
    norm = cfg.BACKBONE.RESNETS.NORM
    stem = BasicStem(
        in_channels=input_shape.channels,
        out_channels=cfg.BACKBONE.RESNETS.STEM_OUT_CHANNELS,
        norm=norm,
        stem_bias=cfg.BACKBONE.RESNETS.STEM_BIAS
    )
    
    freeze_at           = cfg.BACKBONE.RESNETS.FREEZE_AT
    out_features        = cfg.BACKBONE.RESNETS.OUT_FEATURES
    depth               = cfg.BACKBONE.RESNETS.DEPTH
    res5_dilation       = cfg.BACKBONE.RESNETS.RES5_DILATION
    in_channels         = cfg.BACKBONE.RESNETS.STEM_OUT_CHANNELS
    out_channels        = cfg.BACKBONE.RESNETS.RES2_OUT_CHANNELS
    num_groups          = cfg.BACKBONE.RESNETS.NUM_GROUPS
    width_per_group     = cfg.BACKBONE.RESNETS.WIDTH_PER_GROUP
    bottleneck_channels = num_groups * width_per_group
    stride_in_1x1       = cfg.BACKBONE.RESNETS.STRIDE_IN_1X1
    
    assert res5_dilation in {1, 2}, f"res5_dilation cannot be {res5_dilation}."

    num_blocks_per_stage = {50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3]}[depth]

    # Avoid creating variables without gradients
    # It consumes extra memory and may cause allreduce to fail
    out_stage_idx = [{"res2": 2, "res3": 3, "res4": 4, "res5": 5, "linear" : 5}[f] for f in out_features]
    max_stage_idx = max(out_stage_idx)

    stages = []
    for idx, stage_idx in enumerate(range(2, max_stage_idx + 1)):
        dilation = res5_dilation if stage_idx == 5 else 1
        first_stride = 1 if idx == 0 or (stage_idx == 5 and dilation == 2) else 2

        stage_kargs = {
            "num_blocks": num_blocks_per_stage[idx],
            "first_stride": first_stride,
            "in_channels": in_channels,
            "bottleneck_channels": bottleneck_channels,
            "out_channels": out_channels,
            "num_groups": num_groups,
            "norm": norm,
            "stride_in_1x1": stride_in_1x1,
            "dilation": dilation,
        }
        stage_kargs["block_class"] = BottleneckBlock
        blocks = make_stage(**stage_kargs)

        in_channels = out_channels
        out_channels *= 2
        bottleneck_channels *= 2

        # if freeze_at >= stage_idx:
        #     for block in blocks:
        #         block.freeze()

        stages.append(blocks)

    return ResNetBase(stem, stages, out_features=out_features).freeze(freeze_at)
