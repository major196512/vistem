from vistem.structures import ShapeSpec

from vistem.modeling.backbone import BACKBONE_REGISTRY
from vistem.modeling.backbone.fpn import FPN
from .plan import PLANBase

__all__ = ['PLAN']

@BACKBONE_REGISTRY.register()
def PLAN(cfg, input_shape: ShapeSpec):
    fpn = FPN(cfg, input_shape)

    in_channels         = cfg.BACKBONE.FPN.OUT_CHANNELS
    out_channels        = cfg.BACKBONE.PLAN.OUT_CHANNELS

    num_heads           = cfg.BACKBONE.PLAN.NUM_HEADS
    num_convs           = cfg.BACKBONE.PLAN.NUM_CONVS
    num_weights         = cfg.BACKBONE.PLAN.NUM_WEIGHTS
    erf                 = cfg.BACKBONE.PLAN.ERF

    assert in_channels % num_heads == 0

    backbone = PLANBase(
        fpn=fpn,
        out_channels=out_channels,
        num_heads=num_heads,
        num_convs=num_convs,
        num_weights=num_weights,
        erf=erf,
    )
    return backbone