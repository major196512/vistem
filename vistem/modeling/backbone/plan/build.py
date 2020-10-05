from vistem.structures import ShapeSpec

from vistem.modeling.backbone import BACKBONE_REGISTRY
from vistem.modeling.backbone.fpn import FPN
from .plan import PLANBase

__all__ = ['PLAN']

@BACKBONE_REGISTRY.register()
def PLAN(cfg, input_shape: ShapeSpec):
    pyramid = cfg.BACKBONE.PLAN.PYRAMID_NAME
    pyramid = BACKBONE_REGISTRY.get(pyramid)(cfg, input_shape)

    out_channels        = cfg.BACKBONE.PLAN.OUT_CHANNELS
    plan_cfg            = cfg.BACKBONE.PLAN.PLAN_CFG

    backbone = PLANBase(
        pyramid=pyramid,
        out_channels=out_channels,
        plan_cfg=plan_cfg,
    )
    return backbone