from vistem.structures import ShapeSpec

from vistem.modeling.backbone import BACKBONE_REGISTRY
from vistem.modeling.backbone.fpn import FPN
from .plan import PLANBase

__all__ = ['PLAN']

@BACKBONE_REGISTRY.register()
def PLAN(cfg, input_shape: ShapeSpec):
    pyramid = cfg.BACKBONE.PLAN.PYRAMID_NAME
    pyramid = BACKBONE_REGISTRY.get(pyramid)(cfg, input_shape)

    plan_cfg            = cfg.BACKBONE.PLAN.PLAN_CFG
    interlayer_mode     = cfg.BACKBONE.PLAN.INTERLAYER_MODE
    fuse_mode           = cfg.BACKBONE.PLAN.FUSE_MODE
    repeat              = cfg.BACKBONE.PLAN.REPEAT

    backbone = PLANBase(
        pyramid=pyramid,
        interlayer_mode=interlayer_mode,
        fuse_mode=fuse_mode,
        repeat=repeat,
        plan_cfg=plan_cfg,
    )
    return backbone