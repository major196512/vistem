from vistem.structures import ShapeSpec

from vistem.modeling.backbone import BACKBONE_REGISTRY
from vistem.modeling.backbone.fpn import FPN

from .panet import PANetBase

__all__ = ['PANet']

@BACKBONE_REGISTRY.register()
def PANet(cfg, input_shape: ShapeSpec):
    fpn = FPN(cfg, input_shape)

    in_features = cfg.BACKBONE.PANET.IN_FEATURES
    for in_feat in in_features:
        assert in_feat in fpn.out_features, f"'{in_feat}' is not in FPN({fpn.out_features})"

    backbone = PANetBase(
        fpn=fpn,
        in_features=in_features,
        out_channels=cfg.BACKBONE.PANET.OUT_CHANNELS,

    )

    return backbone