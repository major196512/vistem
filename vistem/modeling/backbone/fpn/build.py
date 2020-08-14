from vistem.structures import ShapeSpec

from vistem.modeling.backbone import BACKBONE_REGISTRY
from vistem.modeling.backbone.resnet import ResNet
from .fpn import FPN
from .top_block import LastLevelMaxPool, LastLevelP6P7

__all__ = ['ResNetFPN', 'RetinaNetFPN']

@BACKBONE_REGISTRY.register()
def ResNetFPN(cfg, input_shape: ShapeSpec):
    bottom_up = ResNet(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS

    for feat in in_features:
        assert feat in bottom_up.out_features, f"'{feat}' is not in FPN bottom up({bottom_up.out_features})"

    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone


@BACKBONE_REGISTRY.register()
def RetinaNetFPN(cfg, input_shape: ShapeSpec):
    bottom_up = ResNet(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS

    for feat in in_features:
        assert feat in bottom_up.out_features, f"'{feat}' is not in FPN bottom up({bottom_up.out_features})"

    in_channels_p6p7 = bottom_up.out_feature_channels["res5"]
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelP6P7(in_channels_p6p7, out_channels),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone
    