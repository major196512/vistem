from vistem.structures import ShapeSpec

from vistem.modeling.backbone import BACKBONE_REGISTRY
from vistem.modeling.backbone.resnet import ResNet
from .fpn import FPNBase
from .top_block import LastLevelMaxPool, LastLevelP6P7

__all__ = ['FPN']

@BACKBONE_REGISTRY.register()
def FPN(cfg, input_shape: ShapeSpec):
    if cfg.BACKBONE.FPN.NAME == 'ResNetFPN' : return ResNetFPN(cfg, input_shape)
    elif cfg.BACKBONE.FPN.NAME == 'RetinaNetFPN' : return RetinaNetFPN(cfg, input_shape)
    else : 
        ValueError(f'Not Supported {cfg.BACKBONE.FPN.NAME}')

def ResNetFPN(cfg, input_shape: ShapeSpec):
    bottom_up = ResNet(cfg, input_shape)
    in_features = cfg.BACKBONE.FPN.IN_FEATURES
    out_channels = cfg.BACKBONE.FPN.OUT_CHANNELS

    for feat in in_features:
        assert feat in bottom_up.out_features, f"'{feat}' is not in FPN bottom up({bottom_up.out_features})"

    backbone = FPNBase(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.BACKBONE.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.BACKBONE.FPN.FUSE_TYPE,
    )
    return backbone

def RetinaNetFPN(cfg, input_shape: ShapeSpec):
    bottom_up = ResNet(cfg, input_shape)
    in_features = cfg.BACKBONE.FPN.IN_FEATURES
    out_channels = cfg.BACKBONE.FPN.OUT_CHANNELS

    for feat in in_features:
        assert feat in bottom_up.out_features, f"'{feat}' is not in FPN bottom up({bottom_up.out_features})"

    in_channels_p6p7 = bottom_up.out_feature_channels["res5"]
    backbone = FPNBase(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.BACKBONE.FPN.NORM,
        top_block=LastLevelP6P7(in_channels_p6p7, out_channels),
        fuse_type=cfg.BACKBONE.FPN.FUSE_TYPE,
    )
    return backbone
    