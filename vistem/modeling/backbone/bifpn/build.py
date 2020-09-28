from vistem.structures import ShapeSpec

from vistem.modeling.backbone import BACKBONE_REGISTRY
from vistem.modeling.backbone.efficientnet import EfficientNet

from .model import BiFPNBase
from .top_block import TopBlock

__all__ = ['BiFPN']

@BACKBONE_REGISTRY.register()
def BiFPN(cfg, input_shape: ShapeSpec):
    bottom_up = cfg.BACKBONE.BIFPN.BOTTOM_UP
    if bottom_up == 'EfficientNet':
        bottom_up = EfficientNet(cfg, input_shape)
    else:
        ValueError(f'{bottom_up} : Unsupported Bottom-up Network in BiFPN')

    in_features         = cfg.BACKBONE.BIFPN.IN_FEATURES
    out_features        = cfg.BACKBONE.BIFPN.OUT_FEATURES
    
    num_layers          = cfg.BACKBONE.BIFPN.NUM_LAYERS
    out_channels        = cfg.BACKBONE.BIFPN.OUT_CHANNELS

    for feat in in_features:
        assert feat in bottom_up.out_features, f"'{feat}' is not in BiFPN bottom up({bottom_up.out_features})"

    in_channels = [bottom_up.out_feature_channels[f] for f in in_features]
    top_block = TopBlock(
        in_features=in_features,
        out_features=out_features,
        in_channels=in_channels,
        out_channels=out_channels,
    )

    return BiFPNBase(
        bottom_up=bottom_up,
        top_block=top_block,
        out_channels=out_channels,
        num_layers=num_layers,
        fuse_type=cfg.BACKBONE.BIFPN.FUSE_TYPE
    )