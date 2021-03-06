from vistem.utils.registry import Registry
BACKBONE_REGISTRY = Registry("BACKBONE")

from .backbone import Backbone

from .resnet import ResNet
from .efficientnet import EfficientNet

from .fpn import FPN
from .nas_fpn import NAS_FPN
from .panet import PANet
from .bifpn import BiFPN

from .plan import PLAN

from vistem.structures import ShapeSpec

def build_backbone(cfg, input_shape=None):
    if input_shape is None:
        input_shape = ShapeSpec(channels=len(cfg.INPUT.PIXEL_MEAN))

    if 'PLAN' in cfg.BACKBONE:
        pyramidal = cfg.BACKBONE.NAME
        pyramidal = BACKBONE_REGISTRY.get(pyramidal)(cfg, input_shape)
        backbone = BACKBONE_REGISTRY.get('PLAN')(cfg, input_shape, pyramidal)

    else:
        backbone = cfg.BACKBONE.NAME
        backbone = BACKBONE_REGISTRY.get(backbone)(cfg, input_shape)

    assert isinstance(backbone, Backbone)
    return backbone
