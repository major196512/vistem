from vistem.utils.registry import Registry
BACKBONE_REGISTRY = Registry("BACKBONE")

from .backbone import Backbone

from .resnet import ResNet
from .efficientnet import EfficientNet

from .fpn import FPN
from .nas_fpn import NAS_FPN
from .panet import PANet
# from .plan import PLAN

from vistem.structures import ShapeSpec

def build_backbone(cfg, input_shape=None):
    if input_shape is None:
        input_shape = ShapeSpec(channels=len(cfg.INPUT.PIXEL_MEAN))

    backbone = cfg.BACKBONE.NAME
    backbone = BACKBONE_REGISTRY.get(backbone)(cfg, input_shape)

    assert isinstance(backbone, Backbone)
    return backbone
