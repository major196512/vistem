from vistem.utils.registry import Registry
BACKBONE_REGISTRY = Registry("BACKBONE")

from .backbone import Backbone
from .resnet import ResNet
from .fpn import FPN
# from .plan import PLAN

from vistem.structures import ShapeSpec

def build_backbone(cfg, input_shape=None):
    if input_shape is None:
        input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))

    backbone = cfg.MODEL.BACKBONE
    backbone = BACKBONE_REGISTRY.get(backbone)(cfg, input_shape)

    assert isinstance(backbone, Backbone)
    return backbone
