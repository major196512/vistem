from yacs.config import CfgNode as CN

from .anchor import _ANCHOR_GENERATOR
from .backbone import _BACKBONE
from .fpn import _FPN
from .resnet import _RESNET
from .retinanet import _RETINANET

_MODEL = CN()
_MODEL.META_ARCHITECTURE = 'RetinaNet'
_MODEL.WEIGHTS = ''
_MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675]
_MODEL.PIXEL_STD = [1.0, 1.0, 1.0]

_MODEL.RESNET = _RESNET
_MODEL.FPN = _FPN
_MODEL.RETINANET = _RETINANET
_MODEL.ANCHOR_GENERATOR = _ANCHOR_GENERATOR
_MODEL.BACKBONE = _BACKBONE





