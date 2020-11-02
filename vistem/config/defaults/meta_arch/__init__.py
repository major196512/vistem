from yacs.config import CfgNode as CN

from .resnet import _RESNET
from .efficientnet import _EFFICIENTNET

from .retinanet import _RETINANET

from .rpn import _RPN
from .roi import _ROI

from .efficientdet import _EFFICIENTDET

_META_ARCH = CN()
_META_ARCH.NAME = 'RetinaNet'
_META_ARCH.PROPOSAL_GENERATOR = 'RPN'
_META_ARCH.NUM_CLASSES = 80

_META_ARCH.RESNET = _RESNET
_META_ARCH.EFFICIENTNET = _EFFICIENTNET

_META_ARCH.RETINANET = _RETINANET

_META_ARCH.ROI = _ROI
_META_ARCH.RPN = _RPN

_META_ARCH.EFFICIENTDET = _EFFICIENTDET