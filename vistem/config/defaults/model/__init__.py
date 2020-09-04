from yacs.config import CfgNode as CN

from .anchor import _ANCHOR_GENERATOR
from .backbone import _BACKBONE
from .fpn import _FPN
from .resnet import _RESNETS
from .retinanet import _RETINANET
from .rpn import _RPN
from .roi_head import _ROI_HEAD

_MODEL = CN()
_MODEL.META_ARCHITECTURE = 'RetinaNet'
_MODEL.PROPOSAL_GENERATOR = 'RPN'
_MODEL.ROI_HEAD_MODULE = 'StandardROIHeads'

_MODEL.WEIGHTS = ''
_MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675]
_MODEL.PIXEL_STD = [1.0, 1.0, 1.0]

_MODEL.RESNETS = _RESNETS
_MODEL.FPN = _FPN
_MODEL.RETINANET = _RETINANET
_MODEL.ANCHOR_GENERATOR = _ANCHOR_GENERATOR
_MODEL.BACKBONE = _BACKBONE
_MODEL.RPN = _RPN
_MODEL.ROI_HEAD = _ROI_HEAD
