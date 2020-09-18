from yacs.config import CfgNode as CN

from .retinanet import _RETINANET

from .rpn import _RPN
from .roi import _ROI

_META_ARCH = CN()
_META_ARCH.NAME = 'RetinaNet'
_META_ARCH.PROPOSAL_GENERATOR = 'RPN'

_META_ARCH.RETINANET = _RETINANET

_META_ARCH.ROI = _ROI
_META_ARCH.RPN = _RPN
