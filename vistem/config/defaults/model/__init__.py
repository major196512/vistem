from yacs.config import CfgNode as CN

from .anchor import _ANCHOR_GENERATOR

from .fpn import _FPN
from .nas_fpn import _NAS_FPN
from .panet import _PANET

from .resnet import _RESNETS
from .retinanet import _RETINANET

from .rpn import _RPN
from .roi import _ROI
from .plan import _PLAN

_MODEL = CN()
_MODEL.META_ARCHITECTURE = 'RetinaNet'
_MODEL.BACKBONE = 'RetinaNetFPN'
_MODEL.PROPOSAL_GENERATOR = 'RPN'

_MODEL.WEIGHTS = ''
_MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675]
_MODEL.PIXEL_STD = [1.0, 1.0, 1.0]

_MODEL.ANCHOR_GENERATOR = _ANCHOR_GENERATOR

_MODEL.FPN = _FPN
_MODEL.NAS_FPN = _NAS_FPN
_MODEL.PANET = _PANET
_MODEL.RESNETS = _RESNETS
_MODEL.RETINANET = _RETINANET

_MODEL.ROI = _ROI
_MODEL.RPN = _RPN
_MODEL.PLAN = _PLAN
