from yacs.config import CfgNode as CN

from .resnet import _RESNETS

from .fpn import _FPN
from .nas_fpn import _NAS_FPN
from .panet import _PANET

_BACKBONE = CN()
_BACKBONE.NAME = 'RetinaNetFPN'

_BACKBONE.RESNETS = _RESNETS

_BACKBONE.FPN = _FPN
_BACKBONE.NAS_FPN = _NAS_FPN
_BACKBONE.PANET = _PANET