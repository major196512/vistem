from yacs.config import CfgNode as CN

from .resnet import _RESNETS
from .efficientnet import _EFFICIENTNET

from .fpn import _FPN
from .nas_fpn import _NAS_FPN
from .panet import _PANET
from .bifpn import _BIFPN

_BACKBONE = CN()
_BACKBONE.NAME = 'RetinaNetFPN'

_BACKBONE.RESNETS = _RESNETS
_BACKBONE.EFFICIENTNET = _EFFICIENTNET

_BACKBONE.FPN = _FPN
_BACKBONE.NAS_FPN = _NAS_FPN
_BACKBONE.PANET = _PANET
_BACKBONE.BIFPN = _BIFPN