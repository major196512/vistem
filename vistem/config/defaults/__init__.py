from yacs.config import CfgNode as CN

from .anchor import _ANCHOR_GENERATOR
from .input import _INPUT
from .loader import _LOADER
from .solver import _SOLVER
from .test import _TEST

from .meta_arch import _META_ARCH
from .backbone import _BACKBONE

_C = CN()
_C.WEIGHTS = ''

_C.SEED = -1
_C.DEVICE = 'cuda'
_C.OUTPUT_DIR = './outputs'
_C.VISUALIZE_DIR = './visualize'
_C.PROJECT = 'VISTEM'

_C.BACKBONE = _BACKBONE
_C.META_ARCH = _META_ARCH

_C.ANCHOR_GENERATOR = _ANCHOR_GENERATOR
_C.INPUT = _INPUT
_C.LOADER = _LOADER
_C.SOLVER = _SOLVER
_C.TEST = _TEST