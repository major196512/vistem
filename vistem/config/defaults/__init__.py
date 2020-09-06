from yacs.config import CfgNode as CN

from .loader import _LOADER
from .input import _INPUT
from .solver import _SOLVER
from .model import _MODEL
from .test import _TEST

_C = CN()
_C.SEED = -1
_C.DEVICE = 'cuda'
_C.OUTPUT_DIR = './outputs'

_C.INPUT = _INPUT
_C.LOADER = _LOADER
_C.MODEL = _MODEL
_C.SOLVER = _SOLVER
_C.TEST = _TEST