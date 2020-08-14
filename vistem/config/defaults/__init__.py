from yacs.config import CfgNode as CN

from .loader import _LOADER
from .input import _INPUT
from .solver import _SOLVER
from .model import _MODEL

_C = CN()
_C.SEED = -1
_C.DEVICE = 'cuda'
_C.OUTPUT_DIR = './outputs'

_C.LOADER = _LOADER
_C.INPUT = _INPUT
_C.SOLVER = _SOLVER
_C.MODEL = _MODEL