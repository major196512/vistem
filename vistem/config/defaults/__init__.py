from yacs.config import CfgNode as CN

from .loader import _LOADER

_C = CN()
_C.SEED = -1
_C.DEVICE = 'cuda'
_C.OUTPUT_DIR = './outputs'

_C.LOADER = _LOADER