from yacs.config import CfgNode as CN

_FPN = CN()
_FPN.IN_FEATURES = ['res3', 'res4', 'res5']
_FPN.OUT_CHANNELS = 256
_FPN.NORM = ''
_FPN.FUSE_TYPE = 'sum'