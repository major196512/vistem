from yacs.config import CfgNode as CN

_PANET = CN()
_PANET.ENABLE = False
_PANET.IN_FEATURES = ['p3', 'p4', 'p5', 'p6', 'p7']
_PANET.OUT_CHANNELS = 256