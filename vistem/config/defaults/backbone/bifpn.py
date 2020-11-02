from yacs.config import CfgNode as CN

_BIFPN = CN()
_BIFPN.ENABLE = False

_BIFPN.BOTTOM_UP = 'EfficientNet'
_BIFPN.IN_FEATURES = ['stage5', 'stage7', 'stage9']
_BIFPN.OUT_FEATURES = ['p3', 'p4', 'p5', 'p6', 'p7']
_BIFPN.FUSE_TYPE = 'fast_norm'

_BIFPN.NUM_LAYERS = 3
_BIFPN.OUT_CHANNELS = 64
