from yacs.config import CfgNode as CN

_RESNETS = CN()
_RESNETS.ENABLE = False

_RESNETS.FREEZE_AT = 2
_RESNETS.DEPTH = 50
_RESNETS.NUM_CLASSES = 0
_RESNETS.OUT_FEATURES = ['res3', 'res4', 'res5']
_RESNETS.NORM = 'BN'
_RESNETS.STEM_OUT_CHANNELS = 64
_RESNETS.RES2_OUT_CHANNELS = 256
_RESNETS.RES5_DILATION = 1
_RESNETS.NUM_GROUPS = 1
_RESNETS.WIDTH_PER_GROUP = 64
_RESNETS.STRIDE_IN_1X1 = True