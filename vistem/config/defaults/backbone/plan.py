from yacs.config import CfgNode as CN

_PLAN = CN()
_PLAN.ENABLE = False

_PLAN.PYRAMID_NAME = 'FPN'
_PLAN.INTERLAYER_MODE = 'Default'
_PLAN.FUSE_MODE = 'SENet'
_PLAN.REPEAT = 1
_PLAN.PLAN_CFG = 'h8k64v64e3'