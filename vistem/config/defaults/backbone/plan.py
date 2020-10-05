from yacs.config import CfgNode as CN

_PLAN = CN()
_PLAN.ENABLE = False
_PLAN.PYRAMID_NAME = 'FPN'

_PLAN.PLAN_MODE = 'PLAN_mode1'
_PLAN.PLAN_CFG = 'h8k64v64e3'
_PLAN.OUT_CHANNELS = 256