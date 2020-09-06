from yacs.config import CfgNode as CN

_SOLVER = CN()

_SOLVER.OPTIMIZER = 'SGD'
_SOLVER.MAX_ITER = 90000
_SOLVER.IMG_PER_BATCH = 16
_SOLVER.BASE_LR = 0.01

_SOLVER.SCHEDULER = CN()
_SOLVER.SCHEDULER.NAME = 'WarmupMultiStepLR'
_SOLVER.SCHEDULER.STEPS = (60000, 80000)
_SOLVER.SCHEDULER.GAMMA = 0.1

_SOLVER.WARMUP = CN()
_SOLVER.WARMUP.METHOD = 'linear'
_SOLVER.WARMUP.FACTOR = 0.001
_SOLVER.WARMUP.ITERS = 1000

_SOLVER.WEIGHT_DECAY = CN()
_SOLVER.WEIGHT_DECAY.BASE = 0.0001
_SOLVER.WEIGHT_DECAY.NORM = 0.0
_SOLVER.WEIGHT_DECAY.BIAS = 0.0001

_SOLVER.BIAS_LR_FACTOR = 1.0
_SOLVER.MOMENTUM = 0.9
_SOLVER.ACCUMULATE = 1

_SOLVER.CHECKPOINT_PERIOD = 1000
_SOLVER.CHECKPOINT_KEEP = 0