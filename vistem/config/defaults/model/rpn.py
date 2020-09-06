from yacs.config import CfgNode as CN

_RPN = CN()
_RPN.HEAD_NAME = "StandardRPNHead"
_RPN.IN_FEATURES = ["res4"]

_RPN.MATCHER = CN()
_RPN.MATCHER.IOU_THRESHOLDS = [0.3, 0.7]
_RPN.MATCHER.IOU_LABELS = [0, -1, 1]
_RPN.MATCHER.LOW_QUALITY_MATCHES = True

_RPN.SAMPLING = CN()
_RPN.SAMPLING.BATCH_SIZE_PER_IMAGE = 256
_RPN.SAMPLING.POSITIVE_FRACTION = 0.5

_RPN.LOSS = CN()
_RPN.LOSS.LOC_TYPE = "smooth_l1"
_RPN.LOSS.SMOOTH_L1_BETA = 0.0
_RPN.LOSS.CLS_WEIGHT = 1.0
_RPN.LOSS.LOC_WEIGHT = 1.0

_RPN.TRAIN = CN()
_RPN.TRAIN.PRE_NMS_TOPK = 12000
_RPN.TRAIN.POST_NMS_TOPK = 2000

_RPN.TEST = CN()
_RPN.TEST.BBOX_REG_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
_RPN.TEST.NMS_THRESH = 0.7
_RPN.TEST.PRE_NMS_TOPK = 6000
_RPN.TEST.POST_NMS_TOPK = 1000
_RPN.TEST.MIN_SIZE = 0

# _RPN.BOUNDARY_THRESH = -1
