from yacs.config import CfgNode as CN

_ROI_HEAD = CN()

_ROI_HEAD.IN_FEATURES = ["p2", "p3", "p4", "p5", "p6"]
_ROI_HEAD.NUM_CLASSES = 80

# Matcher
_ROI_HEAD.IOU_THRESHOLDS = [0.5]
_ROI_HEAD.IOU_LABELS = [0, 1]

# Proposal gt
_ROI_HEAD.PROPOSAL_APPEND_GT = True
_ROI_HEAD.BATCH_SIZE_PER_IMAGE = 512
_ROI_HEAD.POSITIVE_FRACTION = 0.5

# ROI Pooling
_ROI_HEAD.BOX_POOLER_RESOLUTION = 14
_ROI_HEAD.BOX_POOLER_SAMPLING_RATIO = 0
_ROI_HEAD.BOX_POOLER_TYPE = "ROIAlignV2"

# ROI Box Head
_ROI_HEAD.BOX_HEAD_NUM_CONV = 0
_ROI_HEAD.BOX_HEAD_CONV_DIM = 256
_ROI_HEAD.BOX_HEAD_NUM_FC = 0
_ROI_HEAD.BOX_HEAD_FC_DIM = 1024
_ROI_HEAD.BOX_HEAD_NORM = ''

_ROI_HEAD.TRAIN_ON_PRED_BOXES = False