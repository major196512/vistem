from yacs.config import CfgNode as CN

_TEST = CN()
_TEST.SCORE_THRESH = 0.05
_TEST.DETECTIONS_PER_IMAGE = 100
_TEST.EVAL_PERIOD = 5000
_TEST.WRITER_PERIOD = 20
_TEST.VIS_PERIOD = 20