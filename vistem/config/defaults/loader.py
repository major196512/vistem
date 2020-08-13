from yacs.config import CfgNode as CN

_LOADER = CN()

_LOADER.TRAIN_DATASET = 'coco_2017_train'
_LOADER.TEST_DATASET = 'coco_2017_val'
_LOADER.NUM_WORKERS = 4
_LOADER.TRAIN_SHUFFLE = True
