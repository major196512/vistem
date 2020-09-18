from yacs.config import CfgNode as CN

_ANCHOR_GENERATOR = CN()
_ANCHOR_GENERATOR.NAME = 'DefaultAnchorGenerator'
_ANCHOR_GENERATOR.SIZES = [[32, 64, 128, 256, 512]]
_ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]
