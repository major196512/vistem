import numpy as np

from vistem.loader.transforms import TRANSFORM_REGISTRY, TransformGen, NoOpTransform
from .transform import HFlipTransform

__all__ = ['RandomFlip']

@TRANSFORM_REGISTRY.register()
class RandomFlip(TransformGen):
    # def __init__(self, prob=0.5):
    def __init__(self, cfg, is_train):
        super().__init__()
        self.init_local(prob=cfg.INPUT.FLIP_PROB)

    def init_local(self, prob=0.5):
        horiz, vert = True, False
        self._init(locals())

    def get_transform(self, img):
        h, w = img.shape[:2]
        do = self._rand_range() < self.prob
        if do:
            return HFlipTransform(w)
        else:
            return NoOpTransform()
