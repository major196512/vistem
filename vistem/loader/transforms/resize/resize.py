import sys
import numpy as np
from PIL import Image

from vistem.loader.transforms import TRANSFORM_REGISTRY, TransformGen, NoOpTransform

from .transform import ResizeTransform

__all__ = ['Resize']

@TRANSFORM_REGISTRY.register()
class Resize(TransformGen):
    def __init__(self, cfg, is_train):
        super().__init__()

        if is_train : shape = cfg.INPUT.MAX_SIZE_TRAIN
        else : shape = cfg.INPUT.MAX_SIZE_TEST

        self.init_local(shape)

    def init_local(self, shape, interp=Image.BILINEAR):
        if isinstance(shape, int) : shape = (shape, shape)
        shape = tuple(shape)
        self._init(locals())

    def get_transform(self, img):
        return ResizeTransform(
            img.shape[0], img.shape[1], self.shape[0], self.shape[1], self.interp
        )
