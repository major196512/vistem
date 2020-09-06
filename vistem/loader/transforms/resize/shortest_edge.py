import sys
import numpy as np
from PIL import Image

from vistem.loader.transforms import TRANSFORM_REGISTRY, TransformGen, NoOpTransform

from .transform import ResizeTransform

__all__ = ['ResizeShortestEdge']

@TRANSFORM_REGISTRY.register()
class ResizeShortestEdge(TransformGen):
    def __init__(self, cfg, is_train):
        super().__init__()

        if is_train:
            sample_style = cfg.INPUT.RESIZE.MIN_SIZE_TRAIN_SAMPLING
            min_size = cfg.INPUT.RESIZE.MIN_SIZE_TRAIN
            max_size = cfg.INPUT.RESIZE.MAX_SIZE_TRAIN
        else:
            sample_style = "choice"
            min_size = cfg.INPUT.RESIZE.MIN_SIZE_TEST
            max_size = cfg.INPUT.RESIZE.MAX_SIZE_TEST

        assert sample_style in ["range", "choice"], sample_style
        if isinstance(min_size, int):
            min_size = (min_size, min_size)

        self.init_local(min_size, max_size, sample_style)

    def init_local(self, min_size, max_size, sample_style = 'choice', interp = Image.BILINEAR):
        is_range = sample_style == "range"
        self._init(locals())

    def get_transform(self, img):
        h, w = img.shape[:2]

        if self.is_range:
            size = np.random.randint(self.min_size[0], self.min_size[1] + 1)
        else:
            size = np.random.choice(self.min_size)

        if size == 0 : return NoOpTransform()

        scale = size * 1.0 / min(h, w)
        if h < w : newh, neww = size, scale * w
        else : newh, neww = scale * h, size

        if max(newh, neww) > self.max_size:
            scale = self.max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale

        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return ResizeTransform(h, w, newh, neww, self.interp)
