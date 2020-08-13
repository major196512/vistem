import numpy as np

from .base import Transform, TransformGen, NoOpTransform

class HFlipTransform(Transform):
    def __init__(self, width: int):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        # NOTE: opencv would be faster:
        # https://github.com/pytorch/pytorch/issues/16424#issuecomment-580695672
        if img.ndim <= 3:  # HxW, HxWxC
            return np.flip(img, axis=1)
        else:
            return np.flip(img, axis=-2)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        coords[:, 0] = self.width - coords[:, 0]
        return coords

class RandomFlip(TransformGen):
    def __init__(self, prob=0.5):
        horiz, vert = True, False
        super().__init__()
        self._init(locals())

    def get_transform(self, img):
        h, w = img.shape[:2]
        do = self._rand_range() < self.prob
        if do:
            return HFlipTransform(w)
        else:
            return NoOpTransform()
