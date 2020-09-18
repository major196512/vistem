import sys
import numpy as np
from PIL import Image

from vistem.loader.transforms import TRANSFORM_REGISTRY, TransformGen, NoOpTransform

from .transform import CropTransform

__all__ = ['RandomCrop']

@TRANSFORM_REGISTRY.register()
class RandomCrop(TransformGen):
    def __init__(self, cfg, is_train):
        super().__init__()

        crop_type = cfg.INPUT.CROP.TYPE
        crop_size = cfg.INPUT.CROP.SIZE

        assert crop_type in ["relative_range", "relative", "absolute", "absolute_range"]

        self._init(locals())

    def get_transform(self, img):
        h, w = img.shape[:2]
        croph, cropw = self.get_crop_size(h, w)
        assert h >= croph and w >= cropw, f"Shape computation in {self} has bugs."

        h0 = np.random.randint(h - croph + 1)
        w0 = np.random.randint(w - cropw + 1)
        return CropTransform(w0, h0, cropw, croph)

    def get_crop_size(self, h, w):
        if self.crop_type == "relative":
            ch, cw = self.crop_size
            return int(h * ch + 0.5), int(w * cw + 0.5)

        elif self.crop_type == "relative_range":
            crop_size = np.asarray(self.crop_size, dtype=np.float32)
            ch, cw = crop_size + np.random.rand(2) * (1 - crop_size)
            return int(h * ch + 0.5), int(w * cw + 0.5)

        elif self.crop_type == "absolute":
            return (min(self.crop_size[0], h), min(self.crop_size[1], w))

        elif self.crop_type == "absolute_range":
            assert self.crop_size[0] <= self.crop_size[1]
            ch = np.random.randint(min(h, self.crop_size[0]), min(h, self.crop_size[1]) + 1)
            cw = np.random.randint(min(w, self.crop_size[0]), min(w, self.crop_size[1]) + 1)
            return ch, cw

        else:
            NotImplementedError(f"Unknown crop type {self.crop_type}")