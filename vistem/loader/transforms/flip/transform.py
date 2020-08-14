import numpy as np
from vistem.loader.transforms import Transform

__all__ = ['HFlipTransform']

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