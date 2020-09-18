import numpy as np

from vistem.loader.transforms import Transform

__all__ = ['CropTransform']

class CropTransform(Transform):
    def __init__(self, x0: int, y0: int, w: int, h: int):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        if len(img.shape) <= 3:
            return img[self.y0 : self.y0 + self.h, self.x0 : self.x0 + self.w]
        else:
            return img[..., self.y0 : self.y0 + self.h, self.x0 : self.x0 + self.w, :]

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        coords[:, 0] -= self.x0
        coords[:, 1] -= self.y0
        return coords