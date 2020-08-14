import numpy as np
from PIL import Image

from vistem.loader.transforms import Transform

__all__ = ['ResizeTransform']

class ResizeTransform(Transform):
    def __init__(self, h, w, new_h, new_w, interp):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img, interp=None):
        assert img.shape[:2] == (self.h, self.w)
        pil_image = Image.fromarray(img)
        interp_method = interp if interp is not None else self.interp
        pil_image = pil_image.resize((self.new_w, self.new_h), interp_method)
        ret = np.asarray(pil_image)
        return ret

    def apply_coords(self, coords):
        coords[:, 0] = coords[:, 0] * (self.new_w / self.w)
        coords[:, 1] = coords[:, 1] * (self.new_h / self.h)
        return coords