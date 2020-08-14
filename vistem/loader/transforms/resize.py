import sys
import numpy as np
from PIL import Image

from .transform import Transform, TransformGen, NoOpTransform

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


class Resize(TransformGen):
    def __init__(self, shape, interp=Image.BILINEAR):
        if isinstance(shape, int):
            shape = (shape, shape)
        shape = tuple(shape)
        self._init(locals())

    def get_transform(self, img):
        return ResizeTransform(
            img.shape[0], img.shape[1], self.shape[0], self.shape[1], self.interp
        )


class ResizeShortestEdge(TransformGen):
    def __init__(self, short_edge_length, max_size=sys.maxsize, sample_style="choice", interp=Image.BILINEAR):
        super().__init__()
        assert sample_style in ["range", "choice"], sample_style

        self.is_range = sample_style == "range"
        if isinstance(short_edge_length, int):
            short_edge_length = (short_edge_length, short_edge_length)
        self._init(locals())

    def get_transform(self, img):
        h, w = img.shape[:2]

        if self.is_range:
            size = np.random.randint(self.short_edge_length[0], self.short_edge_length[1] + 1)
        else:
            size = np.random.choice(self.short_edge_length)

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
