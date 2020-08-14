import numpy as np

from vistem.loader.transforms import TRANSFORM_REGISTRY
from .transform import Transform, TransformGen, TransformList

__all__ = ['build_transform_gen', 'apply_transform']

def build_transform_gen(cfg, is_train):
    tfm_gens = []
    for tfm_gen in cfg.INPUT.TRANSFORM:
        tfm_gens.append(TRANSFORM_REGISTRY.get(tfm_gen)(cfg, is_train))

    return tfm_gens

def apply_transform(transform_gens, img):
    for g in transform_gens:
        assert isinstance(g, TransformGen), g

    assert isinstance(img, np.ndarray), f"[TransformGen] Needs an numpy array, but got a {type(img)}!"
    assert not isinstance(img.dtype, np.integer) or (img.dtype == np.uint8), (
        f"[TransformGen] Got image of type {img.dtype}, use uint8 or floating points instead!")
    assert img.ndim in [2, 3], img.ndim

    tfms = []
    for g in transform_gens:
        tfm = g.get_transform(img)
        assert isinstance(tfm, Transform), f"TransformGen {g} must return an instance of Transform! Got {tfm} instead"
        img = tfm.apply_image(img)
        tfms.append(tfm)
    return img, TransformList(tfms)
