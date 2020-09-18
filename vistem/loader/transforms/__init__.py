from vistem.utils.registry import Registry
TRANSFORM_REGISTRY = Registry("TRANSFORM")

from .transform import Transform, TransformGen, NoOpTransform
from . import flip, resize, crop

from .build import build_transform_gen, apply_transform