from vistem.utils.registry import Registry
ANCHOR_GENERATOR_REGISTRY = Registry("ANCHOR_GENERATOR")

from .base import BufferList, create_grid_offsets
from .default import DefaultAnchorGenerator

def build_anchor_generator(cfg, input_shape):
    anchor_generator = cfg.ANCHOR_GENERATOR.NAME
    return ANCHOR_GENERATOR_REGISTRY.get(anchor_generator)(cfg, input_shape)