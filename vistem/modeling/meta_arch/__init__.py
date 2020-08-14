from vistem.utils.registry import Registry
META_ARCH_REGISTRY = Registry("META_ARCH")

from .retinanet import RetinaNet

def build_model(cfg):
    model = cfg.MODEL.META_ARCHITECTURE
    return META_ARCH_REGISTRY.get(model)(cfg)