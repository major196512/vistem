from vistem.utils.registry import Registry
META_ARCH_REGISTRY = Registry("META_ARCH")

from .default import DefaultMetaArch
from .retinanet import RetinaNet
from .rpn import ProposalNetwork

def build_model(cfg):
    model = cfg.MODEL.META_ARCHITECTURE
    return META_ARCH_REGISTRY.get(model)(cfg).to(cfg.DEVICE)