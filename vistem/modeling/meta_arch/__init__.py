from vistem.utils.registry import Registry
META_ARCH_REGISTRY = Registry("META_ARCH")

from .default import DefaultMetaArch
from .retinanet import RetinaNet
from .rpn import ProposalNetwork
from .faster_rcnn import FasterRCNN

def build_model(cfg):
    model = cfg.META_ARCH.NAME
    return META_ARCH_REGISTRY.get(model)(cfg).to(cfg.DEVICE)