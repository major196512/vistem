from vistem.utils.registry import Registry
PROPOSAL_REGISTRY = Registry("PROPOSAL")

from .rpn import RPN

def build_proposal_generator(cfg, input_shape):
    model = cfg.MODEL.PROPOSAL_GENERATOR
    return PROPOSAL_REGISTRY.get(model)(cfg, input_shape)