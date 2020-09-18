from vistem.utils.registry import Registry
ROI_REGISTRY = Registry("ROI")

from .head import StandardROIHeads

def build_roi_head(cfg, input_shape):
    model = cfg.META_ARCH.ROI.NAME
    return ROI_REGISTRY.get(model)(cfg, input_shape)