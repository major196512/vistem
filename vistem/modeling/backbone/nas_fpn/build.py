from vistem.structures import ShapeSpec

from vistem.modeling.backbone import BACKBONE_REGISTRY
from vistem.modeling.backbone.fpn import FPN

from .nas_fpn import NASFPNBase

__all__ = ['NAS_FPN']

@BACKBONE_REGISTRY.register()
def NAS_FPN(cfg, input_shape: ShapeSpec):
    fpn = FPN(cfg, input_shape)

    cell_inputs = cfg.MODEL.NAS_FPN.CELL_INPUTS
    for feats in cell_inputs:
        for feat in feats:
            assert feat.startswith('cell') or feat.startswith('rcb') or (feat in fpn.out_features), \
                f"CELL INPUTS '{feat}' is not in NAS-FPN cell nodes or FPN({fpn.out_features})"
            
    cell_outputs = cfg.MODEL.NAS_FPN.CELL_OUTPUTS
    for feat in cell_outputs:
        assert feat in fpn.out_features, f"CELL OUTPUTS '{feat}' is not in FPN({fpn.out_features})"
        
    cell_ops = cfg.MODEL.NAS_FPN.CELL_OPS

    assert len(cell_inputs) == len(cell_outputs)
    assert len(cell_inputs) == len(cell_ops)

    backbone = NASFPNBase(
        fpn=fpn,
        cell_inputs=cell_inputs,
        cell_outputs=cell_outputs,
        cell_ops=cell_ops,
        nas_outputs=cfg.MODEL.NAS_FPN.NAS_OUTPUTS,
    )
    return backbone