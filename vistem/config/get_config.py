import yaml
from yacs.config import CfgNode as CN

__all__ = ['get_cfg']

def get_cfg(cfg_file):
    from .defaults import _C
    cfg = _C.clone()

    merge_cfg(cfg, cfg_file)
    enable_cfg(cfg)

    return cfg

def enable_cfg(cfg):
    keys = list(cfg.keys())
    for key in keys:
        if not isinstance(cfg[key], CN) : continue
        enable = cfg[key].pop('ENABLE', True)
        if enable : enable_cfg(cfg[key])
        else : cfg.pop(key)

def merge_cfg(cfg, file):
    cfg_file = yaml.load(open(file), Loader=yaml.Loader)
    base_file = cfg_file.pop('BASE_CFG', None)
    data_file = cfg_file.pop('DATA_CFG', None)
    
    if base_file is not None : merge_cfg(cfg, base_file)
    if data_file is not None : merge_cfg(cfg, data_file)

    cfg_file = CN(cfg_file)
    cfg.merge_from_other_cfg(cfg_file)
    