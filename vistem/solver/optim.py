import torch
from typing import List, Dict, Any

def build_optimizer(cfg, model: torch.nn.Module) -> torch.optim.Optimizer:
    params: List[Dict[str, Any]] = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue

        lr = cfg.SOLVER.BASE_LR / cfg.SOLVER.ACCUMULATE
        weight_decay = cfg.SOLVER.WEIGHT_DECAY

        if key.endswith("norm.weight") or key.endswith("norm.bias"):
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_NORM
        elif key.endswith(".bias"):
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
            
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if cfg.SOLVER.OPTIMIZER == 'SGD':
        optimizer = torch.optim.SGD(params, lr, momentum=cfg.SOLVER.MOMENTUM)
    else:
        raise ValueError(f"Unknown optimizer: {cfg.SOLVER.OPTIMIZER}")

    return optimizer