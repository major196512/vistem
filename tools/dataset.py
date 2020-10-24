from vistem import dist
from vistem.config import get_cfg
from vistem.utils import setup_logger
from vistem.engine import launch, default_argument_parser

import torch
from vistem.loader import build_train_loader, build_test_loader
from vistem.structures import Instances, PolygonMasks

logger = setup_logger(__name__)

def main(args):
    cfg = get_cfg(args.config_file)

    train_loader = build_train_loader(cfg)
    test_loader = build_test_loader(cfg)

    train_iter = iter(train_loader)
    test_iter = iter(test_loader)

    train_data = next(train_iter)
    test_data = next(test_iter)

    if dist.is_main_process():
        logger.debug(f'The Number of Input Data : {len(train_data)}')
        logger.debug(f'Input Data Structure')
        for k, v in train_data[0].items():
            v_type = type(v)
            if v_type == torch.Tensor :
                logger.debug(f'{k} : {v.shape}')
            elif v_type == Instances:
                v_fields = v.get_fields()
                logger.debug(f'{k} : {len(v)}')
                for kk, vv in v_fields.items():
                    if type(vv) == PolygonMasks:
                        logger.debug(f'{k}_{kk} : {vv.polygons}')
                    else:
                        logger.debug(f'{k}_{kk} : {vv}')
            else:
                logger.debug(f'{k} : {v}')


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    logger.info(f"Command Line Args:{args}")
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_ip=args.dist_ip,
        dist_port=args.dist_port,
        args=(args,),
    )
