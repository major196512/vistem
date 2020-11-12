from vistem import dist
from vistem.config import get_cfg
from vistem.utils import setup_logger
from vistem.engine import launch, Visualizer, default_argument_parser

def main(args):
    cfg = get_cfg(args.config_file)
    if args.mini_sgd > 1:
        cfg.SOLVER.IMG_PER_BATCH = int(cfg.SOLVER.IMG_PER_BATCH / args.mini_sgd)

    visualizer = Visualizer(cfg)
    visualizer()

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    logger = setup_logger(__name__)
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
