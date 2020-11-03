from vistem.dist import get_rank
from vistem.config import get_cfg
from vistem.utils import setup_logger
from vistem.engine import launch, Trainer, default_argument_parser

def main(args):
    cfg = get_cfg(args.config_file)
    if args.mini_sgd > 1:
        cfg.SOLVER.IMG_PER_BATCH = int(cfg.SOLVER.IMG_PER_BATCH / args.mini_sgd)
        cfg.SOLVER.BASE_LR /= args.mini_sgd
        cfg.SOLVER.MAX_ITER = int(cfg.SOLVER.MAX_ITER * args.mini_sgd)
        cfg.SOLVER.WARMUP.ITERS = int(cfg.SOLVER.WARMUP.ITERS * args.mini_sgd)
        cfg.SOLVER.CHECKPOINT_PERIOD = int(cfg.SOLVER.CHECKPOINT_PERIOD * args.mini_sgd)
        cfg.TEST.EVAL_PERIOD = int(cfg.TEST.EVAL_PERIOD * args.mini_sgd)
        cfg.SOLVER.SCHEDULER.STEPS = tuple([k * args.mini_sgd for k in cfg.SOLVER.SCHEDULER.STEPS])

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if args.eval_only : trainer.test()
    else : trainer.train()

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
