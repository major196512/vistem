from vistem.dist import get_rank
from vistem.config import get_cfg
from vistem.utils import setup_logger, find_caller
from vistem.engine import launch, Trainer, default_argument_parser

def main(args):
    cfg = get_cfg(args.config_file)

    trainer = Trainer(cfg)
    # trainer.resume_or_load()
    # trainer.train()

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    logger = setup_logger()
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
