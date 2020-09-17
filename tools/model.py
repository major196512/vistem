from vistem import dist
from vistem.config import get_cfg
from vistem.utils import setup_logger
from vistem.engine import launch, default_argument_parser
from vistem.modeling import build_model

logger = setup_logger(__name__)

def main(args):
    cfg = get_cfg(args.config_file)
    model = build_model(cfg)

    if dist.is_main_process():
        logger.info(f'Model Structure\n{model}')
        logger.info(f'Backbone Network\n{model.backbone}')
        logger.debug(f'Backbone Output Shape : {model.backbone.output_shape()}')
        logger.debug(f'Backbone Output Features : {model.backbone.out_features}')
        logger.debug(f'Backbone Stride : {model.backbone.out_feature_strides}')
        logger.debug(f'Backbone Output Channels : {model.backbone.out_feature_channels}')

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
