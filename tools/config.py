import os

from vistem import dist
from vistem.config import get_cfg
from vistem.utils import setup_logger
from vistem.engine import default_argument_parser

def main(args):
    _logger = setup_logger(__name__)

    cfg = get_cfg(args.config_file)
    if cfg.SEED < 0 : cfg.SEED = dist.shared_random_seed()
    
    _logger.debug(f'Config File : \n{cfg}')
    # if cfg.OUTPUT_DIR and not os.path.isdir(cfg.OUTPUT_DIR) : os.makedirs(cfg.OUTPUT_DIR)
    # with open(os.path.join(cfg.OUTPUT_DIR, 'config'), 'w') as f:
    #     f.write(cfg.dump())

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    main(args)