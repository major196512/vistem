import os
import random
from datetime import datetime

import torch
import numpy as np

from .logger import setup_logger

__all__ = ["seed_all_rng"]

def seed_all_rng(seed=None):
    if seed is None:
        seed = (
            os.getpid()
            + int(datetime.now().strftime("%S%f"))
            + int.from_bytes(os.urandom(2), "big")
        )
        logger = setup_logger()
        logger.info("Using a generated random seed {}".format(seed))
    np.random.seed(seed)
    torch.set_rng_state(torch.manual_seed(seed).get_state())
    random.seed(seed)
