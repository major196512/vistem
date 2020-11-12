import torch
from torch.nn.parallel import DistributedDataParallel

import os
import datetime
from contextlib import contextmanager

from vistem.utils import setup_logger, Timer
from vistem import dist

from vistem.loader import build_test_loader
from vistem.modeling import build_model
from vistem.checkpointer import Checkpointer

class Visualizer:
    def __init__(self, cfg):
        self._logger = setup_logger(__name__, all_rank=True)
        
        if dist.is_main_process():
            self._logger.debug(f'Config File : \n{cfg}')
            if cfg.VISUALIZE_DIR and not os.path.isdir(cfg.VISUALIZE_DIR) : os.makedirs(cfg.VISUALIZE_DIR)
        dist.synchronize()
        
        self.test_loader = build_test_loader(cfg)

        self.model = build_model(cfg)
        self.model.eval()
        if dist.is_main_process():
            self._logger.debug(f"Model Structure\n{self.model}")
                
        if dist.get_world_size() > 1:
            self.model = DistributedDataParallel(self.model, device_ids=[dist.get_local_rank()], broadcast_buffers=False)

        self.checkpointer = Checkpointer(
            self.model,
            cfg.OUTPUT_DIR,
        )
        self.checkpointer.load(cfg.WEIGHTS)

    def __call__(self):
        num_devices = dist.get_world_size()

        total = len(self.test_loader)  # inference data loader must have a fixed length
        self._logger.info(f"Start visualize on {total} images")

        timer = Timer(warmup = 5, pause=True)
        total_compute_time = 0
        total_time = 0

        with inference_context(self.model), torch.no_grad():
            for idx, inputs in enumerate(self.test_loader):
                timer.resume()
                outputs = self.model(inputs)
                if torch.cuda.is_available() : torch.cuda.synchronize()
                timer.pause()

                if timer.total_seconds() > 10:
                    total_compute_time += timer.seconds()
                    total_time += timer.total_seconds()
                    timer.reset(pause=True)

                    total_seconds_per_img = total_time / (idx + 1)
                    seconds_per_img = total_compute_time / (idx + 1)
                    eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                    self._logger.info(f"Visualize done {idx + 1}/{total}. {seconds_per_img:.4f} s / img. ETA={eta}")

        total_compute_time += timer.seconds()
        total_time += timer.total_seconds()

        total_time_str = str(datetime.timedelta(seconds=total_time))
        self._logger.info(
            f"Total Visualize time: {total_time_str} ({total_time / total:.6f} s / img per device, on {num_devices} devices)"
        )

        total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
        self._logger.info(
            f"Total visualize pure compute time: {total_compute_time_str} ({total_compute_time / total:.6f} s / img per device, on {num_devices} devices)"
        )

@contextmanager
def inference_context(model):
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)