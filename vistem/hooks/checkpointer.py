import os
from typing import Optional

from .trainer import HookBase
from vistem.utils import setup_logger

class PeriodicCheckpointer(HookBase):
    def __init__(self, cfg, checkpointer):
        self.logger = setup_logger(__name__)
        self.checkpointer = checkpointer
        self.period = int(cfg.SOLVER.CHECKPOINT_PERIOD)
        self.max_to_keep = cfg.SOLVER.CHECKPOINT_KEEP
        self.recent_checkpoints = []
        
    def before_train(self):
        self.max_iter = self.trainer.max_iter

    def after_step(self):
        iteration = int(self.trainer.iter)
        if (iteration + 1) % self.period == 0 :
            self.save_checkpoint(iteration, save_name=f"model_{iteration:07d}")

    def after_train(self):
        self.max_to_keep = 0
        iteration = int(self.trainer.iter)
        self.save_checkpoint(iteration, save_name=f"model_final")

    def save_checkpoint(self, iteration, save_name):
        additional_state = {"iteration": iteration}
        self.checkpointer.save(save_name, **additional_state)

        if self.max_to_keep > 0:
            self.recent_checkpoints.append(self.checkpointer.get_checkpoint_file())
            if len(self.recent_checkpoints) > self.max_to_keep:
                file_to_delete = self.recent_checkpoints.pop(0)
                try:
                    if not file_to_delete.endswith("model_final.pth"):
                        os.remove(file_to_delete)
                except:
                    self.logger.warning(f'Checkpoint File not Exists : {file_to_delete}')