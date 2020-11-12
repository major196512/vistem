from collections import defaultdict
import os
import wandb

from .trainer import HookBase

class WandbWriter(HookBase):
    def __init__(self, cfg, model):
        self._period = cfg.TEST.WRITER_PERIOD
        self._save_period = cfg.SOLVER.CHECKPOINT_PERIOD
        self._output_dir = cfg.OUTPUT_DIR
        self._last_write = -1

        self.wandb = wandb.init(project=cfg.PROJECT)
        # wandb.run.name = cfg.OUTPUT_DIR
        wandb.config.update(cfg)
        # wandb.watch(model)

    def after_step(self):
        next_iter = self.trainer.iter + 1
        if next_iter % self._period == 0:
            self.write()

        if self._save_period > 0 and next_iter % self._save_period == 0:
            wandb.save(os.path.join(self._output_dir, f'model_{int(self.trainer.iter):07d}.pth'))

    def after_train(self):
        self.wandb.finish()
        
    def write(self):
        storage = self.trainer.storage
        new_last_write = self._last_write
        for k, (v, iteration) in storage.latest_with_smoothing(self._period).items():
            if iteration > self._last_write:
                new_last_write = max(new_last_write, iteration)
                if 'loss' in k : k = f'Loss/{k}'
                elif ('time' in k) or ('second' in k) : k = f'Time/{k}'
                wandb.log({k : v}, step=iteration)

        self._last_write = new_last_write

        if len(storage._vis_data) >= 1:
            for img_name, img, step_num in storage._vis_data:
                wandb.log({img_name : wandb.Image(img.transpose((1, 2, 0)), caption=f'step : {step_num}')}, step=self._last_write)
            storage.clear_images()

        # if len(storage._histograms) >= 1:
        #     for params in storage._histograms:
        #         self._writer.add_histogram_raw(**params)
        #     storage.clear_histograms()

        