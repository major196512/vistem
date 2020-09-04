from torch.utils.tensorboard import SummaryWriter

from .trainer import HookBase

class TensorboardXWriter(HookBase):
    def __init__(self, cfg):
        self._period = cfg.TEST.WRITER_PERIOD
        self._writer = SummaryWriter(cfg.OUTPUT_DIR)
        self._last_write = -1

    def after_step(self):
        next_iter = self.trainer.iter + 1
        if next_iter % self._period == 0:
            self.write()

    def write(self):
        storage = self.trainer.storage
        new_last_write = self._last_write
        for k, (v, iteration) in storage.latest_with_smoothing(self._period).items():
            if iteration > self._last_write:
                new_last_write = max(new_last_write, iteration)
                if 'loss' in k : k = f'Loss/{k}'
                elif ('time' in k) or ('second' in k) : k = f'Time/{k}'
                self._writer.add_scalar(k, v, iteration)
        self._last_write = new_last_write

        if len(storage._vis_data) >= 1:
            for img_name, img, step_num in storage._vis_data:
                self._writer.add_image(img_name, img, step_num)
            storage.clear_images()

        if len(storage._histograms) >= 1:
            for params in storage._histograms:
                self._writer.add_histogram_raw(**params)
            storage.clear_histograms()

    def after_train(self):
        if hasattr(self, "_writer"):  # doesn't exist when the code fails at import
            self._writer.close()
