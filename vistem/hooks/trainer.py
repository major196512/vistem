import weakref
from vistem.utils import setup_logger, EventStorage

__all__ = ['HookBase', 'HookTrainer']

class HookBase:
    def before_train(self):
        pass

    def after_train(self):
        pass

    def before_step(self):
        pass

    def after_step(self):
        pass

class HookTrainer:
    def __init__(self, cfg):
        self._logger = setup_logger(__name__)
        self._hooks = []
        
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER

    def register_hooks(self, hooks):
        hooks = [h for h in hooks if h is not None]
        for h in hooks:
            assert isinstance(h, HookBase)
            h.trainer = weakref.proxy(self)
        self._hooks.extend(hooks)

    

    def train(self):
        self._logger.info(f"Starting training from iteration {self.start_iter}")

        with EventStorage(self.start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in range(self.start_iter, self.max_iter):
                    self.before_step()
                    self.run_step()
                    self.after_step()
            except Exception:
                self._logger.critical("Exception during training:")
                raise
            finally:
                self.after_train()

    def before_train(self):
        for h in self._hooks:
            h.before_train()

    def after_train(self):
        for h in self._hooks:
            h.after_train()

    def before_step(self):
        for h in self._hooks:
            h.before_step()

    def after_step(self):
        for h in self._hooks:
            h.after_step()
        # this guarantees, that in each hook's after_step, storage.iter == trainer.iter
        self.storage.step()

    def run_step(self):
        raise NotImplementedError