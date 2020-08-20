from collections import Counter
from .trainer import HookBase

class LRScheduler(HookBase):
    def __init__(self, cfg, optimizer, scheduler):
        self._optimizer = optimizer
        self._scheduler = scheduler
        self.accumulate = cfg.SOLVER.ACCUMULATE

        # NOTE: some heuristics on what LR to summarize
        # summarize the param group with most parameters
        largest_group = max(len(g["params"]) for g in optimizer.param_groups)

        if largest_group == 1:
            # If all groups have one parameter,
            # then find the most common initial LR, and use it for summary
            lr_count = Counter([g["lr"] for g in optimizer.param_groups])
            lr = lr_count.most_common()[0][0]
            for i, g in enumerate(optimizer.param_groups):
                if g["lr"] == lr:
                    self._best_param_group_id = i
                    break
        else:
            for i, g in enumerate(optimizer.param_groups):
                if len(g["params"]) == largest_group:
                    self._best_param_group_id = i
                    break

    def before_train(self):
        self._optimizer.zero_grad()

    def after_step(self):
        lr = self._optimizer.param_groups[self._best_param_group_id]["lr"]
        self.trainer.storage.put_scalar("lr", lr, smoothing_hint=False)
        if (self.trainer.iter + 1) % self.accumulate == 0:
            self._optimizer.step()
            for _ in range(self.accumulate): 
                self._scheduler.step()
            self._optimizer.zero_grad()