from . import HookBase

from vistem import dist

class EvalHook(HookBase):
    def __init__(self, cfg):
        self._period = cfg.TEST.EVAL_PERIOD

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_eval()

    def _do_eval(self):
        results = self.trainer.test()

        if results:
            assert isinstance(results, dict), f"Eval function must return a dict. Got {results} instead."

            flattened_results = flatten_results_dict(results)
            for k, v in flattened_results.items():
                try:
                    v = float(v)
                except Exception:
                    raise ValueError(
                        "[EvalHook] eval_function should return a nested dict of float. "
                        f"Got '{k}: {v}' instead."
                    )
            self.trainer.storage.put_scalars(**flattened_results, smoothing_hint=False)

        dist.synchronize()
