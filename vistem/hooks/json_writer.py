import os
import json
from collections import defaultdict

from .trainer import HookBase

class JSONWriter(HookBase):
    """
    Examples parsing such a json file:
    ::
        $ cat metrics.json | jq -s '.[0:2]'
        [
          {
            "data_time": 0.008433341979980469,
            "iteration": 20,
            "loss": 1.9228371381759644,
            "loss_box_reg": 0.050025828182697296,
            "loss_classifier": 0.5316952466964722,
            "loss_mask": 0.7236229181289673,
            "loss_rpn_box": 0.0856662318110466,
            "loss_rpn_cls": 0.48198649287223816,
            "lr": 0.007173333333333333,
            "time": 0.25401854515075684
          },
          {
            "data_time": 0.007216215133666992,
            "iteration": 40,
            "loss": 1.282649278640747,
            "loss_box_reg": 0.06222952902317047,
            "loss_classifier": 0.30682939291000366,
            "loss_mask": 0.6970193982124329,
            "loss_rpn_box": 0.038663312792778015,
            "loss_rpn_cls": 0.1471673548221588,
            "lr": 0.007706666666666667,
            "time": 0.2490077018737793
          }
        ]

        $ cat metrics.json | jq '.loss_mask'
        0.7126231789588928
        0.689423680305481
        0.6776131987571716
        ...

    """

    def __init__(self, cfg):
        self.json_file = os.path.join(cfg.OUTPUT_DIR, "metrics.json")
        self._period = cfg.TEST.WRITER_PERIOD
        self._last_write = -1

    def before_train(self):
        if self.trainer.start_iter == 0 : self._file_handle = open(self.json_file, "w")
        else : self._file_handle = open(self.json_file, "a")

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or next_iter % self._period == 0:
            self.save_file()

    def after_train(self):
        self._file_handle.close()

    def save_file(self):
        storage = self.trainer.storage
        to_save = defaultdict(dict)

        for k, (v, iteration) in storage.latest_with_smoothing(self._period).items():
            if iteration <= self._last_write:
                continue
            to_save[iteration][k] = v
        all_iters = sorted(to_save.keys())
        self._last_write = max(all_iters)

        for itr, scalars_per_iter in to_save.items():
            if self._last_write != itr : continue
            scalars_per_iter["iteration"] = itr
            self._file_handle.write(json.dumps(round_floats(scalars_per_iter), sort_keys=True) + "\n")
        self._file_handle.flush()
        try:
            os.fsync(self._file_handle.fileno())
        except AttributeError:
            pass

def round_floats(o):
    ret = dict()
    for k, v in o.items():
        if ('time' in k) or ('seconds' in k) : ret[k] = round(v, 4)
        elif 'loss' in k : ret[k] = round(v, 3)
        elif 'lr' in k : ret[k] = round(v, 7)
        else : ret[k] = v
        
    return ret