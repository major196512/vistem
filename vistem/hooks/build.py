from .train_timer import TrainTimer
from .lr_scheduler import LRScheduler
from .checkpointer import PeriodicCheckpointer

from .iter_timer import IterTimer
from .json_writer import JSONWriter
from .tensorboard import TensorboardXWriter
from .wandb import WandbWriter

from .eval import EvalHook

from vistem import dist

def build_hooks(cfg, model, optimizer, scheduler, checkpointer):
    ret = list()
    ret.append(TrainTimer())
    ret.append(LRScheduler(cfg, optimizer, scheduler))

    if dist.is_main_process():
        ret.append(PeriodicCheckpointer(cfg, checkpointer))

    ret.append(EvalHook(cfg))

    if dist.is_main_process():
        ret.append(IterTimer(cfg))
        ret.append(JSONWriter(cfg))
        ret.append(WandbWriter(cfg, model))
        # ret.append(TensorboardXWriter(cfg))

    return ret