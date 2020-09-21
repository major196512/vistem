import torch
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel

import os
import numpy as np
from collections import defaultdict

from vistem.utils import setup_logger, seed_all_rng, Timer
from vistem import dist

from vistem.hooks import HookTrainer, build_hooks
from vistem.loader import build_train_loader, build_test_loader

from vistem.modeling import build_model
from vistem.solver import build_optimizer, build_lr_scheduler
from vistem.checkpointer import Checkpointer
from vistem.evaluation import build_evaluator, evaluator

class Trainer(HookTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)

        if cfg.SEED < 0 : cfg.SEED = dist.shared_random_seed()
        self._seed = cfg.SEED
        seed_all_rng(self._seed)
        
        self._logger.debug(f'Config File : \n{cfg}')
        if cfg.OUTPUT_DIR and not os.path.isdir(cfg.OUTPUT_DIR) : os.makedirs(cfg.OUTPUT_DIR)
        with open(os.path.join(cfg.OUTPUT_DIR, 'config'), 'w') as f:
            f.write(cfg.dump())
        
        self.train_loader = build_train_loader(cfg)
        self.test_loader = build_test_loader(cfg)
        self.train_iter = iter(self.train_loader)

        self.model = build_model(cfg)
        self.model.train()
        if dist.is_main_process():
            self._logger.debug(f"Model Structure\n{self.model}")
        
        self.optimizer = build_optimizer(cfg, self.model)
        self.optimizer.zero_grad()
        self.scheduler = build_lr_scheduler(cfg, self.optimizer)
        self.accumulate = cfg.SOLVER.ACCUMULATE
        
        if dist.get_world_size() > 1:
            self.model = DistributedDataParallel(self.model, device_ids=[dist.get_local_rank()], broadcast_buffers=False)

        self.weight_path = cfg.WEIGHTS
        self.checkpointer = Checkpointer(
            self.model,
            cfg.OUTPUT_DIR,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )

        self.evaluator = build_evaluator(cfg)

        hooks = build_hooks(cfg, self.optimizer, self.scheduler, self.checkpointer)
        self.register_hooks(hooks)

    def resume_or_load(self, resume=True):
        self.start_iter = (
            self.checkpointer.resume_or_load(self.weight_path, resume=resume).get(
                "iteration", -1
            )
            + 1
        )

    def run_step(self):
        assert self.model.training, "model was changed to eval mode!"

        total_loss = defaultdict(list)
        total_timer = list()

        for _ in range(self.accumulate):
            timer = Timer()
            data = next(self.train_iter)
            total_timer.append(timer.seconds())
            timer.pause()
        
            loss_dict = self.model(data)
            losses = sum(loss_dict.values()) / self.accumulate
            losses.backward()

            if not torch.isfinite(losses).all():
                raise FloatingPointError(f"Loss became infinite or NaN at iteration={self.iter}!\nloss_dict = {loss_dict}")

            for k, v in loss_dict.items() : total_loss[k].append(v.item())

        # use a new stream so the ops don't wait for DDP
        with torch.cuda.stream(torch.cuda.Stream()) if losses.device.type == "cuda" else _nullcontext():
            metrics_dict = dict()
            for k, v in total_loss.items(): 
                metrics_dict[k] = sum(v) / self.accumulate
            metrics_dict["data_time"] = sum(total_timer) / self.accumulate
            self._write_metrics(metrics_dict)

        # for param in self.model.parameters():
        #     torch.distributed.all_reduce(param.grad.data)
        #     param.grad.data /= dist.get_world_size()
        self.optimizer.step()
        self.optimizer.zero_grad()


    def _write_metrics(self, metrics_dict: dict):
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        all_metrics_dict = dist.gather(metrics_dict)

        if dist.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
            }
            total_losses_reduced = sum(loss for loss in metrics_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)

    def test(self):
        return evaluator(self.model, self.test_loader, self.evaluator)

