import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
import weakref

from vistem.utils import setup_logger, seed_all_rng
from vistem import dist

from vistem import hooks
from vistem.loader import build_train_loader, build_test_loader

from vistem.modeling import build_model
from vistem.solver import build_optimizer, build_lr_scheduler
from vistem.checkpointer import Checkpointer
from vistem.evaluation import build_evaluator, evaluator

class Trainer:
    def __init__(self, cfg):
        if cfg.SEED < 0 : cfg.SEED = dist.shared_random_seed()
        self._seed = cfg.SEED
        seed_all_rng(self._seed)
        
        self._logger = setup_logger()
        self._logger.debug(f'Config File : \n{cfg}')

        # self.train_loader = build_train_loader(cfg)
        self.test_loader = build_test_loader(cfg)

        self.model = build_model(cfg)
        if dist.is_main_process():
            self._logger.debug(f"Model Structure\n{self.model}")
        
        self.optimizer = build_optimizer(cfg, self.model)
        self.scheduler = build_lr_scheduler(cfg, self.optimizer)
        
        if dist.get_world_size() > 1:
            self.model = DistributedDataParallel(self.model, device_ids=[dist.get_local_rank()], broadcast_buffers=False)

        self.weight_path = cfg.MODEL.WEIGHTS
        self.checkpointer = Checkpointer(
            self.model,
            cfg.OUTPUT_DIR,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )

        self.evaluator = build_evaluator(cfg)
        
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER

    def register_hooks(self, hooks):
        hooks = [h for h in hooks if h is not None]
        for h in hooks:
            assert isinstance(h, HookBase)
            h.trainer = weakref.proxy(self)
        self._hooks.extend(hooks)

    def resume_or_load(self, resume=True):
        self.start_iter = (
            self.checkpointer.resume_or_load(self.weight_path, resume=resume).get(
                "iteration", -1
            )
            + 1
        )

    def test(self):
        evaluator(self.model, self.test_loader, self.evaluator)

    # def train(self):
    #     self.model.train()
    #     for curr_iter in range(self.max_iter):
    #         self.optimizer.zero_grad()
    #         loader_iter = iter(self.train_loader)
    #         data = next(loader_iter)

    #         loss = self.model(data)
    #         print(loss)
    #         loss.backward()
    #         self.optimizer.step()

    #         if dist.is_main_process() and curr_iter % 10 == 0:
    #             self._logger.info(f'iteration({curr_iter}) : loss({loss.item():.5f})')
