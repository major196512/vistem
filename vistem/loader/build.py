import torch
import numpy as np
import itertools

from vistem.utils.env import seed_all_rng
from vistem import dist

from .catalog import DatasetCatalog, MetadataCatalog
from .dataset import ListDataset, MapDataset
from .mapper import DatasetMapper
from .sampler import IterSampler, InferenceSampler

__all__ = ['build_train_loader', 'build_test_loader']

def build_train_loader(cfg):
    images_per_batch = cfg.SOLVER.IMG_PER_BATCH
    assert images_per_batch >= dist.get_world_size()
    assert images_per_batch % dist.get_world_size() == 0

    data = [DatasetCatalog.get(cfg.LOADER.TRAIN_DATASET)]
    data = list(itertools.chain.from_iterable(data))

    dataset = ListDataset(cfg, data)
    mapper = DatasetMapper(cfg, is_train=True)
    dataset = MapDataset(dataset, mapper)

    sampler = IterSampler(cfg, dataset)
    batch_sampler = torch.utils.data.sampler.BatchSampler(
                sampler, images_per_batch // dist.get_world_size(), drop_last=True
            )  # drop last so the batch always have the same size

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.LOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
        worker_init_fn=worker_init_reset_seed,
    )
    return data_loader

def build_test_loader(cfg):
    data = [DatasetCatalog.get(cfg.LOADER.TEST_DATASET)]
    data = list(itertools.chain.from_iterable(data))

    dataset = ListDataset(cfg, data)
    mapper = DatasetMapper(cfg, is_train=False)
    dataset = MapDataset(dataset, mapper)

    sampler = InferenceSampler(len(dataset))
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.LOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )
    return data_loader

def trivial_batch_collator(batch):
    return batch

def worker_init_reset_seed(worker_id):
    seed_all_rng(np.random.randint(2 ** 31) + worker_id)
