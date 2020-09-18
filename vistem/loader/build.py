import torch
import numpy as np
import itertools
import operator

from vistem.utils.env import seed_all_rng
from vistem import dist

from .catalog import DatasetCatalog, MetadataCatalog
from .dataset import ListDataset, MapDataset, AspectRatioGroupedDataset
from .mapper import DatasetMapper
from .sampler import IterSampler, InferenceSampler

__all__ = ['build_train_loader', 'build_test_loader']

def build_train_loader(cfg):
    images_per_batch = cfg.SOLVER.IMG_PER_BATCH
    assert images_per_batch >= dist.get_world_size()
    assert images_per_batch % dist.get_world_size() == 0

    data = [DatasetCatalog.get(train_dataset) for train_dataset in cfg.LOADER.TRAIN_DATASET]
    data = list(itertools.chain.from_iterable(data))

    dataset = ListDataset(cfg, data)
    mapper = DatasetMapper(cfg, is_train=True)
    dataset = MapDataset(dataset, mapper)

    sampler = IterSampler(cfg, dataset)
    batch_sampler = torch.utils.data.sampler.BatchSampler(
                sampler, images_per_batch // dist.get_world_size(), drop_last=True
            )  # drop last so the batch always have the same size

    if cfg.LOADER.ASPECT_GROUPING:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            batch_sampler=None,
            num_workers=cfg.LOADER.NUM_WORKERS,
            collate_fn=operator.itemgetter(0),
            worker_init_fn=worker_init_reset_seed,
        )
        return AspectRatioGroupedDataset(data_loader, images_per_batch // dist.get_world_size())
    
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
                sampler, images_per_batch // dist.get_world_size(), drop_last=True
            )  # drop last so the batch always have the same size

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=cfg.LOADER.NUM_WORKERS,
            collate_fn=trivial_batch_collator,
            worker_init_fn=worker_init_reset_seed,
        )
        return data_loader

def build_test_loader(cfg):
    images_per_batch = cfg.SOLVER.IMG_PER_BATCH
    assert images_per_batch >= dist.get_world_size()
    assert images_per_batch % dist.get_world_size() == 0

    data = [DatasetCatalog.get(cfg.LOADER.TEST_DATASET)]
    data = list(itertools.chain.from_iterable(data))

    dataset = ListDataset(cfg, data)
    mapper = DatasetMapper(cfg, is_train=False)
    dataset = MapDataset(dataset, mapper)

    sampler = InferenceSampler(len(dataset))
    batch_sampler = torch.utils.data.sampler.BatchSampler(
                        sampler, 
                        1 if cfg.LOADER.TEST_SINGLE_IMG else images_per_batch // dist.get_world_size(), 
                        drop_last=False
                    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=cfg.LOADER.NUM_WORKERS,
        collate_fn=trivial_batch_collator,
        worker_init_fn=worker_init_reset_seed,
    )
    return data_loader

def trivial_batch_collator(batch):
    return batch

def worker_init_reset_seed(worker_id):
    seed_all_rng(np.random.randint(2 ** 31) + worker_id)
