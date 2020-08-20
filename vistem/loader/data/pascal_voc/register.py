import os

from vistem.loader import MetadataCatalog, DatasetCatalog
from .load_data import load_voc_instances
from .meta import get_pascal_instances_meta

__all__ = ['register_all_pascal']

_PREDEFINED_SPLITS = {
    'voc_2007_trainval' : ('VOC2007', 'trainval', 2007, True),
    'voc_2007_train' : ('VOC2007', 'train', 2007, True),
    'voc_2007_val' : ('VOC2007', 'val', 2007, False),
    'voc_2007_test' :  ('VOC2007', 'test', 2007, False),
    'voc_2012_trainval' : ('VOC2012', 'trainval', 2012, True),
    'voc_2012_train' : ('VOC2012', 'train', 2012, True),
    'voc_2012_val' : ('VOC2012', 'val', 2012, False),
}


def register_pascal(name, metadata, data_root, split, year, filter):
    DatasetCatalog.register(name, lambda: load_voc_instances(data_root, split, name, filter))
    MetadataCatalog.get(name).set(
        data_root=data_root, split=split, year=year, evaluator_type="voc", **metadata
    )

def register_all_pascal(root="./data"):
    for key, (data_root, split, year, filter) in _PREDEFINED_SPLITS.items():
        register_pascal(
                key,
                get_pascal_instances_meta(),
                os.path.join(root, data_root),
                split,
                year,
                filter=filter
            )