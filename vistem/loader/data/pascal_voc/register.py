import os

from vistem.loader import MetadataCatalog, DatasetCatalog
from .load_data import load_voc_instances
from .meta import get_pascal_instances_meta

__all__ = ['register_all_pascal']

_PREDEFINED_SPLITS = {
    'voc_2007_trainval' : ('VOC2007', 'trainval', True),
    'voc_2007_train' : ('VOC2007', 'train', True),
    'voc_2007_val' : ('VOC2007', 'val', False),
    'voc_2007_test' :  ('VOC2007', 'test', False),
    'voc_2012_trainval' : ('VOC2012', 'trainval', True),
    'voc_2012_train' : ('VOC2012', 'train', True),
    'voc_2012_val' : ('VOC2012', 'val', False),
}


def register_pascal(name, metadata, data_root, split, filter):
    DatasetCatalog.register(name, lambda: load_voc_instances(data_root, split, name, filter))
    MetadataCatalog.get(name).set(
        data_root=data_root, split=split, evaluator_type="voc", **metadata
    )

def register_all_pascal(root="./data"):
    for key, (data_root, split, filter) in _PREDEFINED_SPLITS.items():
        register_pascal(
                key,
                get_pascal_instances_meta(),
                os.path.join(root, data_root),
                split,
                filter=filter
            )