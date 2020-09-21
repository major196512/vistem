import os

from vistem.loader import MetadataCatalog, DatasetCatalog
from .load_data import load_tiny_annotations
from .meta import get_tiny_annotations_meta

__all__ = ['register_all_tiny_imagenet']

_PREDEFINED_SPLITS = {
    'tiny_train' : ('tiny-imagenet-200/train', '', True),
    'tiny_val' : ('tiny-imagenet-200/val', 'val_annotations.txt', False),
    'tiny_test' : ('tiny-imagenet-200/test', '', False),
}

def register_tiny_imagenet(name, metadata : dict, data_root : str, anno_dir : str, is_split : bool):
    DatasetCatalog.register(name, lambda: load_tiny_annotations(data_root, anno_dir, name, is_split))
    MetadataCatalog.get(name).set(
        data_root=data_root, annotation_dir=anno_dir, is_split=is_split, evaluator_type="tiny", **metadata
    )

def register_all_tiny_imagenet(root="./data"):
    for key, (data_root, anno_dir, is_split) in _PREDEFINED_SPLITS.items():
        register_tiny_imagenet(
                key,
                get_tiny_annotations_meta(),
                os.path.join(root, data_root),
                anno_dir,
                is_split
            )