import os

from vistem.loader import MetadataCatalog, DatasetCatalog
from .load_data import load_imagenet_annotations
from .meta import get_imagenet_annotations_meta

__all__ = ['register_all_imagenet']

_PREDEFINED_SPLITS = {
    'imagenet_train' : ('ILSVRC_2012/train',),
    'imagenet_val' : ('ILSVRC_2012/val', ),
}

def register_imagenet(name, metadata : dict, data_root : str):
    DatasetCatalog.register(name, lambda: load_imagenet_annotations(data_root, name))
    MetadataCatalog.get(name).set(
        data_root=data_root, evaluator_type="imagenet", **metadata
    )

def register_all_imagenet(root="./data"):
    for key, (data_root) in _PREDEFINED_SPLITS.items():
        register_imagenet(
                key,
                get_imagenet_annotations_meta(),
                os.path.join(root, data_root),
            )