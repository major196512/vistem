import os

from vistem.data import MetadataCatalog, DatasetCatalog
from .load_json import load_coco_json
from .meta import get_coco_instances_meta

__all__ = ['register_all_coco']

_PREDEFINED_SPLITS = {
    "coco_2014_train": ("coco2014/images/train2014", "coco2014/annotations/instances_train2014.json"),
    "coco_2014_val": ("coco2014/images/val2014", "coco2014/annotations/instances_val2014.json"),
    "coco_2014_minival": ("coco2014/images/val2014", "coco2014/annotations/instances_minival2014.json"),
    "coco_2014_minival_100": ("coco2014/images/val2014", "coco2014/annotations/instances_minival2014_100.json"),
    "coco_2014_valminusminival": (
        "coco2014/images/val2014",
        "coco2014/annotations/instances_valminusminival2014.json",
    ),
    "coco_2017_train": ("coco2017/images/train2017", "coco2017/annotations/instances_train2017.json"),
    "coco_2017_val": ("coco2017/images/val2017", "coco2017/annotations/instances_val2017.json"),
}

def register_coco(name, metadata, json_file, image_root):
    DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root, name))
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="coco", **metadata
    )

def register_all_coco(root="./data"):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS.items():
        register_coco(
                key,
                get_coco_instances_meta(),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )