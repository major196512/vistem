import os
import numpy as np

from vistem.loader import MetadataCatalog

from vistem.utils.logger import setup_logger

def load_imagenet_annotations(data_root: str, dataset_name: str):
    _logger = setup_logger(__name__, all_rank=True)

    meta = MetadataCatalog.get(dataset_name)
    class_names = meta.category_names

    dataset_dicts = []
    for anno in class_names:
        img_root = os.path.join(data_root, anno)
        img_list = os.listdir(img_root)

        for file_name in img_list:
            img_id = int(file_name.split('.')[0].split('_')[-1])
            record = {
                "file_name": os.path.join(img_root, file_name),
                "image_id": img_id,
                "annotations" : class_names.index(anno),
            }
            dataset_dicts.append(record)

    _logger.info(f"Loaded {len(dataset_dicts)} images in ImageNet from {dataset_name}")

    return dataset_dicts
