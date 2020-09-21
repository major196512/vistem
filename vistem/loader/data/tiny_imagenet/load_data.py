import os
import numpy as np

from vistem.loader import MetadataCatalog

from vistem.utils.logger import setup_logger

def load_tiny_annotations(data_root: str, anno_dir: str, dataset_name: str, is_split: bool):
    _logger = setup_logger(__name__, all_rank=True)

    meta = MetadataCatalog.get(dataset_name)
    class_names = meta.category_names

    dataset_dicts = []

    if len(anno_dir):
        annotation_dirname = os.path.join(data_root, anno_dir)
        img_root = os.path.join(data_root, 'images')

        f = open(annotation_dirname, 'r')
        while True:
            line = f.readline()
            if not len(line) : break

            line = line.split('\t')
            file_name, cls_name = line[:2]
            img_id = int(file_name.split('.')[0].split('_')[1])

            record = {
                "file_name": os.path.join(img_root, file_name),
                "image_id": img_id,
                "annotations" : class_names.index(cls_name),
            }
            dataset_dicts.append(record)

    else:
        annos = class_names if is_split else ['']

        for anno in annos:
            img_root = os.path.join(data_root, anno, 'images')
            img_list = os.listdir(img_root)

            for file_name in img_list:
                img_id = int(file_name.split('.')[0].split('_')[1])
                record = {
                    "file_name": os.path.join(img_root, file_name),
                    "image_id": img_id,
                    "annotations" : class_names.index(anno) if len(anno) else -1,
                }
                dataset_dicts.append(record)

    _logger.info(f"Loaded {len(dataset_dicts)} images in Tiny ImageNet from {dataset_name}")

    return dataset_dicts
