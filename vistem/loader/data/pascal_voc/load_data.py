import os
import numpy as np
import xml.etree.ElementTree as ET

from vistem.loader import MetadataCatalog

from vistem.utils.logger import setup_logger
from vistem.structures import BoxMode

def load_voc_instances(dirname: str, split: str, dataset_name, filter=False):
    _logger = setup_logger(__name__, all_rank=True)

    meta = MetadataCatalog.get(dataset_name)
    class_names = meta.category_names

    with open(os.path.join(dirname, "ImageSets", "Main", split + ".txt")) as f:
        fileids = np.loadtxt(f, dtype=np.str)

    annotation_dirname = os.path.join(dirname, "Annotations/")
    dataset_dicts = []
    for fileid in fileids:
        anno_file = os.path.join(annotation_dirname, fileid + ".xml")
        jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".jpg")

        with open(anno_file) as f:
            tree = ET.parse(f)

        record = {
            "file_name": jpeg_file,
            "image_id": fileid,
            "height": int(tree.findall("./size/height")[0].text),
            "width": int(tree.findall("./size/width")[0].text),
        }

        objs = []
        for obj in tree.findall("object"):
            cls = obj.find("name").text

            difficult = int(obj.find("difficult").text)

            bbox = obj.find("bndbox")
            bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]

            # Original annotations are integers in the range [1, W or H]
            # Assuming they mean 1-based pixel indices (inclusive),
            # a box with annotation (xmin=1, xmax=W) covers the whole image.
            # In coordinate space this is represented by (xmin=0, xmax=W)
            bbox[0] -= 1.0
            bbox[1] -= 1.0

            objs.append(
                {"category_id": class_names.index(cls), "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS, 'difficult' : difficult}
            )

        record["annotations"] = objs
        dataset_dicts.append(record)

    _logger.info(f"Loaded {len(dataset_dicts)} images in PASCAL VOC from {dirname}_{split}")

    if filter : dataset_dicts = filter_images_with_difficult(dataset_dicts)

    return dataset_dicts

def filter_images_with_difficult(dataset_dicts):
    _logger = setup_logger(__name__, all_rank=True)
    
    num_before = 0
    num_after = 0

    for dataset in dataset_dicts:
        num_before += len(dataset['annotations'])
        dataset['annotations'] = [ann for ann in dataset['annotations'] if ann.get("difficult", 0) == 0]
        num_after += len(dataset['annotations'])
    
    # All Images have annotations after filtering
    _logger.info(f"Removed {num_before - num_after} annotations with difficult. {num_after} annotations left.")
    return dataset_dicts