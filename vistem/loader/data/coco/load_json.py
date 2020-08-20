from pycocotools.coco import COCO
import os
import io
import time
import contextlib

from vistem.loader import MetadataCatalog, DatasetCatalog

from vistem.utils.logger import setup_logger
from vistem.structures import BoxMode

__all__ = ['load_coco_json']

def load_coco_json(json_file, image_root, dataset_name, filter=True, extra_annotation_keys=None):    
    _logger = setup_logger(__name__, all_rank=True)

    start_time = time.time()
    with contextlib.redirect_stdout(io.StringIO()) : coco_api = COCO(json_file)
    end_time = time.time()
    if end_time - start_time > 1:
        _logger.info(f"Loading {json_file} takes {end_time - start_time:.2f} seconds.")

    meta = MetadataCatalog.get(dataset_name)

    cat_ids = sorted(coco_api.getCatIds())
    cats = coco_api.loadCats(cat_ids)
    category_names = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
    meta.category_names = category_names

    id_map = {v: i for i, v in enumerate(cat_ids)}
    meta.dataset_id_to_contiguous_id = id_map

    img_ids = sorted(list(coco_api.imgs.keys()))
    # list : 'license', 'url', 'file_name', 'height', 'width', 'id', 'date_captured'
    imgs = coco_api.loadImgs(img_ids)
    # list : 'segmentation', 'area', 'iscrowd', 'image_id', 'bbox', 'category_id', 'id'
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]

    imgs_anns = list(zip(imgs, anns))
    _logger.info(f"Loaded {len(imgs_anns)} images in COCO format from {json_file}")

    dataset_dicts = []

    ann_keys = ["iscrowd", "bbox", "category_id"] + (extra_annotation_keys or [])

    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            assert anno["image_id"] == image_id
            assert anno.get("ignore", 0) == 0

            obj = {key: anno[key] for key in ann_keys if key in anno}
            obj["bbox_mode"] = BoxMode.XYWH_ABS
            obj["category_id"] = id_map[obj["category_id"]]
            objs.append(obj)
            
        record["annotations"] = objs
        dataset_dicts.append(record)

    if filter : dataset_dicts = filter_images_with_only_crowd_annotations(dataset_dicts)

    return dataset_dicts

def filter_images_with_only_crowd_annotations(dataset_dicts):
    _logger = setup_logger(__name__, all_rank=True)
    
    num_before = len(dataset_dicts)

    def valid(anns):
        for ann in anns:
            if ann.get("iscrowd", 0) == 0:
                return True
        return False

    dataset_dicts = [x for x in dataset_dicts if valid(x["annotations"])]
    num_after = len(dataset_dicts)
    _logger.info(f"Removed {num_before - num_after} images with no usable annotations. {num_after} images left.")
    return dataset_dicts