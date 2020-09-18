import torch
import copy
import numpy as np
from PIL import Image, ImageOps

from .instances import instance_mapper
from .annotations import annotation_mapper

from vistem.utils.logger import setup_logger
from vistem.loader.transforms import build_transform_gen, apply_transform

__all__ = ["DatasetMapper"]

class DatasetMapper:
    def __init__(self, cfg, is_train=True):
        _logger = setup_logger(__name__)

        self.img_format = cfg.INPUT.FORMAT
        self.exif_transpose = cfg.INPUT.EXIF

        self.tfm_gens = build_transform_gen(cfg, is_train)
        _logger.info(f"TransformGens(Training={is_train}) : {str(self.tfm_gens)}")

        self.is_train = is_train

        # if cfg.INPUT.CROP.ENABLED and is_train:
            # self.crop_gen = T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
            # self._logger.info(f"CropGen used in training: {str(self.crop_gen)}")
            # pass
        # else:

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = self.read_image(dataset_dict["file_name"])
        self.check_image_size(dataset_dict, image)

        image, transforms = apply_transform(self.tfm_gens, image)
        image_shape = image.shape[:2]
        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("instances", None)
            return dataset_dict

        if "instances" in dataset_dict:
            dataset_dict = instance_mapper(dataset_dict, transforms, image_shape)
        if 'annotations' in dataset_dict:
            dataset_dict = annotation_mapper(dataset_dict)


        return dataset_dict

    def read_image(self, file_name):
        image = Image.open(file_name)
        if self.exif_transpose : image = ImageOps.exif_transpose(image)

        if self.img_format is not None:
            conversion_format = self.img_format
            if self.img_format == "BGR":
                conversion_format = "RGB"
            image = image.convert(conversion_format)
            
        image = np.asarray(image)
        if self.img_format == "BGR":
            image = image[:, :, ::-1]
        if self.img_format == "L":
            image = np.expand_dims(image, -1)
        return image

    def check_image_size(self, dataset_dict, image):
        if "width" in dataset_dict or "height" in dataset_dict:
            image_wh = (image.shape[1], image.shape[0])
            expected_wh = (dataset_dict["width"], dataset_dict["height"])
            
            if not image_wh == expected_wh:
                file_name = dataset_dict["file_name"] if 'file_name' in dataset_dict else ''
                _logger = setup_logger(__name__)
                _logger.critical(f"Mismatched (W,H){file_name}, got {image_wh}, expect {expected_wh}")

        if "width" not in dataset_dict:
            dataset_dict["width"] = image.shape[1]
        if "height" not in dataset_dict:
            dataset_dict["height"] = image.shape[0]



