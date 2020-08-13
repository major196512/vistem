import torch
import copy
import numpy as np
from PIL import Image, ImageOps

from vistem.utils.logger import setup_logger
from vistem.structures import Boxes, BoxMode, Instances

from . import transforms as T

__all__ = ["DatasetMapper"]


class DatasetMapper:
    def __init__(self, cfg, is_train=True):
        self._logger = setup_logger()

        # if cfg.INPUT.CROP.ENABLED and is_train:
            # self.crop_gen = T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
            # self._logger.info(f"CropGen used in training: {str(self.crop_gen)}")
            # pass
        # else:
        self.crop_gen = None

        self.tfm_gens = self.build_transform_gen(cfg, is_train)

        self.img_format = cfg.INPUT.FORMAT
        self.is_train = is_train

        

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = self.read_image(dataset_dict["file_name"])
        self.check_image_size(dataset_dict, image)

        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        image_shape = image.shape[:2]
        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            annos = [
                self.transform_instance_annotations(obj, transforms)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = self.annotations_to_instances(annos, image_shape)
            dataset_dict["instances"] = self.filter_empty_instances(instances)


        return dataset_dict

    def read_image(self, file_name):
        image = Image.open(file_name)

        image = ImageOps.exif_transpose(image)

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
                self._logger.critical(f"Mismatched (W,H){file_name}, got {image_wh}, expect {expected_wh}")

        if "width" not in dataset_dict:
            dataset_dict["width"] = image.shape[1]
        if "height" not in dataset_dict:
            dataset_dict["height"] = image.shape[0]

    def build_transform_gen(self, cfg, is_train):
        if is_train:
            min_size = cfg.INPUT.MIN_SIZE_TRAIN
            max_size = cfg.INPUT.MAX_SIZE_TRAIN
            sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        else:
            min_size = cfg.INPUT.MIN_SIZE_TEST
            max_size = cfg.INPUT.MAX_SIZE_TEST
            sample_style = "choice"

        if sample_style == "range":
            assert len(min_size) == 2, f"more than 2 ({len(min_size)}) min_size(s) are provided for ranges"

        tfm_gens = []
        tfm_gens.append(T.ResizeShortestEdge(min_size, max_size, sample_style))
        if is_train:
            tfm_gens.append(T.RandomFlip())
            self._logger.info("TransformGens used in training: " + str(tfm_gens))

        return tfm_gens

    def transform_instance_annotations(self, annotation, transforms):
        bbox = BoxMode.convert(annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
        annotation["bbox"] = transforms.apply_box([bbox])[0]
        annotation["bbox_mode"] = BoxMode.XYXY_ABS

        return annotation

    def annotations_to_instances(self, annos, image_size):
        boxes = [BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos]
        target = Instances(image_size)
        boxes = target.gt_boxes = Boxes(boxes)
        boxes.clip(image_size)

        classes = [obj["category_id"] for obj in annos]
        classes = torch.tensor(classes, dtype=torch.int64)
        target.gt_classes = classes

        return target

    def filter_empty_instances(self, instances):
        r = []
        r.append(instances.gt_boxes.nonempty())

        if not r:
            return instances
        m = r[0]
        for x in r[1:]:
            m = m & x
        return instances[m]
