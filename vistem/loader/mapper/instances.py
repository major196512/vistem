import torch
import numpy as np

from vistem.structures import Boxes, BoxMode, Instances, PolygonMasks

__all__ = ['instance_mapper']

def instance_mapper(dataset_dict : dict, transforms, image_size, seg_on=False):
    boxes = []
    masks = []
    classes = []
    for obj in dataset_dict.pop('instances'):
        if obj.get('iscrowd', 0) == 1 : continue

        bbox = BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS)
        boxes.append(transforms.apply_box([bbox])[0])

        if seg_on:
            assert 'segmentation' in obj
            seg = obj['segmentation']
            if isinstance(seg, list):
                polygons = [np.asarray(p).reshape(-1, 2) for p in seg]
                masks.append([p.reshape(-1) for p in transforms.apply_polygons(polygons)])
            else:
                ValueError(f'Cannot transform segmentaion of type : {type(seg)}')

        classes.append(obj['category_id'])

    instance = Instances(image_size)

    boxes = instance.gt_boxes = Boxes(boxes)
    boxes.clip(image_size)

    if seg_on:
        instance.gt_mask = PolygonMasks(masks)
        instance.gt_boxes = instance.gt_mask.get_bounding_boxes()

    classes = torch.tensor(classes, dtype=torch.int64)
    instance.gt_classes = classes

    dataset_dict["instances"] = filter_empty_instances(instance)

    return dataset_dict

def filter_empty_instances(instances, box_threshold=1e-5):
    r = []
    r.append(instances.gt_boxes.nonempty(threshold=box_threshold))

    if not r:
        return instances
    m = r[0]
    for x in r[1:]:
        m = m & x
    return instances[m]