from .imagenet import register_all_imagenet
from .tiny_imagenet import register_all_tiny_imagenet
from .coco import register_all_coco
from .pascal_voc import register_all_pascal

root='./data'
register_all_imagenet(root)
register_all_tiny_imagenet(root)
register_all_coco(root)
register_all_pascal(root)