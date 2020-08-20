# Syncronize with COCO Dataset IDs
PASCAL_CATEGORIES = [
    {'id' : 5, 'name' : 'aeroplane'},
    {'id' : 2, 'name' : 'bicycle'},
    {'id' : 16, 'name' : 'bird'},
    {'id' : 9, 'name' : 'boat'},
    {'id' : 44, 'name' : 'bottle'},
    {'id' : 6, 'name' : 'bus'},
    {'id' : 3, 'name' : 'car'},
    {'id' : 17, 'name' : 'cat'},
    {'id' : 62, 'name' : 'chair'},
    {'id' : 21, 'name' : 'cow'},
    {'id' : 67, 'name' : 'diningtable'},
    {'id' : 18, 'name' : 'dog'},
    {'id' : 19, 'name' : 'horse'},
    {'id' : 4, 'name' : 'motorbike'},
    {'id' : 1, 'name' : 'person'},
    {'id' : 64, 'name' : 'pottedplant'},
    {'id' : 20, 'name' : 'sheep'},
    {'id' : 63, 'name' : 'sofa'},
    {'id' : 7, 'name' : 'train'},
    {'id' : 72, 'name' : 'tvmonitor'},
]

__all__ = ['get_pascal_instances_meta']

def get_pascal_instances_meta():
    thing_ids = [k["id"] for k in PASCAL_CATEGORIES]
    category_names = [k["name"] for k in PASCAL_CATEGORIES]
    assert len(thing_ids) == 20, len(thing_ids)

    # Mapping from the incontiguous Pascal VOC category id to an id in [0, 19]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    
    # Not use dataset_id_to_contiguous_id in Evaluation
    ret = {
        "dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "category_names": category_names,
    }
    return ret