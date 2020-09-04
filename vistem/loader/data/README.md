# Data Structure
`load_data.py` : loading datasets into data structure below.

    - file_name(str)
    - height(int)
    - width(int)
    - image_id(int)
    - annotations(List[Dict])
        - category_id(int)
        - bbox(List[float])
        - bbox_mode(BoxMode)
        - iscrowd(bool) : only MS-COCO
        - difficult(bool) : only in Pascal VOC

# MetaData
`meta.py` : importing predefined annotation information(eg. Object classes and ids)

    - dataset_id_to_contiguous_id
    - category_names(List[str])
    - thing_colors(List[List[int]]) : only in MS-COCO

# Directory
`register.py` : mapping dataset names with loader functions and data directories.

```bash
+---VOC2007/
|   +---Annotations/
|   +---ImageSets/
|   |   +---Layout/
|   |   +---Main/
|   |   \---Segmentation/
|   +---JPEGImages/
|   +---SegmentationClass/
|   \---SegmentationObject/
|
+---VOC2012/
|   +---Annotations/
|   +---ImageSets/
|   |   +---Action/
|   |   +---Layout/
|   |   +---Main/
|   |   \---Segmentation/
|   +---JPEGImages/
|   +---SegmentationClass/
|   \---SegmentationObject/
|
+---coco2014/
|   +---annotations/
|   |   +---captions_train2014.json
|   |   +---captions_val2014.json
|   |   +---image_info_test2014.json
|   |   +---instances_minival2014.json
|   |   +---instances_train2014.json
|   |   +---instances_val2014.json
|   |   +---person_keypoints_train2014.json
|   |   \---person_keypoints_val2014.json
|   \---images/
|       +---test2014/
|       +---train2014/
|       \---val2014/
|
\---coco2017/
    +---annotations/
    |   +---captions_train2017.json
    |   +---captions_val2017.json
    |   +---instances_train2017.json
    |   +---instances_val2017.json
    |   +---person_keypoints_train2017.json
    |   +---person_keypoints_val2017.json
    |   \---temp.json
    \---images/
        +---train2017/
        \---val2017/
```