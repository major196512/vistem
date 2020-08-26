# Datasets

Several datasets(MS-COCO, Pascal VOC) in each directories.

- `meta.py` : importing predefined annotation information(eg. Object classes and ids)
- `load_data.py` : loading datasets into data structure below.
- `register.py` : mapping dataset names with loader functions and data directories.

## Data Structure
- file_name
- height
- width
- image_id
- annotations
    - iscrowd(only MS-COCO)
    - type : List
        - bbox
        - category_id
        - bbox_mode
        - difficult(only in Pascal VOC)