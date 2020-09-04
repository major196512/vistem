<p align="center"><img width="40%" src="./img/pytorch.png"></p>

# VISTEM : Vision Model Training Template
Train and evaluate all of present Object Detection and Segmentation models. 

Although previous projects implement multiple models(Detectron2, MMDetection and so on), these are missingsome of vision models.  Our goal is to implement all of models by only using our project.

# Installation
## Requirements
- PyTorch : >=1.5
- TensorFlow : >=1.15.0
- CUDA : >=10.2

## Setup
```bash
git clone https://github.com/major196512/vistem
python -m pip install -e vistem
cd vistem
ln -s $PRETRAINED pretrained_weights
ln -s $DATA data
```
where `$PRETRAINED` is the directory of pretrained weights and `$DATA` are of and datasets.

See [here](./vistem/loader/data) for more information about `$DATA`.

# Performance
We train models with 8-gpu and 16 images per batch.

When training using Gradient Accumulation, you must assign a accumulate variable in config file.

## Pascal VOC
| Meta Architecture | Accumulate| BBox AP   | BBox AP50 | BBox AP75 | Config File   |
| :---:             | :---:     | :---:     | :---:     | :---:     | :---:         |
| RetinaNet         | 1         | 55.533    | 81.730    | 60.504    | [R50_FPN_1x_8gpu](./configs/RetinaNet/VOC-Detection/R50_FPN_1x_8gpu.yaml) |
| RetinaNet         | 4         | 51.011    | 79.542    | 54.105    | [R50_FPN_1x_2gpu_4acc](./configs/RetinaNet/VOC-Detection/R50_FPN_1x_2gpu_4acc.yaml) |
| Faster RCNN       | In Progress|          | | | |
| CornerNet         | In Progress|          | | | |
| RepPoints         | In Progress|          | | | |

## MS-COCO
| Meta Architecture | BBox AP   | Config File   |
| :---:             | :---:     | :---:         |
| RetinaNet         | 36.524    | [R50_FPN_1x_8gpu](./configs/RetinaNet/COCO-Detection/R50_FPN_1x_8gpu.yaml) |
| Faster RCNN       | In Progress|          |
| CornerNet         | In Progress|          |
| RepPoints         | In Progress|          |

# Training
## Single Machine
```bash
python tools/train.py --config-file ./configs/RetinaNet/VOC-Detection/R50_FPN_1x_8gpu.yaml --num-gpu 8
```

If you want to resume training, just set `--resume` in argument.
```bash
python tools/train.py --config-file ./configs/RetinaNet/VOC-Detection/R50_FPN_1x_8gpu.yaml --num-gpu 8 --resume
```

## Multi Machine
[For collective communication](https://pytorch.org/tutorials/intermediate/dist_tuto.html#collective-communication) in pytorch, it needs to execute process in main machine.
They automatically set main machine IP address and unused port number for TCP communication.

For main process, you must set `machine-rank` to zero and `num-machine` to the number of machines.
```bash
python tools/train.py --config-file ./configs/train.yaml --num-gpu 4 --num-machine 2 --machine-rank 0
```

In other machines, you clarify `machine-rank` and must set `dist-ip` and `dist-port` arguments which is the same with main machine values.
```bash
python tools/train.py --config-file ./configs/train.yaml --num-gpu 4 --dist-ip xxx.xxx.xxx.xxx dist-port xxxx --num-machine 2 --machine-rank 1
```

# Evaluation
```bash
python tools/test.py --config-file ./configs/test.yaml --eval-only
```