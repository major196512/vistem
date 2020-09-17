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
`$PRETRAINED` : the directory of pretrained backbone network weights(download pretrained models from [here](https://drive.google.com/drive/folders/18xcK6pS3bapqHaU0UQ2jSQ1eE1bn0iuK?usp=sharing))

`$DATA` : the directory of datasets(See [here](./vistem/loader/data) for more information).

# Performance

## Pascal VOC
We train our models with 8-gpu and 16 images per batch.

| Meta <br>Architecture | Backbone <br>Network | BBox <br>AP | BBox <br>AP50 | BBox <br>AP75 | Config File |
| :---:                 | :---:                   | :---:       | :---:         | :---:         | :---:       |
| [RetinaNet](https://drive.google.com/file/d/17Ygzh4kVOQIwfpgejpvIhD9tCS1_yaqB/view?usp=sharing) | ResNet-50<br>with FPN | 55.533 | 81.730 | 60.504 | [retinanet_R50_FPN](./configs/VOC-Detection/retinanet_R50_FPN.yaml) |
| [RetinaNet]() | ResNet-50<br>with NAS-FPN | In Progress |  |  | [retinanet_R50_NASFPN](./configs/VOC-Detection/retinanet_R50_NASFPN.yaml) |
| [Faster RCNN](https://drive.google.com/file/d/1om1m7a77_ZYTcDCwEcXmv8Xx8IUgysYc/view?usp=sharing) | ResNet-50<br>with FPN | 54.282 | 81.827 | 60.048 | [faster_R50_FPN](./configs/VOC-Detection/faster_R50_FPN.yaml) |
| [Faster RCNN]() | ResNet-50<br>with NAS-FPN | In Progress |  |  | [faster_R50_NASFPN](./configs/VOC-Detection/faster_R50_NASFPN.yaml) |
| CornerNet | ResNet-50<br>with FPN | In Progress | | | |
| RepPoints | ResNet-50<br>with FPN | In Progress | | | |

When training using `Gradient Accumulation`, you must assign a `cfg.SOLVER.ACUUMULATE` and `cfg.SOLVER.IMG_PER_BATCH` in config file.
In this table below, we run our models with 4 gradient accumulation and 4 images per batch with 2-gpu.

| Meta <br>Architecture | Backbone <br>Network | BBox <br>AP   | BBox <br>AP50 | BBox <br>AP75 |
| :---:             | :---:     | :---:     | :---:     | :---:     | :---:         |
| [RetinaNet](https://drive.google.com/file/d/17akQ5GgxWgVYWZb57rzjEQZo0ZF197zI/view?usp=sharing) | ResNet-50<br>with FPN | 51.011 | 79.542 | 54.105 |
| [RetinaNet]() | ResNet-50<br>with NAS-FPN | In Progress |  |  |
| [Faster RCNN](https://drive.google.com/file/d/1228vNhWED2M_Iv0tT0LpUHp5CIf6rypa/view?usp=sharing) | ResNet-50<br>with FPN | 49.928 | 80.683 | 53.101 |
| [Faster RCNN]() | ResNet-50<br>with NAS-FPN | In Progress |  |  |
| CornerNet | ResNet-50<br>with FPN | In Progress | | |
| RepPoints | ResNet-50<br>with FPN | In Progress | | |

## MS-COCO
In training MS-COCO datasets, We only evaluate with 8-gpu settings.
| Meta <br>Architecture | Backbone <br>Network | BBox <br>AP   | Config File   |
| :---:                 | :---:         | :---:         | :---:         |
| [RetinaNet](https://drive.google.com/file/d/1Tyq3O56WkbdVVOpTBNlcC1vf620Z6Czv/view?usp=sharing) | ResNet-50<br>with FPN | 36.524 | [retinanet_R50_FPN](./configs/COCO-Detection/retinanet_R50_FPN.yaml) |
| [RetinaNet]() | ResNet-50<br>with FPN | In Progress | [retinanet_R50_NASFPN](./configs/COCO-Detection/retinanet_R50_NASFPN.yaml) |
| [Faster RCNN](https://drive.google.com/file/d/1fC1G--BwGabal2Pe1rFt2m_WZUmgKcdT/view?usp=sharing) | ResNet-50<br>with FPN | 38.021 | [faster_R50_FPN](./configs/COCO-Detection/faster_R50_FPN.yaml) |
| [Faster RCNN]() | ResNet-50<br>with NAS-FPN | In Progress | [faster_R50_NASFPN](./configs/COCO-Detection/faster_R50_NASFPN.yaml) |
| CornerNet              | ResNet-50<br>with FPN| In Progress|          |
| RepPoints              | ResNet-50<br>with FPN| In Progress|          |

# Training
## Single Machine

When training in a single machine, you should only specify `--config-file` and `--num-gpu` in argument.
You can select the training model and datasets by using or modifying a [config file](./configs). For more information about factors in config, see [here](./vistem/config/defaults/README.md).

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
python tools/train.py --config-file ./configs/train.yaml --num-gpu 4 --num-machine 2 --machine-rank 1 --dist-ip xxx.xxx.xxx.xxx --dist-port xxxx
```

# Evaluation
```bash
python tools/test.py --config-file ./configs/test.yaml --eval-only
```
