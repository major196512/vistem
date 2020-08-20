<p align="center"><img width="40%" src="./img/pytorch.png"></p>

## Requirements
- PyTorch : >=1.5
- TensorFlow : >=1.15.0
- CUDA : >=10.2

## Setup
```bash
cd ..
python -m pip install -e vistem
cd vistem
ln -s $PRETRAINED pretrained_weights
ln -s $DATA data
```

## Training(Single Machine)
```bash
python tools/train.py --config-file ./configs/train.yaml --num-gpu 8
```

```bash
python tools/train.py --config-file ./configs/train.yaml --num-gpu 8 --resume
```

## Training(Multi Machine)
```bash
python tools/train.py --config-file ./configs/train.yaml --num-gpu 4 --num-machine 2 --machine-rank 0
```

```bash
python tools/train.py --config-file ./configs/train.yaml --num-gpu 4 --dist-ip xxx.xxx.xxx.xxx dist-port xxxx --num-machine 2 --machine-rank 1
```

## Evaluation
```bash
python tools/test.py --config-file ./configs/test.yaml --eval-only
```