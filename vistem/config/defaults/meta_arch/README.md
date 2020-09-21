## Defaults

- `NAME`(str) : Choose one of training model architecture.

- `PROPOSAL_GENERATOR`(str) : Choose one of proposal generator.

- `NUM_CLASSES`(int) : The number of output classes.

## RESNETS

- `ENABLE`

- `IN_FEATURES`(List[str]) : Input levels of features from backbone network.
- 
## RETINANET

- `ENABLE`

- `IN_FEATURES`(List[str]) : Input levels of features from backbone network.

- `MATCHER` :

    - `IOU_THRESHOLDS`(List[float]) : Labeling points from IOUs between anchors and ground-truth bounding box.

    - `IOU_LABELS`(List[float]) : Labeling values from IOU_THRESHOLDS to clarify if a candidate is positive, negative or ignore label. 

    - `LOW_QUALITY_MATCHES`(bool) : Toggle for Producing additional matches for predictions that have only low-quality matches.

- `LOSS` :
    
    - `FOCAL_ALPHA`(float) : Alpha factor of focal loss.

    - `FOCAL_GAMMA`(float) : gamma factor of focal loss.

    - `SMOOTH_L1_BETA`(float) : beta factor of smooth l1 loss.

    - `NORMALIZER`(float) : Init of loss normalizer. This factor refers to [RetinaNet in Detectron2](https://github.com/facebookresearch/detectron2/blob/master/detectron2/modeling/meta_arch/retinanet.py).

    - `NORMALIZER_MOMENTUM`(float) : Momentum factor of loss normalizer.

- `HEAD` :
    
    - `NUM_CONVS`(int) : The number of classification and regression convolution layers.

    - `PRIOR_PROB`(float) : The probability of detection an object. See [Network Initialization(Sec 5.1) in Focal Loss](https://arxiv.org/abs/1708.02002).

- `TEST` :

    - `BBOX_REG_WEIGHTS`(List[float]) : Reression weights of (x, y, w, h) in [box_transform.py](../../../modeling/box_transform.py).

    - `TOPK_CANDIDATES`(int) : Maximum number of candidate boxes in each feature maps before NMS.

    - `NMS_THRESH`(float) : Threshold in NMS.

## ROI

- `ENABLE`

- `NAME`(str) : Choose one of ROI Heads.

- `IN_FEATURES`(List[str]) : Input levels of features from backbone network.

- `MATCHER` :

    - `IOU_THRESHOLDS`(List[float]) : Labeling points from IOUs between anchors and ground-truth bounding box.

    - `IOU_LABELS`(List[float]) : Labeling values from IOU_THRESHOLDS to clarify if a candidate is positive, negative or ignore label.

    - `LOW_QUALITY_MATCHES`(bool) : Toggle for Producing additional matches for predictions that have only low-quality matches.

- `SAMPLING` :

    - `PROPOSAL_APPEND_GT`(bool) : Toggle whether appending ground-truth in proposals or not because of low quality when training starts.

    - `BATCH_SIZE_PER_IMAGE`(int) : The number of samples per images.

    - `POSITIVE_FRACTION`(float) : Ratio of positive and negative samples.

- `BOX_POOLING` :

    - `TYPE`(str) : Choose one of pooling functions in {'ROIPool', 'ROIAlign', 'ROIAlignV2'}.

    - `RESOLUTION`(int) : Output size after pooling.

    - `SAMPLING_RATIO`(int) : Sampling ratio for grid bin in ROIAlign.

- `BOX_HEAD` :

    - `NUM_CONV`(int) : The number of convolution layers.

    - `CONV_DIM`(int) : Dimension of convolusion layers.

    - `CONV_NORM`(str) : Choose one of [normized functions](../../../modeling/layers/norm/__init__.py).

    - `NUM_FC`(int) : The number of fully-connected layers.

    - `FC_DIM`(int) : Dimension of fully-connected layers.

- `BOX_LOSS` :

    - `LOSS_WEIGHT`(Union[Dict[str : float], float]) : Normalizer weights for {loss_cls, loss_loc}.

    - `SMOOTH_L1_BETA`(float) : Factor of beta in smooth l1 loss.

- `TEST` :

    - `BBOX_REG_WEIGHTS`(Tuple[float]) : Reression weights of (x, y, w, h) in [box_transform.py](../../../modeling/box_transform.py).

    - `NMS_THRESH`(float) : Threshold in NMS.

<!-- TRAIN_ON_PRED_BOXES(bool : False)-->

## RPN

- `ENABLE`

- `HEAD_NAME`(str) : Choose one of the RPN Heads.

- `IN_FEATURES`(List[str]) : Input levels of features from backbone network

- `MATCHER` : 

    - `IOU_THRESHOLDS`(List[float]) : Labeling points from IOUs between anchors and ground-truth bounding box.

    - `IOU_LABELS`(List[int]) : Labeling values from IOU_THRESHOLDS to clarify if a candidate is positive, negative or ignore label.

    - `LOW_QUALITY_MATCHES`(bool) : Toggle for Producing additional matches for predictions that have only low-quality matches.

- `SAMPLING` : 

    - `BATCH_SIZE_PER_IMAGE`(int) : The number of samples per images.

    - `POSITIVE_FRACTION`(float) : Ratio of positive and negative samples.

- `LOSS` : 

    - `LOC_TYPE`(str) : Choose one of functions for loss_cls.

    - `SMOOTH_L1_BETA`(float) : Beta factor of smooth l1 loss.

    - `CLS_WEIGHT`(float) : Normalizer weights for loss_cls.

    - `LOC_WEIGHT`(float) : Normalizer weights for loss_loc.

- `TRAIN` : 

    - `PRE_NMS_TOPK`(int) : Maximum number of candidates in each feature maps before NMS when model.training=True.

    - `POST_NMS_TOPK`(int) : Maximum number of candidates after NMS when model.training=True.

- `TEST` : 

    - `BBOX_REG_WEIGHTS`(Tuple[float]) : Reression weights of (x, y, w, h) in [box_transform.py](../../../modeling/box_transform.py)

    - `NMS_THRESH`(float) : Threshold in NMS.

    - `PRE_NMS_TOPK`(int) : Maximum number of candidates in each feature maps before NMS when model.training=False.

    - `POST_NMS_TOPK`(int) : Maximum number of candidates after NMS when model.training=False.

    - `MIN_SIZE`(int) : Minimum area of candidates.


<!--_RPN.BOUNDARY_THRESH = -1-->

    