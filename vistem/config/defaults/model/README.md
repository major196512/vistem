## Defaults

- `META_ARCHITECTURE`(str) : Choose one of training model architecture.

- `BACKBONE`(str) : The backbone network which is used in META_ARCHITECTURE.

- `PROPOSAL_GENERATOR`(str) : Choose one of proposal generator.

- `WEIGHTS`(str) : The directory of a loaded model if specified.

- `PIXEL_MEAN`(Tuple[float]) : Mean of Gaussian Distribution of pixel colors.

- `PIXEL_STD`(Tuple[float]) : Standard deviation of Gaussian Distribution of pixel colors.

## ANCHOR_GENERATOR

- `NAME`(str) : Choose a module for anchor generator.

- `SIZES`(List[List[int]]) : Anchor size for each feature levels.

- `ASPECT_RATIOS`(List[List[float]]) : Anchor ratio of heights and widths for each feature levels.

## FPN

- `IN_FEATURES`(List[str]) : Input levels of features from bottom-up pathway network.

- `OUT_CHANNELS`(int) : Dimension of each output feature maps.

- `NORM`(str) : Choose one of [normized functions](../../../modeling/layers/norm/__init__.py).

- `FUSE_TYPE`(str) : Fuse feature map from bottom-up pathway and top-down pathway model by 'sum' or 'avg'.

## RESNETS

- `FREEZE_AT`(int) : Index of layers to freeze layers.

- `DEPTH`(int) : Select the number of layers{18, 34, 50, 101, 152}.

- `NUM_CLASSES`(int) : The number of classes. If NUM_CLASSES = 0, they only return the set of feature maps in each layer, not for Image Classification.

- `OUT_FEATURES`(List[str]) : The list of output layer names.

- `NORM`(str) : Choose one of [normized functions](../../../modeling/layers/norm/__init__.py).

- `STEM_OUT_CHANNELS`(int) : Dimension of feature from a stem layer.

- `RES2_OUT_CHANNELS`(int) : Dimension of each output feature maps after stem layer.

- `RES5_DILATION`(int) : The dilation value in 3x3 conv layer of 'Res5' layer. It must be in {1, 2}.

- `NUM_GROUPS`(int) : The number of groups in 3x3 conv layers.

- `WIDTH_PER_GROUP`(int) : Dimension of each groups in 3x3 conv layers(So WIDTH_PER_GROUP * NUM_GROUPS = BLOCK_CHANNEL).

- `STRIDE_IN_1X1`(bool) : Whether stride in 1x1 conv layer or 3x3 conv.

## RETINANET

- `IN_FEATURES`(List[str]) : Input levels of features from backbone network.

- `NUM_CLASSES`(int) : The number of classes.

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

- `NAME`(str) : Choose one of ROI Heads.

- `IN_FEATURES`(List[str]) : Input levels of features from backbone network.

- `NUM_CLASSES`(int) : The number of classes.

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

    