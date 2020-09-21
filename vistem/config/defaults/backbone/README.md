## Defaults

- `NAME`(str) : The backbone network which is used in META_ARCHITECTURE.

## RESNETS

- `ENABLE`

- `FREEZE_AT`(int) : Index of layers to freeze layers.

- `DEPTH`(int) : Select the number of layers{18, 34, 50, 101, 152}.

- `OUT_FEATURES`(List[str]) : The list of output layer names.

- `NORM`(str) : Choose one of [normized functions](../../../modeling/layers/norm/__init__.py).

- `STEM_OUT_CHANNELS`(int) : Dimension of feature from a stem layer.

- `RES2_OUT_CHANNELS`(int) : Dimension of each output feature maps after stem layer.

- `RES5_DILATION`(int) : The dilation value in 3x3 conv layer of 'Res5' layer. It must be in {1, 2}.

- `NUM_GROUPS`(int) : The number of groups in 3x3 conv layers.

- `WIDTH_PER_GROUP`(int) : Dimension of each groups in 3x3 conv layers(So WIDTH_PER_GROUP * NUM_GROUPS = BLOCK_CHANNEL).

- `STRIDE_IN_1X1`(bool) : Whether stride in 1x1 conv layer or 3x3 conv.

## FPN

- `ENABLE`

- `NAME`(str) : Choose a module for FPN.
  
- `IN_FEATURES`(List[str]) : Input levels of features from bottom-up pathway network.

- `OUT_CHANNELS`(int) : Dimension of each output feature maps.

- `NORM`(str) : Choose one of [normized functions](../../../modeling/layers/norm/__init__.py).

- `FUSE_TYPE`(str) : Fuse feature map from bottom-up pathway and top-down pathway model by 'sum' or 'avg'.

## NAS_FPN

- `ENABLE`
  
- `CELL_INPUTS`(List[List[str]]) : Input levels of features for each NAS_FPN cells.

- `CELL_OUTPUTS`(List[str]) : Output levels of features for each NAS_FPN cells.

- `CELL_OPS`(List[str]) : Choose one of binary operations for each cells.

- `NAS_OUTPUTS`(List[str]) : Final output features.

## PANET

- `ENABLE`

- `IN_FEATURES`(List[str]) : Input levels of features from bottom-up pathway network.
  
- `OUT_CHANNELS`(int) : Dimension of each output feature maps.