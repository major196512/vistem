import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union, Tuple, List

__all__ = ['unfold_2d', 'fold_2d']


def unfold_2d(
    features : torch.Tensor, # (B, C, H, W)
    kernel_size : Union[int, Tuple[int, int], List[int]] = 1, 
    padding : Union[int, Tuple[int, int], List[int]] = 0, 
    stride : int = 1, 
    bias : Union[torch.Tensor, None] = None
):
    device = features.device
    B, in_channels, H, W = features.shape
    if isinstance(kernel_size, int) : kernel_size = (kernel_size, kernel_size)
    if isinstance(padding, int) : padding = (padding, padding)

    out_features = F.unfold(features, kernel_size=kernel_size, padding=padding, stride=stride) # (B, -1, H*W)
    H = int((H-kernel_size[0]+2*padding[0]) / stride + 1)
    W = int((W-kernel_size[1]+2*padding[1]) / stride + 1)
    out_features = out_features.reshape(B, kernel_size[0]*kernel_size[1], -1, H, W)

    return out_features

def fold_2d(
    features : torch.Tensor, # (B, F, C, H, W)
    out_shape : Tuple[int, int], 
    kernel_size : Union[int, Tuple[int, int], List[int]] = 1, 
    padding : Union[int, Tuple[int, int], List[int]] = 0, 
    stride : int = 1, 
    bias : Union[torch.Tensor, None] = None
):
    device = features.device
    B, _, _, H, W = features.shape
    if isinstance(kernel_size, int) : kernel_size = (kernel_size, kernel_size)
    if isinstance(padding, int) : padding = (padding, padding)

    features = features.reshape(B, -1, H*W)
    out_features = F.fold(features, out_shape, kernel_size=kernel_size, padding=padding, stride=stride)

    return out_features
