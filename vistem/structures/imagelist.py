from __future__ import division
from typing import Any, List, Sequence, Tuple, Union
import torch
from torch.nn import functional as F

class ImageList(object):
    def __init__(self, tensor: torch.Tensor, image_sizes: List[Tuple[int, int]]):
        self.tensor = tensor
        self.image_sizes = image_sizes

    def __len__(self) -> int:
        return len(self.image_sizes)

    def __getitem__(self, idx: Union[int, slice]) -> torch.Tensor:
        size = self.image_sizes[idx]
        return self.tensor[idx, ..., : size[0], : size[1]]  # type: ignore

    def to(self, *args: Any, **kwargs: Any) -> "ImageList":
        cast_tensor = self.tensor.to(*args, **kwargs)
        return ImageList(cast_tensor, self.image_sizes)

    @staticmethod
    def from_tensors(
        tensors: Sequence[torch.Tensor], size_divisibility: int = 0, pad_value: float = 0.0
    ) -> "ImageList":
        assert len(tensors) > 0
        assert isinstance(tensors, (tuple, list))
        for t in tensors:
            assert isinstance(t, torch.Tensor), type(t)
            assert t.shape[1:-2] == tensors[0].shape[1:-2], t.shape
        # per dimension maximum (H, W) or (C_1, ..., C_K, H, W) where K >= 1 among all tensors
        max_size = tuple(max(s) for s in zip(*[img.shape for img in tensors]))

        if size_divisibility > 0:
            import math

            stride = size_divisibility
            max_size = list(max_size)  # type: ignore
            max_size[-2] = int(math.ceil(max_size[-2] / stride) * stride)  # type: ignore
            max_size[-1] = int(math.ceil(max_size[-1] / stride) * stride)  # type: ignore
            max_size = tuple(max_size)

        image_sizes = [im.shape[-2:] for im in tensors]

        if len(tensors) == 1:
            # This seems slightly (2%) faster.
            # TODO: check whether it's faster for multiple images as well
            image_size = image_sizes[0]
            padded = F.pad(
                tensors[0],
                [0, max_size[-1] - image_size[1], 0, max_size[-2] - image_size[0]],
                value=pad_value,
            )
            batched_imgs = padded.unsqueeze_(0)
        else:
            batch_shape = (len(tensors),) + max_size
            batched_imgs = tensors[0].new_full(batch_shape, pad_value)
            for img, pad_img in zip(tensors, batched_imgs):
                pad_img[..., : img.shape[-2], : img.shape[-1]].copy_(img)

        return ImageList(batched_imgs.contiguous(), image_sizes)
