import numpy as np
from enum import Enum, unique
from typing import Iterator, List, Tuple, Union
import torch

_RawBoxType = Union[List[float], Tuple[float, ...], torch.Tensor, np.ndarray]

__all__ = ['BoxMode', 'Boxes']

@unique
class BoxMode(Enum):
    """
    Enum of different ways to represent a box.

    Attributes:

        XYXY_ABS: (x0, y0, x1, y1) in absolute floating points coordinates.
            The coordinates in range [0, width or height].
        XYWH_ABS: (x0, y0, w, h) in absolute floating points coordinates.
        XYXY_REL: (x0, y0, x1, y1) in range [0, 1]. They are relative to the size of the image.
        XYWH_REL: (x0, y0, w, h) in range [0, 1]. They are relative to the size of the image.
    """

    XYXY_ABS = 0
    XYWH_ABS = 1
    XYXY_REL = 2
    XYWH_REL = 3

    @staticmethod
    def convert(box: _RawBoxType, from_mode: "BoxMode", to_mode: "BoxMode") -> _RawBoxType:
        """
        Args:
            box: can be a 4-tuple, 4-list or a Nx4 array/tensor.
            from_mode, to_mode (BoxMode)

        Returns:
            The converted box of the same type.
        """
        if from_mode == to_mode:
            return box

        original_type = type(box)
        single_box = isinstance(box, (list, tuple))
        if single_box:
            arr = np.array(box)
            assert arr.shape == (
                4,
            ), "BoxMode.convert takes either a 4-tuple/list or a Nx4 array/tensor"
        else:
            arr = box

        assert to_mode.value < 2 and from_mode.value < 2, "Relative mode not yet supported!"

        original_shape = arr.shape
        arr = arr.reshape(-1, 4)
        if to_mode == BoxMode.XYXY_ABS and from_mode == BoxMode.XYWH_ABS:
            arr[:, 2] += arr[:, 0]
            arr[:, 3] += arr[:, 1]
        elif from_mode == BoxMode.XYXY_ABS and to_mode == BoxMode.XYWH_ABS:
            arr[:, 2] -= arr[:, 0]
            arr[:, 3] -= arr[:, 1]
        else:
            raise RuntimeError("Cannot be here!")
        if single_box:
            return original_type(arr.flatten())
        return arr.reshape(*original_shape)
    

class Boxes:
    BoxSizeType = Union[List[int], Tuple[int, int]]

    def __init__(self, tensor: torch.Tensor):
        device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu")
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            tensor = torch.zeros(0, 4, dtype=torch.float32, device=device)
        assert tensor.dim() == 2 and tensor.size(-1) == 4, tensor.size()

        self.tensor = tensor

    def clone(self) -> "Boxes":
        return Boxes(self.tensor.clone())

    def to(self, device: str) -> "Boxes":
        return Boxes(self.tensor.to(device))

    def area(self) -> torch.Tensor:
        box = self.tensor
        area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
        return area

    def clip(self, box_size: BoxSizeType) -> None:
        assert torch.isfinite(self.tensor).all()
        h, w = box_size
        self.tensor[:, 0].clamp_(min=0, max=w)
        self.tensor[:, 1].clamp_(min=0, max=h)
        self.tensor[:, 2].clamp_(min=0, max=w)
        self.tensor[:, 3].clamp_(min=0, max=h)

    def nonempty(self, threshold: int = 0) -> torch.Tensor:
        box = self.tensor
        widths = box[:, 2] - box[:, 0]
        heights = box[:, 3] - box[:, 1]
        keep = (widths > threshold) & (heights > threshold)
        return keep

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "Boxes":
        if isinstance(item, int):
            return Boxes(self.tensor[item].view(1, -1))
        b = self.tensor[item]
        assert b.dim() == 2, f"Indexing on Boxes with {item} failed to return a matrix!"
        return Boxes(b)

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def __repr__(self) -> str:
        return f"Boxes({str(self.tensor)})"

    def inside_box(self, box_size: BoxSizeType, boundary_threshold: int = 0) -> torch.Tensor:
        height, width = box_size
        inds_inside = (
            (self.tensor[..., 0] >= -boundary_threshold)
            & (self.tensor[..., 1] >= -boundary_threshold)
            & (self.tensor[..., 2] < width + boundary_threshold)
            & (self.tensor[..., 3] < height + boundary_threshold)
        )
        return inds_inside

    def get_centers(self) -> torch.Tensor:
        return (self.tensor[:, :2] + self.tensor[:, 2:]) / 2

    def scale(self, scale_x: float, scale_y: float) -> None:
        self.tensor[:, 0::2] *= scale_x
        self.tensor[:, 1::2] *= scale_y

    @staticmethod
    def cat(boxes_list: List["Boxes"]) -> "Boxes":
        assert isinstance(boxes_list, (list, tuple))
        assert len(boxes_list) > 0
        assert all(isinstance(box, Boxes) for box in boxes_list)

        cat_boxes = type(boxes_list[0])(torch.cat([b.tensor for b in boxes_list], dim=0))
        return cat_boxes

    @property
    def device(self) -> str:
        return self.tensor.device

    def __iter__(self) -> Iterator[torch.Tensor]:
        yield from self.tensor
