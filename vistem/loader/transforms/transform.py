import inspect
import numpy as np
from abc import ABCMeta, abstractmethod
from typing import Any, List, Optional
import pprint

__all__ = ['Transform', 'TransformGen', 'NoOpTransform', 'TransformList']
class Transform(metaclass=ABCMeta):
    def _set_attributes(self, params: Optional[List[Any]] = None) -> None:
        if params:
            for k, v in params.items():
                if k != "self" and not k.startswith("_"):
                    setattr(self, k, v)

    @abstractmethod
    def apply_image(self, img: np.ndarray):
        pass

    @abstractmethod
    def apply_coords(self, coords: np.ndarray):
        pass

    def apply_box(self, box: np.ndarray) -> np.ndarray:
        # Indexes of converting (x0, y0, x1, y1) box into 4 coordinates of
        # ([x0, y0], [x1, y0], [x0, y1], [x1, y1]).
        idxs = np.array([(0, 1), (2, 1), (0, 3), (2, 3)]).flatten()
        coords = np.asarray(box).reshape(-1, 4)[:, idxs].reshape(-1, 2)
        coords = self.apply_coords(coords).reshape((-1, 4, 2))
        minxy = coords.min(axis=1)
        maxxy = coords.max(axis=1)
        trans_boxes = np.concatenate((minxy, maxxy), axis=1)
        return trans_boxes

class TransformGen(metaclass=ABCMeta):
    def _init(self, params=None):
        if params:
            for k, v in params.items():
                if k != "self" and not k.startswith("_"):
                    setattr(self, k, v)

    @abstractmethod
    def init_local(self):
        pass

    @abstractmethod
    def get_transform(self, img):
        pass

    def _rand_range(self, low=1.0, high=None, size=None):
        if high is None:
            low, high = 0, low
        if size is None:
            size = []
        return np.random.uniform(low, high, size)

    def __repr__(self):
        try:
            sig = inspect.signature(self.init_local)
            classname = type(self).__name__
            argstr = []
            for name, param in sig.parameters.items():
                assert (
                    param.kind != param.VAR_POSITIONAL and param.kind != param.VAR_KEYWORD
                ), "The default __repr__ doesn't support *args or **kwargs"
                assert hasattr(self, name), f"Attribute {name} not found! "
                
                attr = getattr(self, name)
                default = param.default
                if default == attr : continue

                argstr.append(f"{name}={pprint.pformat(attr)}")

            return f"{classname}({', '.join(argstr)})"

        except AssertionError:
            return super().__repr__()

    __str__ = __repr__

class NoOpTransform(Transform):
    def __init__(self):
        super().__init__()

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        return img

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords

    def __getattr__(self, name: str):
        if name.startswith("apply_"):
            return lambda x: x
        raise AttributeError(f"NoOpTransform object has no attribute {name}")
    
class TransformList:
    def __init__(self, transforms: list):
        super().__init__()
        for t in transforms:
            assert isinstance(t, Transform), t
        self.transforms = transforms

    def _apply(self, x, meth: str):
        for t in self.transforms:
            x = getattr(t, meth)(x)
        return x

    def __getattribute__(self, name: str):
        # use __getattribute__ to win priority over any registered dtypes
        if name.startswith("apply_"):
            return lambda x: self._apply(x, name)
        return super().__getattribute__(name)

    def __add__(self, other: "TransformList") -> "TransformList":
        others = other.transforms if isinstance(other, TransformList) else [other]
        return TransformList(self.transforms + others)

    def __len__(self) -> int:
        return len(self.transforms)
