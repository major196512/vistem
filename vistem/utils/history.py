# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import List, Optional, Tuple

import numpy as np


class HistoryBuffer:
    def __init__(self, max_length: int = 1000000) -> None:
        self._max_length: int = max_length
        self._data: List[Tuple[float, float]] = []  # (value, iteration) pairs
        self._count: int = 0
        self._global_avg: float = 0

    def update(self, value: float, iteration: Optional[float] = None) -> None:
        if iteration is None:
            iteration = self._count
        if len(self._data) == self._max_length:
            self._data.pop(0)
        self._data.append((value, iteration))

        self._count += 1
        self._global_avg += (value - self._global_avg) / self._count

    def latest(self) -> float:
        return self._data[-1][0]

    def median(self, window_size: int) -> float:
        return np.median([x[0] for x in self._data[-window_size:]])

    def avg(self, window_size: int) -> float:
        return np.mean([x[0] for x in self._data[-window_size:]])

    def global_avg(self) -> float:
        return self._global_avg

    def values(self) -> List[Tuple[float, float]]:
        return self._data