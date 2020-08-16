from time import perf_counter
from typing import Optional


class Timer:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._start = perf_counter()
        self._paused: Optional[float] = None
        self._total_paused = 0
        self._count_start = 1

    def pause(self) -> None:
        if self._paused is not None:
            raise ValueError("Trying to pause a Timer that is already paused!")
        self._paused = perf_counter()

    def is_paused(self) -> bool:
        return self._paused is not None

    def resume(self) -> None:
        if self._paused is None:
            raise ValueError("Trying to resume a Timer that is not paused!")
        self._total_paused += perf_counter() - self._paused  # pyre-ignore
        self._paused = None
        self._count_start += 1

    def seconds(self) -> float:
        if self._paused is not None:
            end_time: float = self._paused  # type: ignore
        else:
            end_time = perf_counter()
        return end_time - self._start - self._total_paused

    def avg_seconds(self) -> float:
        return self.seconds() / self._count_start
