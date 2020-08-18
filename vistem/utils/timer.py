from time import perf_counter
from typing import Optional


class Timer:
    def __init__(self, warmup=0, pause=False) -> None:
        self.reset(warmup, pause)

    def reset(self, warmup=0, pause=False) -> None:
        self._start = perf_counter()
        self._warmup = warmup
        self._paused: Optional[float] = None

        self._total_paused = 0
        self._total_warmup = 0
        self._count_start = 0

        if pause : self._paused = perf_counter()
        else : self._count_start += 1

    def pause(self) -> None:
        if self._paused is not None:
            raise ValueError("Trying to pause a Timer that is already paused!")

        if self._warmup > 0: 
            self._total_warmup = perf_counter() - self._start - self._total_paused
            self._warmup -= 1
            self._count_start -= 1

        self._paused = perf_counter()

    def is_paused(self) -> bool:
        return self._paused is not None

    def resume(self) -> None:
        if self._paused is None:
            raise ValueError("Trying to resume a Timer that is not paused!")

        self._total_paused += (perf_counter() - self._paused)
        self._paused = None
        self._count_start += 1

    def seconds(self) -> float:
        if self._paused is not None:
            end_time: float = self._paused  # type: ignore
        else:
            end_time = perf_counter()
        return end_time - self._start - self._total_paused - self._total_warmup

    def avg_seconds(self) -> float:
        return self.seconds() / self._count_start

    def total_seconds(self) -> float:
        return perf_counter() - self._start - self._total_warmup
