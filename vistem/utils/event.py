import torch
from collections import defaultdict
from .history import HistoryBuffer

__all__ = [
    "EventStorage",
]

_CURRENT_STORAGE = []

class EventStorage:
    def __init__(self, start_iter=0):
        self._smoothing_hints = {}
        self._history = defaultdict(HistoryBuffer)
        self._vis_data = []
        self._histograms = []
        self._iter = start_iter

    def put_image(self, img_name, img_tensor):
        self._vis_data.append((img_name, img_tensor, self._iter))

    def put_scalar(self, name, value, smoothing_hint=True):
        history = self._history[name]
        value = float(value)
        history.update(value, self._iter)

        existing_hint = self._smoothing_hints.get(name)
        if existing_hint is not None:
            assert existing_hint == smoothing_hint, f"Scalar {name} was put with a different smoothing_hint!"
        else:
            self._smoothing_hints[name] = smoothing_hint

    def put_scalars(self, *, smoothing_hint=True, **kwargs):
        for k, v in kwargs.items():
            self.put_scalar(k, v, smoothing_hint=smoothing_hint)

    def put_histogram(self, hist_name, hist_tensor, bins=1000):
        ht_min, ht_max = hist_tensor.min().item(), hist_tensor.max().item()

        # Create a histogram with PyTorch
        hist_counts = torch.histc(hist_tensor, bins=bins)
        hist_edges = torch.linspace(start=ht_min, end=ht_max, steps=bins + 1, dtype=torch.float32)

        # Parameter for the add_histogram_raw function of SummaryWriter
        hist_params = dict(
            tag=hist_name,
            min=ht_min,
            max=ht_max,
            num=len(hist_tensor),
            sum=float(hist_tensor.sum()),
            sum_squares=float(torch.sum(hist_tensor ** 2)),
            bucket_limits=hist_edges[1:].tolist(),
            bucket_counts=hist_counts.tolist(),
            global_step=self._iter,
        )
        self._histograms.append(hist_params)

    def latest_with_smoothing(self, window_size=20):
        result = {}
        for k, h in self._history.items():
            if self._smoothing_hints[k] : result[k] = (h.median(window_size), iter)
            else : result[k] = (v, iter)
        return result

    def step(self):
        self._iter += 1

    @property
    def iter(self):
        return self._iter

    @property
    def iteration(self):
        # for backward compatibility
        return self._iter

    def history(self, name):
        ret = self._history.get(name, None)
        if ret is None:
            raise KeyError("No history metric available for {name}!")
        return ret

    def histories(self):
        return self._history
        
    def __enter__(self):
        _CURRENT_STORAGE.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert _CURRENT_STORAGE[-1] == self
        _CURRENT_STORAGE.pop()

