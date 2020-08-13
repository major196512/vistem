import types
import copy
from typing import List

from vistem.utils.logger import setup_logger

__all__ = ["DatasetCatalog", "MetadataCatalog"]


_logger = setup_logger()

class DatasetCatalog(object):
    _REGISTERED = {}

    @staticmethod
    def register(name, func):
        assert callable(func), "You must register a function with `DatasetCatalog.register`!"
        assert name not in DatasetCatalog._REGISTERED, f"Dataset '{name}' is already registered!"
        DatasetCatalog._REGISTERED[name] = func
        _logger.debug(f'DatasetCatalog({name})')

    @staticmethod
    def get(name):
        try:
            f = DatasetCatalog._REGISTERED[name]
        except KeyError:
            raise KeyError(
                f"Dataset '{name}' is not registered! Available datasets are: {', '.join(DatasetCatalog._REGISTERED.keys())}"
            )
        return f()

    @staticmethod
    def list() -> List[str]:
        return list(DatasetCatalog._REGISTERED.keys())

    @staticmethod
    def clear():
        DatasetCatalog._REGISTERED.clear()

class Metadata(types.SimpleNamespace):
    def __setattr__(self, key, val):
        if hasattr(self, key):
            oldval = getattr(self, key)
            if oldval != val :
                _logger.warn(f"Metadata '{key}' was updated!")

        super().__setattr__(key, val)

    def __hasattr__(self, key):
        return hasattr(self, key)

    def as_dict(self):
        return copy.copy(self.__dict__)

    def set(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

        return self

    def get(self, key, default=None):
        try:
            return getattr(self, key)
        except AttributeError:
            _logger.warn("Metadata '{key}' was not found!")
            return default


class MetadataCatalog:
    _NAME_TO_META = {}

    @staticmethod
    def get(name):
        assert len(name)
        if name in MetadataCatalog._NAME_TO_META:
            return MetadataCatalog._NAME_TO_META[name]
        else:
            m = MetadataCatalog._NAME_TO_META[name] = Metadata(name=name)
            return m
