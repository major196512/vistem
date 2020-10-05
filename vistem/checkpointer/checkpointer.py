import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel

import os
import copy
import pickle
import numpy as np
from termcolor import colored
from collections import defaultdict
from typing import Any, Dict, List, Optional

from .missing import get_missing_parameters_message, get_unexpected_parameters_message
from .heuristic import align_and_update_state_dicts

from vistem.utils.logger import setup_logger

__all__ = ['Checkpointer']

class Checkpointer:
    def __init__(
        self,
        model: nn.Module,
        save_dir: str = "",
        *,
        save_to_disk: bool = True,
        **checkpointables: object,
    ) -> None:
        if isinstance(model, (DistributedDataParallel, DataParallel)):
            model = model.module

        self.model = model
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk
        self.checkpointables = copy.copy(checkpointables)

        self._logger = setup_logger(__name__)
        if not self.save_to_disk:
            self._logger.warning('No saving checkpoint mode')
        if not self.save_dir:
            self._logger.error('Not clarify saving directory')
            self.save_to_disk = False

    def _load_file(self, filename) -> object:
        if filename.endswith(".pkl"):
            with open(filename, "rb") as f:
                data = pickle.load(f, encoding="latin1")

            if "model" in data and "__author__" in data:
                # file is in Detectron2 model zoo format
                self._logger.info(f"Reading a file from '{data['__author__']}'")
                return data
            else:
                # assume file is from Caffe2 / Detectron1 model zoo
                data = {k: v for k, v in data.items() if not k.endswith("_momentum")}
                return {"model": data, "__author__": "Caffe2", "matching_heuristics": True}

        loaded = torch.load(filename, map_location=torch.device("cpu"))
        if '__author__' not in loaded:
            if filename.split('/')[-1].startswith('efficientnet'):
                loaded = {'model' : loaded, '__author__' : 'lukemelas', 'matching_heuristics' : True} # https://github.com/lukemelas/EfficientNet-PyTorch

        else : 
            assert loaded['__author__'] == 'vistem'

        return loaded

    def _load_model(self, checkpoint: Any):
        if checkpoint.get("matching_heuristics", False):
            self._convert_ndarray_to_tensor(checkpoint["model"])
            model_state_dict = self.model.state_dict()
            align_and_update_state_dicts(
                model_state_dict,
                checkpoint["model"],
                c2_conversion=checkpoint.get("__author__", None),
            )
            checkpoint["model"] = model_state_dict

        checkpoint_state_dict = checkpoint.pop("model")
        self._convert_ndarray_to_tensor(checkpoint_state_dict)
        self._strip_prefix_if_present(checkpoint_state_dict, "module.")

        model_state_dict = self.model.state_dict()
        incorrect_shapes = []
        for k in list(checkpoint_state_dict.keys()):
            if k in model_state_dict:
                shape_model = tuple(model_state_dict[k].shape)
                shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                if shape_model != shape_checkpoint:
                    incorrect_shapes.append((k, shape_checkpoint, shape_model))
                    checkpoint_state_dict.pop(k)
        
        incompatible = self.model.load_state_dict(checkpoint_state_dict, strict=False)

        return incompatible, incorrect_shapes

    def load(self, path: str, checkpointables: Optional[List[str]] = None) -> object:
        if not path:
            self._logger.warning("No checkpoint found. Initializing model from scratch")
            return {}
            
        self._logger.info("Loading checkpoint from {}".format(path))
        assert os.path.isfile(path), "Checkpoint {} not found!".format(path)

        checkpoint = self._load_file(path)
        incompatible, incorrect_shapes = self._load_model(checkpoint)
        self._log_incompatible_keys(incompatible, incorrect_shapes)
        for key in self.checkpointables if checkpointables is None else checkpointables:
            if key in checkpoint:
                self._logger.info("Loading {} from {}".format(key, path))
                obj = self.checkpointables[key]
                obj.load_state_dict(checkpoint.pop(key))

        return checkpoint

    def save(self, name: str, **kwargs: Dict[str, str]) -> None:
        data = {}
        data['__author__'] = 'vistem'
        data["model"] = self.model.state_dict()
        for key, obj in self.checkpointables.items():
            data[key] = obj.state_dict()
        data.update(kwargs)

        basename = "{}.pth".format(name)
        save_file = os.path.join(self.save_dir, basename)
        assert os.path.basename(save_file) == basename, basename

        self._logger.info("Saving checkpoint to {}".format(save_file))
        with open(save_file, "wb") as f:
            torch.save(data, f)

        self.tag_last_checkpoint(basename)

    def resume_or_load(self, path: str, *, resume: bool = True) -> object:
        if resume and self.has_checkpoint():
            path = self.get_checkpoint_file()
            return self.load(path)
        else:
            return self.load(path, checkpointables=[])


    def has_checkpoint(self) -> bool:
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        return os.path.isfile(save_file)

    def get_checkpoint_file(self) -> str:
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        try:
            with open(save_file, "r") as f:
                last_saved = f.read().strip()
        except IOError:
            return ""
        return os.path.join(self.save_dir, last_saved)

    def tag_last_checkpoint(self, last_filename_basename: str) -> None:
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        with open(save_file, "w") as f:
            f.write(last_filename_basename)


    def _log_incompatible_keys(self, incompatible, incorrect_shapes) -> None:
        for k, shape_checkpoint, shape_model in incorrect_shapes:
            self._logger.error(
                f"Unable to load '{k}' to the model due to incompatible shapes: "
                f"{shape_checkpoint} in the checkpoint "
                f"but {shape_model} in the model!"
            )

        if incompatible.missing_keys:
            msg = get_missing_parameters_message(incompatible.missing_keys)
            self._logger.info(msg)

        if incompatible.unexpected_keys:
            msg = get_unexpected_parameters_message(incompatible.unexpected_keys)
            self._logger.info(msg)

    def _convert_ndarray_to_tensor(self, state_dict: Dict[str, Any]) -> None:
        for k in list(state_dict.keys()):
            v = state_dict[k]
            if not isinstance(v, np.ndarray) and not isinstance(v, torch.Tensor):
                raise ValueError(f"Unsupported type found in checkpoint! {k}: {type(v)}")
            if not isinstance(v, torch.Tensor):
                state_dict[k] = torch.from_numpy(v)

    def _strip_prefix_if_present(self, state_dict: Dict[str, Any], prefix: str) -> None:
        keys = sorted(state_dict.keys())
        if not all(len(key) == 0 or key.startswith(prefix) for key in keys):
            return

        for key in keys:
            newkey = key[len(prefix) :]
            state_dict[newkey] = state_dict.pop(key)

        try:
            metadata = state_dict._metadata
        except AttributeError:
            pass
        else:
            for key in list(metadata.keys()):
                # for the metadata dict, the key can be:
                # '': for the DDP module, which we want to remove.
                # 'module': for the actual model.
                # 'module.xx.xx': for the rest.

                if len(key) == 0:
                    continue
                newkey = key[len(prefix) :]
                metadata[newkey] = metadata.pop(key)
