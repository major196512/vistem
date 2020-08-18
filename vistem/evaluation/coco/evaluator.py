import torch
import numpy as np
import io
import os
import json
import copy
import contextlib
import itertools
from tabulate import tabulate
from collections import OrderedDict

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from vistem import dist
from vistem.utils import setup_logger
from vistem.utils.table import create_small_table

from vistem.loader import MetadataCatalog
from vistem.evaluation.default import Evaluator

from .instances import COCOInstanceEvaluator

class COCOEvaluator(COCOInstanceEvaluator, Evaluator):
    def __init__(self, cfg, distributed=True):
        self._distributed = distributed
        self._output_dir = cfg.OUTPUT_DIR
        if self._output_dir and not os.path.isdir(self._output_dir) : os.makedirs(self._output_dir)

        self._cpu_device = torch.device("cpu")
        self._logger = setup_logger(__name__)

        dataset_name = cfg.LOADER.TEST_DATASET
        _metadata = MetadataCatalog.get(dataset_name)

        self._category = _metadata.get("category_names")
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCO(_metadata.json_file)

        super().__init__(cfg)

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}

            if "instances" in output:
                prediction["instances"] = self._instances_to_coco_json(input, output)
            if "proposals" in output:
                prediction["proposals"] = output["proposals"].to(self._cpu_device)
            self._predictions.append(prediction)

    def evaluate(self):
        if self._distributed:
            dist.synchronize()
            predictions = dist.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not dist.is_main_process() : return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.error("[COCOEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "instances_predictions.pth")
            with open(file_path, "wb") as f:
                torch.save(predictions, f)

        self._results = OrderedDict()
        if "instances" in predictions[0] : self._eval_instances(predictions)

        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)
