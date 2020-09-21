# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import os
import copy
from collections import OrderedDict, defaultdict

from vistem.utils import setup_logger

from vistem.loader import MetadataCatalog
from vistem.evaluation.default import Evaluator
from .annotations import AnnotationEvaluator

__all__ = ['ImageNetEvaluator']

class ImageNetEvaluator(Evaluator, AnnotationEvaluator):
    def __init__(self, cfg, distributed=True):
        self._distributed = distributed

        self._cpu_device = torch.device("cpu")
        self._logger = setup_logger(__name__)

        self._dataset_name = cfg.LOADER.TEST_DATASET
        self._metadata = MetadataCatalog.get(self._dataset_name)

        self._category = self._metadata.get("category_names")

    def reset(self):
        self._pred_annotations = {'top_1' : [], 'top_5' : [], 'top_10' : []}

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            if 'annotations' in output : self._process_annotations(input, output)
            

    def evaluate(self):
        self._logger.info(
            f"Evaluating {self._dataset_name}. Note that results do not use the official Matlab API."
        )

        self._results = OrderedDict()
        if len(self._pred_annotations) > 0 : self._results.update(self._eval_annotations())
        return copy.deepcopy(self._results)
