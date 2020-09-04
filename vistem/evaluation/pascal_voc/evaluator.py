# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import os
import copy
from collections import OrderedDict, defaultdict

from vistem.utils import setup_logger

from vistem.loader import MetadataCatalog
from vistem.evaluation.default import Evaluator
from .instances import VOCInstanceEvaluator
from .proposals import VOCProposalEvaluator

class VOCEvaluator(Evaluator, VOCInstanceEvaluator, VOCProposalEvaluator):
    """
    Evaluate Pascal VOC style AP for Pascal VOC dataset.
    It contains a synchronization, therefore has to be called from all ranks.
    Note that the concept of AP can be implemented in different ways and may not
    produce identical results. This class mimics the implementation of the official
    Pascal VOC Matlab API, and should produce similar but not identical results to the
    official API.
    """
    def __init__(self, cfg, distributed=True):
        self._distributed = distributed

        self._cpu_device = torch.device("cpu")
        self._logger = setup_logger(__name__)

        self._dataset_name = cfg.LOADER.TEST_DATASET
        self._metadata = MetadataCatalog.get(self._dataset_name)

        self._category = self._metadata.get("category_names")

        data_root = self._metadata.get('data_root')
        self._anno_file_template = os.path.join(data_root, "Annotations", "{}.xml")
        self._image_set_path = os.path.join(data_root, "ImageSets", "Main", self._metadata.get('split') + ".txt")

        year = self._metadata.get('year')
        assert year in [2007, 2012], year
        self._is_2007 = year == 2007

        super().__init__(cfg)

    def reset(self):
        self._pred_instances = defaultdict(list)
        self._pred_proposals = list()

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            if 'proposals' in output : self._process_proposals(input, output)
            if 'instances' in output : self._process_instances(input, output)
            

    def evaluate(self):
        self._logger.info(
            "Evaluating {} using {} metric. "
            "Note that results do not use the official Matlab API.".format(
                self._dataset_name, 2007 if self._is_2007 else 2012
            )
        )

        self._results = OrderedDict()
        if len(self._pred_instances) > 0 : self._results.update(self._eval_instances())
        if len(self._pred_proposals) > 0 : self._results.update(self._eval_proposals())
        return copy.deepcopy(self._results)
