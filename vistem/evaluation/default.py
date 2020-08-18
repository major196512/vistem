

from vistem import dist
from vistem.utils import setup_logger

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import datetime
import logging
import time
from collections import OrderedDict

import torch


class Evaluator:
    def reset(self):
        pass

    def process(self, inputs, outputs):
        pass

    def evaluate(self):
        pass


class Evaluators:
    def __init__(self, evaluators):
        self._logger = setup_logger(__name__)
        self._evaluators = evaluators
        for e in self._evaluators:
            assert isinstance(e, Evaluator), f'{e} is not Evaluator!'

    def reset(self):
        for evaluator in self._evaluators:
            evaluator.reset()

    def process(self, inputs, outputs):
        for evaluator in self._evaluators:
            evaluator.process(inputs, outputs)

    def evaluate(self):
        results = OrderedDict()
        for evaluator in self._evaluators:
            result = evaluator.evaluate()
            if dist.is_main_process() and result is not None:
                for k, v in result.items():
                    assert (
                        k not in results
                    ), "Different evaluators produce results with the same key {}".format(k)
                    results[k] = v
        return results
