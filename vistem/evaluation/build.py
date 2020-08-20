from vistem.utils import setup_logger
from vistem.loader import MetadataCatalog

from .default import Evaluators
from .coco import COCOEvaluator
from .pascal_voc import VOCEvaluator

__all__ = ['build_evaluator']

def build_evaluator(cfg):
    _logger = setup_logger(__name__)
    test_dataset = cfg.LOADER.TEST_DATASET
    eval_type = MetadataCatalog.get(test_dataset).evaluator_type

    evaluators = []
    if eval_type == 'coco':
        evaluators.append(COCOEvaluator(cfg))
    elif eval_type == 'voc':
        evaluators.append(VOCEvaluator(cfg))
    else:
        _logger.error(f'There is no Evaluation Type : {test_dataset}')

    return Evaluators(evaluators)
