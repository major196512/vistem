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

from .instances import instances_to_coco_json

class COCOEvaluator(Evaluator):
    def __init__(self, cfg, distributed=True):
        self._tasks = self._tasks_from_config(cfg)
        self._distributed = distributed
        self._output_dir = cfg.OUTPUT_DIR

        self._cpu_device = torch.device("cpu")
        self._logger = setup_logger(__name__)

        dataset_name = cfg.LOADER.TEST_DATASET
        self._metadata = MetadataCatalog.get(dataset_name)

        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCO(self._metadata.json_file)

        self._do_evaluation = "annotations" in self._coco_api.dataset

    def _tasks_from_config(self, cfg):
        tasks = ("bbox",)
        return tasks

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}

            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = instances_to_coco_json(instances, input["image_id"])
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
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            if not  os.path.isdir(self._output_dir) : os.makedirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "instances_predictions.pth")
            with open(file_path, "wb") as f:
                torch.save(predictions, f)

        self._results = OrderedDict()
        self._eval_predictions(predictions)

        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _eval_predictions(self, predictions):
        self._logger.info("Preparing results for COCO format ...")
        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))

        # unmap the category ids for COCO
        if hasattr(self._metadata, "dataset_id_to_contiguous_id"):
            reverse_id_mapping = {
                v: k for k, v in self._metadata.dataset_id_to_contiguous_id.items()
            }
            for result in coco_results:
                category_id = result["category_id"]
                assert (
                    category_id in reverse_id_mapping
                ), "A prediction has category_id={}, which is not available in the dataset.".format(
                    category_id
                )
                result["category_id"] = reverse_id_mapping[category_id]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "coco_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with open(file_path, "w") as f:
                f.write(json.dumps(coco_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info("Evaluating predictions with COCO API...")
        for task in sorted(self._tasks):
            coco_eval = (
                _evaluate_predictions_on_coco(
                    self._coco_api,
                    coco_results,
                    task,
                )
                if len(coco_results) > 0
                else None  # cocoapi does not handle empty results very well
            )

            res = self._derive_coco_results(
                coco_eval, task, class_names=self._metadata.get("category_names")
            )
            self._results[task] = res

    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        metrics = {
            "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
        }[iou_type]

        if coco_eval is None:
            self._logger.warn("No predictions from the model!")
            return {metric: float("nan") for metric in metrics}

        # the standard metrics
        results = {
            metric: float(coco_eval.stats[idx] * 100 if coco_eval.stats[idx] >= 0 else "nan")
            for idx, metric in enumerate(metrics)
        }
        self._logger.info(
            "Evaluation results for {}: \n".format(iou_type) + create_small_table(results)
        )
        if not np.isfinite(sum(results.values())):
            self._logger.info("Some metrics cannot be computed and is shown as NaN.")

        if class_names is None or len(class_names) <= 1:
            return results
            
        precisions = coco_eval.eval["precision"]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]

        results_per_category = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category.append(("{}".format(name), float(ap * 100)))

        # tabulate it
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category {} AP: \n".format(iou_type) + table)

        results.update({"AP-" + name: ap for name, ap in results_per_category})
        return results

def _evaluate_predictions_on_coco(coco_gt, coco_results, iou_type):
    assert len(coco_results) > 0

    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval
