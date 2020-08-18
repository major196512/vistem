import numpy as np
import pycocotools.mask as mask_util

from vistem.structures import BoxMode

from vistem.evaluation.default import Evaluator



class COCOInstanceEvaluator(Evaluator):
    def __init__(self, cfg):
        self._tasks = self._tasks_from_config(cfg)

        self.VISTEM_BOX_MODE = BoxMode.XYXY_ABS
        self.COCO_BOX_MODE = BoxMode.XYWH_ABS

    def _tasks_from_config(self, cfg):
        tasks = ("bbox",)
        return tasks

    def _instances_to_coco_json(self, input, output):
        img_id = input["image_id"]
        instances = output["instances"].to(self._cpu_device)

        num_instance = len(instances)
        if num_instance == 0 : return []

        results = []
        scores = instances.scores.tolist()
        classes = instances.pred_classes.tolist()
        for k in range(num_instance):
            result = {
                "image_id": img_id,
                "category_id": classes[k],
                "score": scores[k],
            }
            results.append(result)

        has_box = instances.has('pred_boxes')
        has_mask = instances.has("pred_masks")
        has_keypoint = instances.has("pred_keypoints")

        # BBox Prediction
        if has_box:
            boxes = instances.pred_boxes.tensor.numpy()
            boxes = BoxMode.convert(boxes, self.VISTEM_BOX_MODE, self.COCO_BOX_MODE)
            boxes = boxes.tolist()
            for k in range(num_instance):
                results[k]['bbox'] = boxes[k]

        # Segmentation Prediction
        if has_mask:
            rles = [
                mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
                for mask in instances.pred_masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")
            for k in range(num_instance):
                results[k]["segmentation"] = rles[k]

        # Keypoint Prediction
        if has_keypoint:
            keypoints = instances.pred_keypoints
            keypoints[:][:, :2] -= 0.5
            for k in range(num_instance):
                results[k]["keypoints"] = keypoints[k].flatten().tolist()

        return results

    def _eval_instances(self, predictions):
        self._logger.info("Preparing results for COCO format ...")
        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))
        if len(coco_results) == 0:
            self._logger.warn("No predictions from the model!")
            return

        if hasattr(self._metadata, "dataset_id_to_contiguous_id"):
            reverse_id_mapping = {
                v: k for k, v in self._metadata.dataset_id_to_contiguous_id.items()
            }
            for result in coco_results:
                category_id = result["category_id"]
                assert category_id in reverse_id_mapping, f"A prediction has category_id={category_id}, which is not available in the dataset."
                result["category_id"] = reverse_id_mapping[category_id]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "coco_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with open(file_path, "w") as f:
                f.write(json.dumps(coco_results))
                f.flush()

        if "annotations" not in self._coco_api.dataset:
            self._logger.error("Annotations are not available for evaluation.")
            return

        self._logger.info("Evaluating predictions with COCO API...")
        for task in sorted(self._tasks):
            coco_gt = self._coco_api
            coco_dt = coco_gt.loadRes(coco_results)
            coco_eval = COCOeval(coco_gt, coco_dt, task)

            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            res = self._print_coco_results(coco_eval, task, class_names=self._category)
            self._results[task] = res

    def _print_coco_results(self, coco_eval, iou_type, class_names=None):
        metrics = {
            "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
        }[iou_type]

        # precision has dims (iou, recall, cls, area range, max dets)    
        precisions = coco_eval.eval["precision"]
        assert len(class_names) == precisions.shape[2]

        # the standard metrics
        results = {
            metric: float(coco_eval.stats[idx] * 100 if coco_eval.stats[idx] >= 0 else "nan")
            for idx, metric in enumerate(metrics)
        }
        if not np.isfinite(sum(results.values())):
            self._logger.info("Some metrics cannot be computed and is shown as NaN.")
        table = create_small_table(results)
        self._logger.info(f"Evaluation results for {iou_type}: \n{table}")

        if class_names is None or len(class_names) <= 1:
            return results

        # metrics per category
        category_results = []
        for idx, name in enumerate(class_names):
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            category_results.append((name, float(ap * 100)))

        N_COLS = min(3, len(class_names))
        results_flatten = list(itertools.chain(*category_results))
        results_flatten = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])

        table = create_multi_column_table(results_flatten, 3, headers=["category", "AP"], align='left')
        self._logger.info(f"Per-category {iou_type} AP: \ntable")
        
        results.update({f"AP-{name}": ap for name, ap in cagegory_results})
        return results
