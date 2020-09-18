import numpy as np
from collections import OrderedDict, defaultdict

from vistem import dist
from vistem.utils.table import create_small_table

class AnnotationEvaluator:
    def _process_annotations(self, input, output):
        image_id = input["file_name"].split('/')[-2]
        annotations = output["annotations"].to(self._cpu_device)
        
        sorted_score = annotations.argsort(descending=True)
        gt = self._category.index(image_id)
        
        self._pred_annotations['top_1'].append(gt in sorted_score[:1])
        self._pred_annotations['top_5'].append(gt in sorted_score[:5])
        self._pred_annotations['top_10'].append(gt in sorted_score[:10])

    def _eval_annotations(self):
        if self._distributed:
            dist.synchronize()
            all_predictions = dist.gather(self._pred_annotations, dst=0)
            if not dist.is_main_process() : return {}

            predictions = defaultdict(list)
            for predictions_per_rank in all_predictions:
                for clsid, lines in predictions_per_rank.items():
                    predictions[clsid].extend(lines)
            del all_predictions

        else:
            predictions = self._pred_instances

        results = OrderedDict()

        results["annotations"] = dict()
        results["annotations"]["Top-1"] = np.mean(self._pred_annotations['top_1'])
        results["annotations"]["Top-5"] = np.mean(self._pred_annotations['top_5'])
        results["annotations"]["Top-10"] = np.mean(self._pred_annotations['top_10'])
        
        table = create_small_table(results['annotations'])
        self._logger.info(f"\n{table}")

        return results
