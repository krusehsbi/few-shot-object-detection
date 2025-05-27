import logging
import copy
import itertools
from collections import OrderedDict
import torch
import numpy as np
from detectron2.data import MetadataCatalog
from fsdet.evaluation.evaluator import DatasetEvaluator
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from fsdet.utils.file_io import PathManager
import os
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.utils.logger import create_small_table
from detectron2.structures import BoxMode

class CocoTacoEvaluator(DatasetEvaluator):
    """
    Evaluator for your own fused dataset which might combine multiple datasets or
    class splits for evaluation.
    """

    def __init__(self, dataset_name, cfg, distributed, output_dir=None):
        """
        Args:
            dataset_name (str): Name of the fused dataset registered in MetadataCatalog.
            cfg: config node (optional, if you want to use cfg).
            distributed (bool): Whether to collect results from all ranks.
            output_dir (str): Optional directory to save results.
        """
        self._distributed = distributed
        self._output_dir = output_dir
        self._dataset_name = dataset_name

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        # Load dataset metadata and COCO API
        self._metadata = MetadataCatalog.get(dataset_name)
        if not hasattr(self._metadata, "json_file"):
            self._logger.warning(f"json_file was not found in MetaDataCatalog for '{dataset_name}'")
            cache_path = convert_to_coco_json(dataset_name, output_dir)
            self._metadata.json_file = cache_path

        json_file = PathManager.get_local_path(self._metadata.json_file)
        self._coco_api = COCO(json_file)

        # Whether annotations exist (e.g., test sets may not have annotations)
        self._do_evaluation = "annotations" in self._coco_api.dataset

        # Optionally define your fused dataset splits or special class sets here
        self._fused_classes = getattr(self._metadata, "fused_classes", None)
        self._split_classes = getattr(self._metadata, "split_classes", None)  # e.g. {"part1": [...], "part2": [...]}

        self.reset()

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        """
        Process the outputs from the model and convert to COCO json format.
        """
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}
            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = instances_to_coco_json(instances, input["image_id"])
            self._predictions.append(prediction)

    def evaluate(self):
        if self._distributed:
            import detectron2.utils.comm as comm
            comm.synchronize()
            self._predictions = comm.gather(self._predictions, dst=0)
            self._predictions = list(itertools.chain(*self._predictions))
            if not comm.is_main_process():
                return {}

        if len(self._predictions) == 0:
            self._logger.warning("[FusedDatasetEvaluator] No valid predictions received.")
            return {}

        if self._output_dir:
            os.makedirs(self._output_dir, exist_ok=True)  # Ensure directory exists
            file_path = os.path.join(self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(self._predictions, f)


        self._results = OrderedDict()
        if "instances" in self._predictions[0]:
            self._eval_predictions()

        return copy.deepcopy(self._results)

    def _eval_predictions(self):
        self._logger.info("Preparing fused dataset results for COCO format ...")
        self._coco_results = list(itertools.chain(*[x["instances"] for x in self._predictions]))

        # If your dataset uses contiguous IDs, unmap to original category ids
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            reverse_id_mapping = {v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()}
            for result in self._coco_results:
                result["category_id"] = reverse_id_mapping[result["category_id"]]

        if self._output_dir:
            os.makedirs(self._output_dir, exist_ok=True)
            file_path = os.path.join(self._output_dir, "fused_instances_results.json")
            self._logger.info(f"Saving fused results to {file_path}")
            with PathManager.open(file_path, "w") as f:
                import json
                f.write(json.dumps(self._coco_results))
                f.flush()


        if not self._do_evaluation:
            self._logger.info("Annotations not available for evaluation.")
            return

        self._logger.info("Evaluating fused dataset predictions ...")

        # Evaluate on entire fused dataset
        coco_eval = _evaluate_predictions_on_coco(self._coco_api, self._coco_results, "bbox") if self._coco_results else None
        self._results["bbox"] = self._derive_coco_results(coco_eval, "bbox", class_names=self._metadata.get("thing_classes"))

        # If your fused dataset has splits (like parts or categories), evaluate separately
        if self._split_classes:
            for split_name, class_ids in self._split_classes.items():
                self._logger.info(f"Evaluating split '{split_name}' with classes: {class_ids}")
                split_eval = _evaluate_predictions_on_coco(self._coco_api, self._coco_results, "bbox", catIds=class_ids)
                split_results = self._derive_coco_results(split_eval, "bbox", class_names=[self._metadata.get("thing_classes")[i] for i in class_ids])
                # Prefix split results with split name
                prefixed_results = {f"{split_name}_{k}": v for k, v in split_results.items()}
                self._results["bbox"].update(prefixed_results)

    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        """
        Same as COCOEvaluator._derive_coco_results — returns dictionary of metrics.
        """
        metrics = ["AP", "AP50", "AP75", "APs", "APm", "APl"]

        if coco_eval is None:
            self._logger.warning("No predictions from the model! Set scores to -1")
            return {metric: -1 for metric in metrics}

        results = {metric: float(coco_eval.stats[idx] * 100) for idx, metric in enumerate(metrics)}
        self._logger.info(f"Evaluation results for {iou_type}:\n" + create_small_table(results))

        if class_names is None or len(class_names) <= 1:
            return results

        precisions = coco_eval.eval["precision"]
        assert len(class_names) == precisions.shape[2]

        results_per_category = []
        for idx, name in enumerate(class_names):
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category.append((name, float(ap * 100)))

        from tabulate import tabulate
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(results_2d, tablefmt="pipe", floatfmt=".3f", headers=["category", "AP"] * (N_COLS // 2), numalign="left")
        self._logger.info(f"Per-category {iou_type} AP:\n" + table)

        results.update({f"AP-{name}": ap for name, ap in results_per_category})
        return results


def instances_to_coco_json(instances, img_id):
    """
    Helper to convert predictions to COCO json format.
    (Same as in your code.)
    """
    num_instance = len(instances)
    if num_instance == 0:
        return []

    boxes = instances.pred_boxes.tensor.numpy()
    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()

    results = []
    for k in range(num_instance):
        results.append({
            "image_id": img_id,
            "category_id": classes[k],
            "bbox": boxes[k].tolist(),
            "score": scores[k],
        })
    return results


def _evaluate_predictions_on_coco(coco_gt, coco_results, iou_type, catIds=None):
    """
    Evaluate the predictions using COCOeval.
    (Same as in your code.)
    """
    if len(coco_results) == 0:
        return None

    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    if catIds is not None:
        coco_eval.params.catIds = catIds
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval
