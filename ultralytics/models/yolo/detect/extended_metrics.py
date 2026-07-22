# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import LOGGER, RANK, ops
from ultralytics.utils.metrics import box_iou, compute_ap


AREA_RANGES = {
    "small": (0.0, 32.0**2),
    "tiny": (0.0, 16.0**2),
}


def _box_area(boxes: torch.Tensor) -> torch.Tensor:
    """Return areas for xyxy boxes."""
    wh = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0)
    return wh[:, 0] * wh[:, 1]


def match_predictions_with_area_ignore(
    pred_boxes: torch.Tensor,
    pred_conf: torch.Tensor,
    pred_cls: torch.Tensor,
    pred_area: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_cls: torch.Tensor,
    gt_area: torch.Tensor,
    iouv: torch.Tensor,
    area_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match detections while treating ground truths outside an area range as ignored.

    The matching order follows COCO evaluation semantics: valid ground truths are preferred over ignored ground
    truths, each non-crowd ground truth can be matched once per IoU threshold, and unmatched detections outside the
    requested area range are ignored instead of counted as false positives.
    """
    num_pred, num_iou = pred_boxes.shape[0], iouv.numel()
    tp = np.zeros((num_pred, num_iou), dtype=bool)
    ignored = np.zeros_like(tp)
    min_area, max_area = area_range
    pred_conf_np = pred_conf.detach().cpu().numpy()
    pred_cls_np = pred_cls.detach().cpu().numpy()
    pred_area_np = pred_area.detach().cpu().numpy()
    gt_cls_np = gt_cls.detach().cpu().numpy()
    gt_area_np = gt_area.detach().cpu().numpy()
    thresholds = iouv.detach().cpu().numpy()
    valid_gt = (gt_area_np >= min_area) & (gt_area_np < max_area)

    if num_pred == 0:
        return tp, ignored, gt_cls_np[valid_gt]

    pred_outside = (pred_area_np < min_area) | (pred_area_np >= max_area)
    if gt_boxes.shape[0] == 0:
        ignored[:] = pred_outside[:, None]
        return tp, ignored, gt_cls_np[valid_gt]

    ious = box_iou(gt_boxes, pred_boxes).detach().cpu().numpy()
    for class_id in np.union1d(pred_cls_np, gt_cls_np):
        pred_idx = np.where(pred_cls_np == class_id)[0]
        gt_idx = np.where(gt_cls_np == class_id)[0]
        if pred_idx.size == 0:
            continue
        pred_idx = pred_idx[np.argsort(-pred_conf_np[pred_idx])]
        if gt_idx.size == 0:
            ignored[pred_idx] = pred_outside[pred_idx, None]
            continue

        gt_ignore = ~valid_gt[gt_idx]
        gt_order = np.argsort(gt_ignore, kind="stable")
        gt_idx = gt_idx[gt_order]
        gt_ignore = gt_ignore[gt_order]

        for threshold_index, threshold in enumerate(thresholds):
            gt_matched = np.zeros(gt_idx.size, dtype=bool)
            for prediction_index in pred_idx:
                best_gt = -1
                best_iou = float(threshold)
                for ordered_gt_index, ground_truth_index in enumerate(gt_idx):
                    if gt_matched[ordered_gt_index]:
                        continue
                    if best_gt >= 0 and not gt_ignore[best_gt] and gt_ignore[ordered_gt_index]:
                        break
                    current_iou = ious[ground_truth_index, prediction_index]
                    if current_iou < best_iou:
                        continue
                    best_iou = current_iou
                    best_gt = ordered_gt_index

                if best_gt >= 0:
                    gt_matched[best_gt] = True
                    if gt_ignore[best_gt]:
                        ignored[prediction_index, threshold_index] = True
                    else:
                        tp[prediction_index, threshold_index] = True
                elif pred_outside[prediction_index]:
                    ignored[prediction_index, threshold_index] = True

    return tp, ignored, gt_cls_np[valid_gt]


def calculate_area_ap(
    tp: np.ndarray,
    ignored: np.ndarray,
    conf: np.ndarray,
    pred_cls: np.ndarray,
    target_cls: np.ndarray,
    num_classes: int,
) -> dict[str, Any]:
    """Calculate AP50, AP75, and AP50-95 with a per-IoU ignored-detection mask."""
    num_iou = tp.shape[1] if tp.ndim == 2 else 10
    ap = np.zeros((num_classes, num_iou), dtype=np.float64)
    target_counts = np.bincount(target_cls.astype(int), minlength=num_classes)

    for class_id in range(num_classes):
        class_targets = int(target_counts[class_id])
        if class_targets == 0:
            continue
        class_predictions = np.where(pred_cls == class_id)[0]
        if class_predictions.size == 0:
            continue
        class_predictions = class_predictions[np.argsort(-conf[class_predictions])]

        for threshold_index in range(num_iou):
            keep = ~ignored[class_predictions, threshold_index]
            prediction_indices = class_predictions[keep]
            if prediction_indices.size == 0:
                continue
            true_positive = tp[prediction_indices, threshold_index].astype(np.float64)
            false_positive = 1.0 - true_positive
            true_positive = np.cumsum(true_positive)
            false_positive = np.cumsum(false_positive)
            recall = true_positive / max(class_targets, 1)
            precision = true_positive / np.maximum(true_positive + false_positive, 1e-16)
            ap[class_id, threshold_index] = compute_ap(recall, precision)[0]

    evaluated_classes = target_counts > 0
    evaluated_ap = ap[evaluated_classes]
    if evaluated_ap.size == 0:
        map50 = map75 = map5095 = 0.0
    else:
        map50 = float(evaluated_ap[:, 0].mean())
        map75 = float(evaluated_ap[:, 5].mean())
        map5095 = float(evaluated_ap.mean())

    return {
        "AP50": map50,
        "AP75": map75,
        "AP50-95": map5095,
        "targets": int(target_counts.sum()),
        "targets_per_class": target_counts.tolist(),
        "AP50-95_per_class": ap.mean(axis=1).tolist(),
    }


class ExtendedDetectionValidator(DetectionValidator):
    """Detection validator that adds AP75, scale-specific AP, and unambiguous FPS metrics."""

    def init_metrics(self, model: torch.nn.Module) -> None:
        super().init_metrics(model)
        self.area_stats = {
            name: {"tp": [], "ignored": [], "conf": [], "pred_cls": [], "target_cls": []}
            for name in AREA_RANGES
        }
        self.extended_results = {}

    def update_metrics(self, preds: list[dict[str, torch.Tensor]], batch: dict[str, Any]) -> None:
        super().update_metrics(preds, batch)
        for sample_index, pred in enumerate(preds):
            prepared_batch = self._prepare_batch(sample_index, batch)
            prepared_pred = self._prepare_pred(pred)

            gt_boxes_original = ops.scale_boxes(
                prepared_batch["imgsz"],
                prepared_batch["bboxes"].clone(),
                prepared_batch["ori_shape"],
                ratio_pad=prepared_batch["ratio_pad"],
            )
            pred_original = self.scale_preds(prepared_pred, prepared_batch)
            gt_area = _box_area(gt_boxes_original)
            pred_area = _box_area(pred_original["bboxes"])

            for name, area_range in AREA_RANGES.items():
                tp, ignored, target_cls = match_predictions_with_area_ignore(
                    prepared_pred["bboxes"],
                    prepared_pred["conf"],
                    prepared_pred["cls"],
                    pred_area,
                    prepared_batch["bboxes"],
                    prepared_batch["cls"],
                    gt_area,
                    self.iouv,
                    area_range,
                )
                stats = self.area_stats[name]
                stats["tp"].append(tp)
                stats["ignored"].append(ignored)
                stats["conf"].append(prepared_pred["conf"].cpu().numpy())
                stats["pred_cls"].append(prepared_pred["cls"].cpu().numpy())
                stats["target_cls"].append(target_cls)

    def gather_stats(self) -> None:
        super().gather_stats()
        if RANK == 0:
            gathered = [None] * dist.get_world_size()
            dist.gather_object(self.area_stats, gathered, dst=0)
            merged = {
                name: {key: [] for key in values}
                for name, values in self.area_stats.items()
            }
            for rank_stats in gathered:
                for name, values in rank_stats.items():
                    for key, items in values.items():
                        merged[name][key].extend(items)
            self.area_stats = merged
        elif RANK > 0:
            dist.gather_object(self.area_stats, None, dst=0)
            for values in self.area_stats.values():
                for items in values.values():
                    items.clear()

    def get_stats(self) -> dict[str, Any]:
        stats = super().get_stats()
        self.extended_results = {"mAP75": float(self.metrics.box.map75)}
        for name, values in self.area_stats.items():
            concatenated = {
                key: np.concatenate(items, axis=0) if items else np.array([])
                for key, items in values.items()
            }
            if concatenated["tp"].size == 0:
                result = {
                    "AP50": 0.0,
                    "AP75": 0.0,
                    "AP50-95": 0.0,
                    "targets": 0,
                    "targets_per_class": [0] * self.nc,
                    "AP50-95_per_class": [0.0] * self.nc,
                }
            else:
                result = calculate_area_ap(
                    concatenated["tp"],
                    concatenated["ignored"],
                    concatenated["conf"],
                    concatenated["pred_cls"],
                    concatenated["target_cls"],
                    self.nc,
                )
            result["area_range"] = list(AREA_RANGES[name])
            result["area_coordinate_space"] = "original_image_pixels"
            self.extended_results[name] = result

        stats.update(
            {
                "metrics/mAP75(B)": self.extended_results["mAP75"],
                "metrics/AP_small(B)": self.extended_results["small"]["AP50-95"],
                "metrics/AP_tiny(B)": self.extended_results["tiny"]["AP50-95"],
            }
        )
        return stats

    def finalize_metrics(self) -> None:
        super().finalize_metrics()
        preprocess_ms = float(self.speed["preprocess"])
        inference_ms = float(self.speed["inference"])
        postprocess_ms = float(self.speed["postprocess"])
        end_to_end_ms = preprocess_ms + inference_ms + postprocess_ms
        self.extended_results["speed"] = {
            "preprocess_ms_per_image": preprocess_ms,
            "inference_ms_per_image": inference_ms,
            "postprocess_ms_per_image": postprocess_ms,
            "end_to_end_ms_per_image": end_to_end_ms,
            "inference_FPS": 1000.0 / inference_ms if inference_ms > 0 else 0.0,
            "end_to_end_FPS": 1000.0 / end_to_end_ms if end_to_end_ms > 0 else 0.0,
            "batch_size": int(self.args.batch),
            "image_size": self.args.imgsz,
        }
        self.metrics.extended_results = self.extended_results

        json_path = Path(self.save_dir) / "extended_metrics.json"
        csv_path = Path(self.save_dir) / "extended_metrics.csv"
        with open(json_path, "w", encoding="utf-8") as file:
            json.dump(self.extended_results, file, ensure_ascii=False, indent=2)

        rows = [
            {"metric": "mAP75", "value": self.extended_results["mAP75"]},
            {"metric": "AP_small", "value": self.extended_results["small"]["AP50-95"]},
            {"metric": "AP_tiny", "value": self.extended_results["tiny"]["AP50-95"]},
            {"metric": "inference_FPS", "value": self.extended_results["speed"]["inference_FPS"]},
            {"metric": "end_to_end_FPS", "value": self.extended_results["speed"]["end_to_end_FPS"]},
        ]
        with open(csv_path, "w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=("metric", "value"))
            writer.writeheader()
            writer.writerows(rows)

        LOGGER.info(
            "Extended metrics: mAP75=%.4f, AP_small=%.4f, AP_tiny=%.4f, inference FPS=%.2f, end-to-end FPS=%.2f",
            self.extended_results["mAP75"],
            self.extended_results["small"]["AP50-95"],
            self.extended_results["tiny"]["AP50-95"],
            self.extended_results["speed"]["inference_FPS"],
            self.extended_results["speed"]["end_to_end_FPS"],
        )
        LOGGER.info(f"Extended metrics saved to {json_path}")
