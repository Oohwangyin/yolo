"""Run a small Soft-NMS validation sweep for YOLO detection models.

This script keeps the Ultralytics source unchanged. It monkey-patches NMS only
inside this process, runs a short set of validation configurations, and writes a
CSV summary that is easy to compare.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".tmp_ultralytics_config"))

from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils import nms as nms_mod


ORIGINAL_NMS = nms_mod.non_max_suppression


@dataclass(frozen=True)
class Experiment:
    name: str
    nms_type: str
    conf: float
    iou: float
    sigma: float
    max_det: int
    candidate_limit: int


# ==============================================================================
# 修改点 1：根据论文实践，扩展实验矩阵。
# 引入了更低的 conf (0.001) 以确保软化后的密集小目标不被误杀，同时测试了不同的 sigma 和 linear 策略。
# ==============================================================================
EXPERIMENTS = [
    # 基线测试 (标准 NMS)
    Experiment("baseline_nms_standard", "nms", 0.001, 0.60, 0.50, 300, 3000),

    # 高斯衰减测试 (论文推荐：不同 sigma 严厉度测试)
    Experiment("soft_gaussian_sigma30", "gaussian", 0.001, 0.60, 0.30, 300, 3000),
    Experiment("soft_gaussian_sigma50", "gaussian", 0.001, 0.60, 0.50, 300, 3000),
    Experiment("soft_gaussian_sigma70", "gaussian", 0.001, 0.60, 0.70, 300, 3000),

    # 线性衰减测试
    Experiment("soft_linear_standard", "linear", 0.001, 0.60, 0.50, 300, 3000),
]


def box_soft_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_thres: float,
    sigma: float,
    score_thres: float,
    max_det: int,
    method: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return kept indices and decayed scores for thresholded Soft-NMS."""
    if boxes.numel() == 0:
        empty = torch.empty((0,), dtype=torch.long, device=boxes.device)
        return empty, scores

    sigma = max(float(sigma), 1e-6)
    scores_work = scores.clone()
    idxs = torch.arange(scores.shape[0], device=boxes.device)
    keep = []

    while idxs.numel() > 0 and len(keep) < max_det:
        max_pos = torch.argmax(scores_work[idxs])
        current = idxs[max_pos]
        if scores_work[current] < score_thres:
            break

        keep.append(current)
        idxs = idxs[idxs != current]
        if idxs.numel() == 0:
            break

        ious = nms_mod.box_iou(boxes[current].unsqueeze(0), boxes[idxs]).squeeze(0)
        overlap = ious > iou_thres

        if method == "linear":
            decay = torch.ones_like(ious)
            decay[overlap] = 1.0 - ious[overlap]
        elif method == "gaussian":
            decay = torch.ones_like(ious)
            decay[overlap] = torch.exp(-(ious[overlap] * ious[overlap]) / sigma)
        else:
            raise ValueError(f"Unsupported Soft-NMS method: {method}")

        scores_work[idxs] *= decay
        # 修改点 2：动态过滤。降低下一步循环的矩阵规模，优化密集目标下的计算开销
        idxs = idxs[scores_work[idxs] >= score_thres]

    if not keep:
        empty = torch.empty((0,), dtype=torch.long, device=boxes.device)
        return empty, scores_work
    return torch.stack(keep), scores_work


def build_standard_nms(candidate_limit: int):
    """Create a standard NMS wrapper with the same candidate cap as Soft-NMS."""

    def standard_non_max_suppression(
        prediction,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        classes=None,
        agnostic: bool = False,
        multi_label: bool = False,
        labels=(),
        max_det: int = 300,
        nc: int = 0,
        max_time_img: float = 0.05,
        max_nms: int = 30000,
        max_wh: int = 7680,
        rotated: bool = False,
        end2end: bool = False,
        return_idxs: bool = False,
    ):
        return ORIGINAL_NMS(
            prediction,
            conf_thres,
            iou_thres,
            classes,
            agnostic,
            multi_label,
            labels,
            max_det,
            nc,
            max_time_img,
            min(max_nms, candidate_limit),
            max_wh,
            rotated,
            end2end,
            return_idxs,
        )

    return standard_non_max_suppression


def build_soft_nms(method: str, sigma: float, candidate_limit: int):
    """Create a non_max_suppression replacement using Soft-NMS."""

    def soft_non_max_suppression(
        prediction,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        classes=None,
        agnostic: bool = False,
        multi_label: bool = False,
        labels=(),
        max_det: int = 300,
        nc: int = 0,
        max_time_img: float = 0.05,
        max_nms: int = 30000,
        max_wh: int = 7680,
        rotated: bool = False,
        end2end: bool = False,
        return_idxs: bool = False,
    ):
        """Soft-NMS replacement for standard detection validation."""
        if rotated or end2end or return_idxs:
            return ORIGINAL_NMS(
                prediction,
                conf_thres,
                iou_thres,
                classes,
                agnostic,
                multi_label,
                labels,
                max_det,
                nc,
                max_time_img,
                max_nms,
                max_wh,
                rotated,
                end2end,
                return_idxs,
            )

        assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}"
        assert 0 <= iou_thres <= 1, f"Invalid IoU threshold {iou_thres}"
        if isinstance(prediction, (list, tuple)):
            prediction = prediction[0]
        if classes is not None:
            classes = torch.tensor(classes, device=prediction.device)

        bs = prediction.shape[0]
        nc_ = nc or (prediction.shape[1] - 4)
        extra = prediction.shape[1] - nc_ - 4
        mi = 4 + nc_
        xc = prediction[:, 4:mi].amax(1) > conf_thres

        effective_max_nms = min(max_nms, candidate_limit)
        # 修改点 3：小目标场景候选框极多，适当放宽超时限制，防止验证被意外截断
        time_limit = 5.0 + max_time_img * bs
        multi_label &= nc_ > 1
        prediction = prediction.transpose(-1, -2)
        prediction[..., :4] = nms_mod.xywh2xyxy(prediction[..., :4])

        t = time.time()
        output = [torch.zeros((0, 6 + extra), device=prediction.device) for _ in range(bs)]
        for xi, x in enumerate(prediction):
            x = x[xc[xi]]
            if labels and len(labels[xi]):
                lb = labels[xi]
                v = torch.zeros((len(lb), nc_ + extra + 4), device=x.device)
                v[:, :4] = nms_mod.xywh2xyxy(lb[:, 1:5])
                v[range(len(lb)), lb[:, 0].long() + 4] = 1.0
                x = torch.cat((x, v), 0)

            if not x.shape[0]:
                continue

            box, cls, mask = x.split((4, nc_, extra), 1)
            if multi_label:
                i, j = torch.where(cls > conf_thres)
                x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
            else:
                conf, j = cls.max(1, keepdim=True)
                x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

            if classes is not None:
                x = x[(x[:, 5:6] == classes).any(1)]

            n = x.shape[0]
            if not n:
                continue
            if n > effective_max_nms:
                keep_top = x[:, 4].argsort(descending=True)[:effective_max_nms]
                x = x[keep_top]

            c = x[:, 5:6] * (0 if agnostic else max_wh)
            boxes = x[:, :4] + c
            scores = x[:, 4]
            keep, decayed_scores = box_soft_nms(
                boxes=boxes,
                scores=scores,
                iou_thres=iou_thres,
                sigma=sigma,
                score_thres=conf_thres,
                max_det=max_det,
                method=method,
            )
            if keep.numel():
                x = x[keep]
                x[:, 4] = decayed_scores[keep]
                output[xi] = x[:max_det]

            if (time.time() - t) > time_limit:
                LOGGER.warning(f"Soft-NMS time limit {time_limit:.3f}s exceeded")
                break

        return output

    return soft_non_max_suppression


def metrics_to_row(exp: Experiment, metrics, elapsed_s: float) -> dict[str, object]:
    """Convert Ultralytics metrics object to a flat CSV row."""
    box = getattr(metrics, "box", None)
    speed = getattr(metrics, "speed", {}) or {}
    results_dict = getattr(metrics, "results_dict", {}) or {}

    return {
        "name": exp.name,
        "nms_type": exp.nms_type,
        "conf": exp.conf,
        "iou": exp.iou,
        "sigma": exp.sigma,
        "max_det": exp.max_det,
        "candidate_limit": exp.candidate_limit,
        "precision": getattr(box, "mp", results_dict.get("metrics/precision(B)", "")),
        "recall": getattr(box, "mr", results_dict.get("metrics/recall(B)", "")),
        "map50": getattr(box, "map50", results_dict.get("metrics/mAP50(B)", "")),
        "map50_95": getattr(box, "map", results_dict.get("metrics/mAP50-95(B)", "")),
        "fitness": getattr(metrics, "fitness", results_dict.get("fitness", "")),
        "preprocess_ms": speed.get("preprocess", ""),
        "inference_ms": speed.get("inference", ""),
        "postprocess_ms": speed.get("postprocess", ""),
        "elapsed_s": round(elapsed_s, 2),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="VisDrone/yolov8s/train/weights/best.pt")
    parser.add_argument("--data", default="VisDrone.yaml")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", default="runs/soft_nms_experiments")
    parser.add_argument("--out", default="runs/soft_nms_experiments/summary.csv")
    parser.add_argument("--dry-run", action="store_true", help="Print experiment matrix without running validation.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("Soft-NMS experiment matrix:")
    for exp in EXPERIMENTS:
        print(f"  - {exp}")
    if args.dry_run:
        return 0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for exp in EXPERIMENTS:
        print(f"\n=== Running {exp.name} ===")
        if exp.nms_type == "nms":
            nms_mod.non_max_suppression = build_standard_nms(exp.candidate_limit)
        else:
            nms_mod.non_max_suppression = build_soft_nms(exp.nms_type, exp.sigma, exp.candidate_limit)
        model = YOLO(args.weights)
        start = time.time()
        metrics = model.val(
            data=args.data,
            split=args.split,
            batch=args.batch,
            imgsz=args.imgsz,
            conf=exp.conf,
            iou=exp.iou,
            max_det=exp.max_det,
            device=args.device,
            project=args.project,
            name=exp.name,
            plots=False,
            verbose=False,
        )
        rows.append(metrics_to_row(exp, metrics, time.time() - start))

    nms_mod.non_max_suppression = ORIGINAL_NMS
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved summary: {out_path}")
    print("Decision rule: Check if 'recall' and 'map50' improve significantly under 'soft_gaussian' settings, and assess the latency via 'postprocess_ms'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())