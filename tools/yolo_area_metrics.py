#!/usr/bin/env python3
"""Evaluate YOLO detections by normalized object-size bins."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


AREA_BINS = (
    ("tiny", 0.0, 0.0005),
    ("small", 0.0005, 0.0025),
    ("medium", 0.0025, 0.01),
    ("large", 0.01, float("inf")),
)


@dataclass(frozen=True)
class Detection:
    image_id: str
    cls: int
    confidence: float
    x: float
    y: float
    w: float
    h: float

    @property
    def area(self) -> float:
        return self.w * self.h


def bin_name(area: float) -> str:
    for name, lower, upper in AREA_BINS:
        if lower <= area < upper:
            return name
    return "large"


def xywh_to_xyxy(det: Detection) -> tuple[float, float, float, float]:
    return (
        det.x - det.w / 2.0,
        det.y - det.h / 2.0,
        det.x + det.w / 2.0,
        det.y + det.h / 2.0,
    )


def bbox_iou(a: Detection, b: Detection) -> float:
    ax1, ay1, ax2, ay2 = xywh_to_xyxy(a)
    bx1, by1, bx2, by2 = xywh_to_xyxy(b)
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    union = a.area + b.area - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def load_labels(labels_dir: Path) -> list[Detection]:
    labels: list[Detection] = []
    for path in sorted(labels_dir.glob("*.txt")):
        image_id = path.stem
        for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            row = raw.strip()
            if not row:
                continue
            parts = row.split()
            if len(parts) != 5:
                raise ValueError(f"Invalid YOLO label row in {path}: {raw}")
            cls = int(float(parts[0]))
            x, y, w, h = [float(value) for value in parts[1:]]
            labels.append(Detection(image_id=image_id, cls=cls, confidence=1.0, x=x, y=y, w=w, h=h))
    return labels


def load_predictions(predictions_path: Path) -> list[Detection]:
    payload = json.loads(predictions_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        rows = payload.get("predictions", [])
    else:
        rows = payload
    predictions: list[Detection] = []
    for row in rows:
        image_id = str(row.get("image_id") or row.get("image") or row.get("path", ""))
        if image_id.endswith(".jpg") or image_id.endswith(".jpeg") or image_id.endswith(".png"):
            image_id = Path(image_id).stem
        cls = int(row.get("class", row.get("cls", 0)))
        confidence = float(row.get("confidence", row.get("conf", 1.0)))
        bbox = row.get("bbox") or row.get("xywh")
        if not isinstance(bbox, list | tuple) or len(bbox) != 4:
            raise ValueError(f"Prediction row must include bbox/xywh with 4 values: {row}")
        x, y, w, h = [float(value) for value in bbox]
        predictions.append(Detection(image_id=image_id, cls=cls, confidence=confidence, x=x, y=y, w=w, h=h))
    return predictions


def average_precision(matches: list[tuple[float, bool]], ground_truth_count: int) -> float:
    if ground_truth_count == 0:
        return 0.0
    sorted_matches = sorted(matches, key=lambda item: item[0], reverse=True)
    true_positives = 0
    precision_sum = 0.0
    for rank, (_, matched) in enumerate(sorted_matches, start=1):
        if matched:
            true_positives += 1
            precision_sum += true_positives / rank
    return precision_sum / ground_truth_count


def match_predictions_by_bin(
    labels: list[Detection],
    predictions: list[Detection],
    iou_threshold: float,
) -> tuple[dict[str, list[tuple[float, bool]]], dict[str, int], int]:
    gt_by_bin: dict[str, list[Detection]] = {name: [] for name, _, _ in AREA_BINS}
    for label in labels:
        gt_by_bin[bin_name(label.area)].append(label)

    matched_ids: set[int] = set()
    pred_matches: dict[str, list[tuple[float, bool]]] = {name: [] for name, _, _ in AREA_BINS}

    for pred in sorted(predictions, key=lambda item: item.confidence, reverse=True):
        best_index = None
        best_iou = 0.0
        best_label = None
        for index, label in enumerate(labels):
            if index in matched_ids or label.image_id != pred.image_id or label.cls != pred.cls:
                continue
            iou = bbox_iou(pred, label)
            if iou > best_iou:
                best_iou = iou
                best_index = index
                best_label = label
        if best_label is None:
            pred_matches[bin_name(pred.area)].append((pred.confidence, False))
            continue
        target_bin = bin_name(best_label.area)
        matched = best_iou >= iou_threshold
        pred_matches[target_bin].append((pred.confidence, matched))
        if matched and best_index is not None:
            matched_ids.add(best_index)

    gt_counts = {name: len(gt_by_bin[name]) for name, _, _ in AREA_BINS}
    return pred_matches, gt_counts, len(matched_ids)


def evaluate_area_bins(
    labels: list[Detection],
    predictions: list[Detection],
    iou_threshold: float = 0.50,
) -> dict[str, Any]:
    matches50, gt_counts, total_tp50 = match_predictions_by_bin(labels, predictions, iou_threshold)
    matches75, _, total_tp75 = match_predictions_by_bin(labels, predictions, 0.75)

    bins: dict[str, Any] = {}
    metrics: dict[str, float] = {}
    for name, _, _ in AREA_BINS:
        gt_count = gt_counts[name]
        tp50 = sum(1 for _, matched in matches50[name] if matched)
        tp75 = sum(1 for _, matched in matches75[name] if matched)
        recall50 = round(tp50 / gt_count, 6) if gt_count else 0.0
        recall75 = round(tp75 / gt_count, 6) if gt_count else 0.0
        ap50 = round(average_precision(matches50[name], gt_count), 6)
        ap75 = round(average_precision(matches75[name], gt_count), 6)
        bins[name] = {
            "ground_truth": gt_count,
            "predictions": len(matches50[name]),
            "true_positives": tp50,
            "true_positives_75": tp75,
            "recall": recall50,
            "recall75": recall75,
            "AP": ap50,
            "AP50": ap50,
            "AP75": ap75,
        }
        metrics[f"AP_{name}"] = ap50
        metrics[f"AP75_{name}"] = ap75
        metrics[f"Recall_{name}"] = recall50
        metrics[f"Recall75_{name}"] = recall75

    return {
        "summary": {
            "ground_truth": len(labels),
            "predictions": len(predictions),
            "true_positives": total_tp50,
            "true_positives_75": total_tp75,
            "iou_threshold": iou_threshold,
        },
        "metrics": metrics,
        "bins": bins,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate YOLO predictions by tiny/small/medium/large object bins.")
    parser.add_argument("--labels", type=Path, required=True, help="Directory containing YOLO label txt files.")
    parser.add_argument("--predictions", type=Path, required=True, help="JSON list of predictions with image_id/class/confidence/bbox.")
    parser.add_argument("--out", type=Path, required=True, help="Output JSON report path.")
    parser.add_argument("--iou", type=float, default=0.50, help="IoU threshold for true positives.")
    args = parser.parse_args()

    report = evaluate_area_bins(load_labels(args.labels), load_predictions(args.predictions), iou_threshold=args.iou)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"area metrics saved: {args.out}")


if __name__ == "__main__":
    main()
