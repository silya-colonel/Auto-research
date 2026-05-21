#!/usr/bin/env python3
"""Select qualitative TPS-YOLO11 cases for paper figures and failure analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from yolo_area_metrics import Detection, bbox_iou, load_labels, load_predictions


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def detection_to_dict(det: Detection) -> dict[str, Any]:
    return {
        "image_id": det.image_id,
        "class": det.cls,
        "confidence": round(float(det.confidence), 6),
        "bbox": [round(det.x, 6), round(det.y, 6), round(det.w, 6), round(det.h, 6)],
    }


def find_best_match(
    target: Detection,
    predictions: list[Detection],
    iou_threshold: float,
    conf_threshold: float = 0.0,
    same_class: bool = True,
) -> tuple[Detection | None, float]:
    best_prediction = None
    best_iou = 0.0
    for prediction in predictions:
        if prediction.image_id != target.image_id or prediction.confidence < conf_threshold:
            continue
        if same_class and prediction.cls != target.cls:
            continue
        iou = bbox_iou(target, prediction)
        if iou > best_iou:
            best_prediction = prediction
            best_iou = iou
    if best_iou < iou_threshold:
        return None, best_iou
    return best_prediction, best_iou


def max_iou_with_labels(prediction: Detection, labels: list[Detection]) -> float:
    best = 0.0
    for label in labels:
        if label.image_id != prediction.image_id:
            continue
        best = max(best, bbox_iou(prediction, label))
    return best


def image_path_for(images_dir: Path | None, image_id: str) -> str | None:
    if images_dir is None:
        return None
    for extension in IMAGE_EXTENSIONS:
        path = images_dir / f"{image_id}{extension}"
        if path.exists():
            return str(path)
    matches = sorted(images_dir.glob(f"**/{image_id}.*"))
    for path in matches:
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            return str(path)
    return None


def attach_image_path(case: dict[str, Any], images_dir: Path | None) -> dict[str, Any]:
    image_path = image_path_for(images_dir, str(case["image_id"]))
    if image_path:
        case["image_path"] = image_path
    return case


def select_qualitative_cases(
    labels: list[Detection],
    baseline_predictions: list[Detection],
    candidate_predictions: list[Detection],
    iou_threshold: float = 0.50,
    fp_iou_threshold: float = 0.10,
    fp_conf_threshold: float = 0.25,
    limit_per_group: int = 50,
    images_dir: Path | None = None,
) -> dict[str, list[dict[str, Any]]]:
    improved: list[dict[str, Any]] = []
    remaining_misses: list[dict[str, Any]] = []
    suppressed_fps: list[dict[str, Any]] = []
    persistent_fps: list[dict[str, Any]] = []

    for label in labels:
        baseline_match, baseline_iou = find_best_match(label, baseline_predictions, iou_threshold)
        candidate_match, candidate_iou = find_best_match(label, candidate_predictions, iou_threshold)
        if baseline_match is None and candidate_match is not None:
            improved.append(
                attach_image_path(
                    {
                        "case_type": "improved_detection",
                        "image_id": label.image_id,
                        "class": label.cls,
                        "ground_truth": detection_to_dict(label),
                        "tps_prediction": detection_to_dict(candidate_match),
                        "baseline_best_iou": round(baseline_iou, 6),
                        "tps_iou": round(candidate_iou, 6),
                    },
                    images_dir,
                )
            )
        if candidate_match is None:
            remaining_misses.append(
                attach_image_path(
                    {
                        "case_type": "remaining_miss",
                        "image_id": label.image_id,
                        "class": label.cls,
                        "ground_truth": detection_to_dict(label),
                        "baseline_detected": baseline_match is not None,
                        "tps_best_iou": round(candidate_iou, 6),
                    },
                    images_dir,
                )
            )

    for prediction in sorted(baseline_predictions, key=lambda item: item.confidence, reverse=True):
        if prediction.confidence < fp_conf_threshold:
            continue
        max_gt_iou = max_iou_with_labels(prediction, labels)
        if max_gt_iou >= fp_iou_threshold:
            continue
        candidate_match, candidate_iou = find_best_match(
            prediction,
            candidate_predictions,
            iou_threshold,
            conf_threshold=fp_conf_threshold,
            same_class=False,
        )
        case = attach_image_path(
            {
                "image_id": prediction.image_id,
                "class": prediction.cls,
                "baseline_prediction": detection_to_dict(prediction),
                "baseline_max_gt_iou": round(max_gt_iou, 6),
                "tps_overlap_iou": round(candidate_iou, 6),
            },
            images_dir,
        )
        if candidate_match is None:
            case["case_type"] = "suppressed_false_positive"
            suppressed_fps.append(case)
        else:
            case["case_type"] = "persistent_false_positive"
            case["tps_prediction"] = detection_to_dict(candidate_match)
            persistent_fps.append(case)

    return {
        "improved_detections": improved[:limit_per_group],
        "suppressed_false_positives": suppressed_fps[:limit_per_group],
        "remaining_misses": remaining_misses[:limit_per_group],
        "persistent_false_positives": persistent_fps[:limit_per_group],
    }


def build_manifest(cases: dict[str, list[dict[str, Any]]], args: argparse.Namespace) -> dict[str, Any]:
    summary = {name: len(rows) for name, rows in cases.items()}
    return {
        "summary": summary,
        "settings": {
            "iou_threshold": args.iou,
            "fp_iou_threshold": args.fp_iou,
            "fp_conf_threshold": args.fp_conf,
            "limit_per_group": args.limit,
        },
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export qualitative TPS-YOLO11 improvement and failure cases.")
    parser.add_argument("--labels", type=Path, required=True, help="Directory containing YOLO validation labels.")
    parser.add_argument("--baseline", type=Path, required=True, help="Baseline prediction JSON.")
    parser.add_argument("--tps", type=Path, required=True, help="TPS/candidate prediction JSON.")
    parser.add_argument("--out", type=Path, required=True, help="Output case manifest JSON.")
    parser.add_argument("--images", type=Path, default=None, help="Optional image directory for visualization paths.")
    parser.add_argument("--iou", type=float, default=0.50, help="IoU threshold for successful detections.")
    parser.add_argument("--fp-iou", type=float, default=0.10, help="Maximum GT IoU for false-positive cases.")
    parser.add_argument("--fp-conf", type=float, default=0.25, help="Minimum baseline confidence for false positives.")
    parser.add_argument("--limit", type=int, default=50, help="Maximum rows kept per case group.")
    args = parser.parse_args()

    cases = select_qualitative_cases(
        labels=load_labels(args.labels),
        baseline_predictions=load_predictions(args.baseline),
        candidate_predictions=load_predictions(args.tps),
        iou_threshold=args.iou,
        fp_iou_threshold=args.fp_iou,
        fp_conf_threshold=args.fp_conf,
        limit_per_group=args.limit,
        images_dir=args.images,
    )
    manifest = build_manifest(cases, args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"qualitative cases saved: {args.out}")


if __name__ == "__main__":
    main()
