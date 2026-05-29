#!/usr/bin/env python3
"""Run claim-focused TPS-YOLO evaluations for tiny defects and hard backgrounds."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.yolo_area_metrics import Detection, average_precision, bbox_iou, evaluate_area_bins, load_labels, load_predictions
from tools.yolo_hard_background import evaluate_hard_background, mine_hard_background, write_candidates


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def resolve_data_entry(data_yaml: Path, key: str) -> Path:
    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    base = Path(payload.get("path") or data_yaml.parent)
    if not base.is_absolute():
        base = (data_yaml.parent / base).resolve()
    value = payload[key]
    if isinstance(value, list):
        value = value[0]
    path = Path(value)
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


def image_paths(images_dir: Path, max_images: int | None = None) -> list[Path]:
    paths = sorted(path for path in images_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    if max_images is not None:
        return paths[:max_images]
    return paths


def write_predictions(
    weights: Path,
    images: list[Path],
    out: Path,
    imgsz: int,
    conf: float,
    device: str | None,
    max_det: int,
    batch_size: int,
) -> None:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    rows: list[dict[str, Any]] = []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        results = model.predict(
            source=[str(path) for path in batch],
            imgsz=imgsz,
            conf=conf,
            device=device,
            max_det=max_det,
            verbose=False,
            stream=True,
        )
        for image_path, result in zip(batch, results, strict=False):
            image_id = image_path.stem
            boxes = result.boxes
            if boxes is None:
                continue
            xywhn = boxes.xywhn.cpu().tolist()
            classes = boxes.cls.cpu().tolist()
            confidences = boxes.conf.cpu().tolist()
            for cls, score, bbox in zip(classes, confidences, xywhn, strict=False):
                rows.append(
                    {
                        "image_id": image_id,
                        "class": int(cls),
                        "confidence": float(score),
                        "bbox": [float(value) for value in bbox],
                    }
            )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"predictions saved: {out} ({len(rows)} boxes)")


def match_predictions(labels: list[Detection], predictions: list[Detection], iou_threshold: float) -> tuple[list[tuple[Detection, bool]], int]:
    matched_ids: set[int] = set()
    matches: list[tuple[Detection, bool]] = []
    for pred in sorted(predictions, key=lambda item: item.confidence, reverse=True):
        best_index = None
        best_iou = 0.0
        for index, label in enumerate(labels):
            if index in matched_ids or label.image_id != pred.image_id or label.cls != pred.cls:
                continue
            iou = bbox_iou(pred, label)
            if iou > best_iou:
                best_iou = iou
                best_index = index
        matched = best_index is not None and best_iou >= iou_threshold
        matches.append((pred, matched))
        if matched and best_index is not None:
            matched_ids.add(best_index)
    return matches, len(matched_ids)


def evaluate_per_class(labels: list[Detection], predictions: list[Detection], rare_threshold: int) -> dict[str, Any]:
    gt_counts = Counter(label.cls for label in labels)
    pred_by_class: dict[int, list[Detection]] = defaultdict(list)
    label_by_class: dict[int, list[Detection]] = defaultdict(list)
    for label in labels:
        label_by_class[label.cls].append(label)
    for pred in predictions:
        pred_by_class[pred.cls].append(pred)

    classes: dict[str, Any] = {}
    for cls in sorted(set(gt_counts) | set(pred_by_class)):
        class_labels = label_by_class.get(cls, [])
        class_predictions = pred_by_class.get(cls, [])
        matches50, tp50 = match_predictions(class_labels, class_predictions, 0.50)
        matches75, tp75 = match_predictions(class_labels, class_predictions, 0.75)
        gt_count = len(class_labels)
        pred_count = len(class_predictions)
        classes[str(cls)] = {
            "ground_truth": gt_count,
            "predictions": pred_count,
            "AP50": round(average_precision([(pred.confidence, ok) for pred, ok in matches50], gt_count), 6),
            "AP75": round(average_precision([(pred.confidence, ok) for pred, ok in matches75], gt_count), 6),
            "recall50": round(tp50 / gt_count, 6) if gt_count else 0.0,
            "recall75": round(tp75 / gt_count, 6) if gt_count else 0.0,
            "precision50": round(tp50 / pred_count, 6) if pred_count else 0.0,
            "rare": gt_count <= rare_threshold,
        }
    rare = {cls: row for cls, row in classes.items() if row["rare"]}
    return {"rare_threshold": rare_threshold, "classes": classes, "rare_classes": rare}


def write_summary(out: Path, area: dict[str, Any], hard: dict[str, Any], rare: dict[str, Any]) -> None:
    lines = [
        "# TPS Claim Evaluation Summary",
        "",
        "## Tiny and Small Objects",
        "",
        "| bin | GT | predictions | AP50 | AP75 | Recall50 | Recall75 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ("tiny", "small", "medium", "large"):
        row = area["bins"][name]
        lines.append(
            f"| {name} | {row['ground_truth']} | {row['predictions']} | {row['AP50']:.4f} | "
            f"{row['AP75']:.4f} | {row['recall']:.4f} | {row['recall75']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Hard Background",
            "",
            f"- candidate_regions: {hard['candidate_regions']}",
            f"- candidate_images: {hard['candidate_images']}",
            f"- fp_per_image: {hard['fp_per_image']}",
            f"- high_conf_fp_per_image: {hard['high_conf_fp_per_image']}",
            "",
            "## Rare Classes",
            "",
            "| class | GT | predictions | AP50 | AP75 | Recall50 | Precision50 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for cls, row in rare["rare_classes"].items():
        lines.append(
            f"| {cls} | {row['ground_truth']} | {row['predictions']} | {row['AP50']:.4f} | "
            f"{row['AP75']:.4f} | {row['recall50']:.4f} | {row['precision50']:.4f} |"
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TPS-YOLO paper claims from a YOLO weight file.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data-yaml", type=Path, required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--imgsz", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--predict-conf", type=float, default=0.001)
    parser.add_argument("--hard-conf", type=float, default=0.25)
    parser.add_argument("--hard-max-iou", type=float, default=0.10)
    parser.add_argument("--high-conf", type=float, default=0.50)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--predict-batch-size", type=int, default=64)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--rare-threshold", type=int, default=100)
    args = parser.parse_args()

    labels_dir = resolve_data_entry(args.data_yaml, args.split).parent.parent / "labels" / args.split
    images_dir = resolve_data_entry(args.data_yaml, args.split)
    images = image_paths(images_dir, args.max_images)
    if not images:
        raise SystemExit(f"no images found under {images_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.out_dir / "predictions.json"
    write_predictions(
        args.weights.resolve(),
        images,
        predictions_path,
        args.imgsz,
        args.predict_conf,
        args.device,
        args.max_det,
        args.predict_batch_size,
    )

    labels = load_labels(labels_dir)
    if args.max_images is not None:
        keep = {path.stem for path in images}
        labels = [label for label in labels if label.image_id in keep]
    predictions = load_predictions(predictions_path)

    area = evaluate_area_bins(labels, predictions)
    (args.out_dir / "area_metrics.json").write_text(json.dumps(area, indent=2, ensure_ascii=False), encoding="utf-8")

    candidates = mine_hard_background(labels, predictions, conf_threshold=args.hard_conf, max_iou_threshold=args.hard_max_iou)
    hard_csv = args.out_dir / "hard_background.csv"
    write_candidates(hard_csv, candidates)
    hard = evaluate_hard_background(candidates, predictions, high_conf_threshold=args.high_conf)
    (args.out_dir / "hard_background_metrics.json").write_text(json.dumps(hard, indent=2, ensure_ascii=False), encoding="utf-8")

    rare = evaluate_per_class(labels, predictions, args.rare_threshold)
    (args.out_dir / "rare_class_metrics.json").write_text(json.dumps(rare, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary(args.out_dir / "claim_eval_summary.md", area, hard, rare)
    print(f"claim evaluation saved: {args.out_dir}")


if __name__ == "__main__":
    main()
