#!/usr/bin/env python3
"""OSD-YOLO leave-one-class open-set protocol helpers."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.idea_diag_common import (
    dataset_root,
    class_agnostic_best_iou,
    iter_image_paths,
    label_path_for_image,
    load_data_yaml,
    load_yolo_labels,
    normalize_predictions,
    write_json,
    yolo_result_to_predictions,
)


def filter_train_labels(lines: list[str], unknown_classes: set[int]) -> list[str]:
    kept: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        cls = int(float(stripped.split()[0]))
        if cls not in unknown_classes:
            kept.append(stripped)
    return kept


def select_candidate_unknown_classes(counts: dict[int, dict[str, int]], min_val_boxes: int = 10) -> list[int]:
    return [
        cls
        for cls, split_counts in sorted(counts.items())
        if int(split_counts.get("val", 0)) >= min_val_boxes and int(split_counts.get("train", 0)) > 0
    ]


def evaluate_unknown_proposals(
    predictions: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    *,
    unknown_classes: set[int],
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    unknown_labels = [label for label in labels if int(label["cls"]) in unknown_classes]
    recalled = 0
    for label in unknown_labels:
        best = class_agnostic_best_iou(label, predictions)
        if float(best["iou"]) >= iou_threshold:
            recalled += 1
    unknown_gt = len(unknown_labels)
    return {
        "unknown_gt": unknown_gt,
        "unknown_recalled": recalled,
        "unknown_recall": round(recalled / max(unknown_gt, 1), 6),
    }


def write_report(summary: dict[str, Any], out: Path) -> None:
    lines = [
        "# OSD-YOLO Leave-One-Class Diagnostic",
        "",
        f"- Mode: `{summary['mode']}`",
        f"- Unknown classes: `{summary['unknown_classes']}`",
        f"- Unknown GT: `{summary.get('unknown_gt', 0)}`",
        f"- Unknown recalled: `{summary.get('unknown_recalled', 0)}`",
        f"- Unknown recall: `{summary.get('unknown_recall', 0.0)}`",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_split_dataset(data_yaml: Path, output_root: Path, unknown_classes: set[int]) -> dict[str, Any]:
    payload = load_data_yaml(data_yaml)
    root = dataset_root(data_yaml, payload)
    copied_images = 0
    removed_labels = 0
    kept_labels = 0

    for split in ("train", "val", "test"):
        image_paths = iter_image_paths(data_yaml, split)
        if not image_paths:
            continue
        for image_path in image_paths:
            try:
                rel_image = image_path.resolve().relative_to(root)
            except ValueError:
                rel_image = Path("images") / split / image_path.name
            dest_image = output_root / rel_image
            dest_image.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, dest_image)
            copied_images += 1

            label_path = label_path_for_image(image_path, data_yaml)
            if not label_path.exists():
                continue
            original_lines = [line for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            filtered = filter_train_labels(original_lines, unknown_classes) if split == "train" else original_lines
            removed_labels += max(0, len(original_lines) - len(filtered))
            kept_labels += len(filtered)
            try:
                rel_label = label_path.resolve().relative_to(root)
            except ValueError:
                rel_label = Path("labels") / split / label_path.name
            dest_label = output_root / rel_label
            dest_label.parent.mkdir(parents=True, exist_ok=True)
            dest_label.write_text("\n".join(filtered) + ("\n" if filtered else ""), encoding="utf-8")

    out_yaml = output_root / "data.yaml"
    out_payload = dict(payload)
    out_payload["path"] = "."
    for split in ("train", "val", "test"):
        if payload.get(split):
            out_payload[split] = str(payload[split])
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise SystemExit("PyYAML is required to write OSD split data.yaml files.") from exc
    out_yaml.write_text(yaml.safe_dump(out_payload, sort_keys=False), encoding="utf-8")

    return {
        "mode": "build-split",
        "data_yaml": str(data_yaml),
        "output_data_yaml": str(out_yaml),
        "unknown_classes": sorted(unknown_classes),
        "copied_images": copied_images,
        "kept_labels": kept_labels,
        "removed_unknown_labels": removed_labels,
    }


def read_predictions_json(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("--predictions-json must contain a JSON list.")
    return normalize_predictions(payload)


def predict_with_yolo(
    weights: Path,
    image_paths: list[str],
    *,
    imgsz: int,
    conf: float,
    max_det: int,
    device: str | None,
) -> list[dict[str, Any]]:
    from ultralytics import YOLO  # type: ignore

    model = YOLO(str(weights))
    results = model.predict(
        source=image_paths,
        imgsz=imgsz,
        conf=conf,
        max_det=max_det,
        device=device,
        verbose=False,
    )
    return yolo_result_to_predictions(results)


def evaluate_dataset_unknowns(
    data_yaml: Path,
    split: str,
    unknown_classes: set[int],
    predictions: list[dict[str, Any]],
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    labels = load_yolo_labels(data_yaml, split)
    summary = evaluate_unknown_proposals(
        predictions,
        labels,
        unknown_classes=unknown_classes,
        iou_threshold=iou_threshold,
    )
    summary.update(
        {
            "mode": "evaluate",
            "data_yaml": str(data_yaml),
            "split": split,
            "unknown_classes": sorted(unknown_classes),
            "label_count": len(labels),
            "prediction_count": len(predictions),
        }
    )
    summary["gate"] = {
        "passes_gate": summary["unknown_gt"] > 0 and bool(predictions),
        "checks": {
            "unknown_gt_gt_zero": {
                "value": summary["unknown_gt"],
                "threshold": 1,
                "passed": summary["unknown_gt"] > 0,
            },
            "predictions_present": {
                "value": len(predictions),
                "threshold": 1,
                "passed": bool(predictions),
            },
        },
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or evaluate OSD-YOLO leave-one-class diagnostics.")
    parser.add_argument("--mode", choices=["build-split", "evaluate"], required=True)
    parser.add_argument("--data-yaml", type=Path, required=True)
    parser.add_argument("--unknown-class", type=int, action="append", required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--predictions-json", type=Path, default=None)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--split", default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--device", default=None)
    parser.add_argument("--enable-clearml", action="store_true")
    parser.add_argument("--clearml-project", default="yolo-steel-defect")
    parser.add_argument("--clearml-task-name", default=None)
    args = parser.parse_args()

    output = args.out_dir or args.output_root
    if output is None:
        raise SystemExit("--out-dir or --output-root is required")
    unknown_classes = set(args.unknown_class)
    if args.mode == "build-split":
        summary = build_split_dataset(args.data_yaml, output, unknown_classes)
        proposals: list[dict[str, Any]] = []
    else:
        predictions = read_predictions_json(args.predictions_json)
        if not predictions and args.weights and args.weights.exists():
            image_paths = [str(path) for path in iter_image_paths(args.data_yaml, args.split)]
            predictions = predict_with_yolo(
                args.weights,
                image_paths,
                imgsz=args.imgsz,
                conf=args.conf,
                max_det=args.max_det,
                device=args.device,
            )
        summary = evaluate_dataset_unknowns(
            args.data_yaml,
            args.split,
            unknown_classes,
            predictions,
            iou_threshold=args.iou_threshold,
        )
        proposals = predictions
    write_json(output / "summary.json", summary)
    write_json(output / "unknown_proposals.json", proposals)
    write_report(summary, output / "report.md")


if __name__ == "__main__":
    main()
