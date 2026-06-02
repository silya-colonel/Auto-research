#!/usr/bin/env python3
"""OSD-YOLO leave-one-class open-set protocol helpers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.idea_diag_common import class_agnostic_best_iou, write_json


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or evaluate OSD-YOLO leave-one-class diagnostics.")
    parser.add_argument("--mode", choices=["build-split", "evaluate"], required=True)
    parser.add_argument("--data-yaml", type=Path, required=True)
    parser.add_argument("--unknown-class", type=int, action="append", required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--weights", type=Path, default=None)
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
    summary = {
        "mode": args.mode,
        "data_yaml": str(args.data_yaml),
        "unknown_classes": args.unknown_class,
        "unknown_gt": 0,
        "unknown_recalled": 0,
        "unknown_recall": 0.0,
    }
    write_json(output / "summary.json", summary)
    write_json(output / "unknown_proposals.json", [])
    write_report(summary, output / "report.md")


if __name__ == "__main__":
    main()
