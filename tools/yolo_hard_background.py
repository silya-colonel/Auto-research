#!/usr/bin/env python3
"""Mine hard-background false positives from YOLO predictions."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

from yolo_area_metrics import Detection, bbox_iou, load_labels, load_predictions


@dataclass(frozen=True)
class HardBackgroundCandidate:
    image_id: str
    cls: int
    confidence: float
    x: float
    y: float
    w: float
    h: float
    max_iou: float


def max_iou_with_ground_truth(prediction: Detection, labels: list[Detection]) -> float:
    best = 0.0
    for label in labels:
        if label.image_id != prediction.image_id:
            continue
        best = max(best, bbox_iou(prediction, label))
    return best


def mine_hard_background(
    labels: list[Detection],
    predictions: list[Detection],
    conf_threshold: float,
    max_iou_threshold: float,
) -> list[HardBackgroundCandidate]:
    candidates: list[HardBackgroundCandidate] = []
    for prediction in sorted(predictions, key=lambda item: (item.image_id, -item.confidence)):
        if prediction.confidence < conf_threshold:
            continue
        max_iou = max_iou_with_ground_truth(prediction, labels)
        if max_iou >= max_iou_threshold:
            continue
        candidates.append(
            HardBackgroundCandidate(
                image_id=prediction.image_id,
                cls=prediction.cls,
                confidence=prediction.confidence,
                x=prediction.x,
                y=prediction.y,
                w=prediction.w,
                h=prediction.h,
                max_iou=max_iou,
            )
        )
    return candidates


def write_candidates(path: Path, candidates: list[HardBackgroundCandidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["image_id", "class", "confidence", "x", "y", "w", "h", "max_iou"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(
                {
                    "image_id": candidate.image_id,
                    "class": candidate.cls,
                    "confidence": f"{candidate.confidence:.6f}",
                    "x": f"{candidate.x:.6f}",
                    "y": f"{candidate.y:.6f}",
                    "w": f"{candidate.w:.6f}",
                    "h": f"{candidate.h:.6f}",
                    "max_iou": f"{candidate.max_iou:.6f}",
                }
            )


def load_candidates(path: Path) -> list[HardBackgroundCandidate]:
    candidates: list[HardBackgroundCandidate] = []
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            candidates.append(
                HardBackgroundCandidate(
                    image_id=row["image_id"],
                    cls=int(row["class"]),
                    confidence=float(row["confidence"]),
                    x=float(row["x"]),
                    y=float(row["y"]),
                    w=float(row["w"]),
                    h=float(row["h"]),
                    max_iou=float(row["max_iou"]),
                )
            )
    return candidates


def candidate_as_detection(candidate: HardBackgroundCandidate) -> Detection:
    return Detection(
        image_id=candidate.image_id,
        cls=candidate.cls,
        confidence=candidate.confidence,
        x=candidate.x,
        y=candidate.y,
        w=candidate.w,
        h=candidate.h,
    )


def candidate_has_false_positive(
    candidate: HardBackgroundCandidate,
    predictions: list[Detection],
    match_iou_threshold: float,
    conf_threshold: float,
) -> bool:
    region = candidate_as_detection(candidate)
    for prediction in predictions:
        if prediction.image_id != candidate.image_id or prediction.confidence < conf_threshold:
            continue
        if bbox_iou(region, prediction) >= match_iou_threshold:
            return True
    return False


def evaluate_hard_background(
    candidates: list[HardBackgroundCandidate],
    predictions: list[Detection],
    high_conf_threshold: float = 0.50,
    match_iou_threshold: float = 0.50,
) -> dict[str, float | int]:
    images = {candidate.image_id for candidate in candidates}
    false_positives = sum(
        1 for candidate in candidates if candidate_has_false_positive(candidate, predictions, match_iou_threshold, 0.0)
    )
    high_conf_false_positives = sum(
        1
        for candidate in candidates
        if candidate_has_false_positive(candidate, predictions, match_iou_threshold, high_conf_threshold)
    )
    image_count = len(images)
    return {
        "candidate_regions": len(candidates),
        "candidate_images": image_count,
        "false_positives": false_positives,
        "fp_per_image": round(false_positives / image_count, 6) if image_count else 0.0,
        "high_conf_threshold": high_conf_threshold,
        "high_conf_false_positives": high_conf_false_positives,
        "high_conf_fp_per_image": round(high_conf_false_positives / image_count, 6) if image_count else 0.0,
        "match_iou_threshold": match_iou_threshold,
    }


def cmd_mine(args: argparse.Namespace) -> None:
    candidates = mine_hard_background(
        labels=load_labels(args.labels),
        predictions=load_predictions(args.predictions),
        conf_threshold=args.conf,
        max_iou_threshold=args.max_iou,
    )
    write_candidates(args.out, candidates)
    print(f"hard-background candidates saved: {args.out} ({len(candidates)} rows)")


def cmd_evaluate(args: argparse.Namespace) -> None:
    candidates = load_candidates(args.candidates)
    report = evaluate_hard_background(
        candidates=candidates,
        predictions=load_predictions(args.predictions),
        high_conf_threshold=args.high_conf,
        match_iou_threshold=args.match_iou,
    )
    if args.baseline_predictions:
        baseline_report = evaluate_hard_background(
            candidates=candidates,
            predictions=load_predictions(args.baseline_predictions),
            high_conf_threshold=args.high_conf,
            match_iou_threshold=args.match_iou,
        )
        baseline_fp = int(baseline_report["false_positives"])
        current_fp = int(report["false_positives"])
        report["baseline_false_positives"] = baseline_fp
        report["baseline_fp_per_image"] = baseline_report["fp_per_image"]
        report["false_positive_reduction_rate"] = (
            round((baseline_fp - current_fp) / baseline_fp, 6) if baseline_fp else 0.0
        )
        report["baseline_high_conf_false_positives"] = baseline_report["high_conf_false_positives"]
        report["baseline_high_conf_fp_per_image"] = baseline_report["high_conf_fp_per_image"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"hard-background metrics saved: {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine or evaluate YOLO hard-background false positives.")
    sub = parser.add_subparsers(dest="command", required=True)

    mine = sub.add_parser("mine", help="Mine high-confidence low-IoU false positives.")
    mine.add_argument("--labels", type=Path, required=True, help="Directory containing YOLO label txt files.")
    mine.add_argument("--predictions", type=Path, required=True, help="JSON list of predictions.")
    mine.add_argument("--conf", type=float, default=0.25, help="Minimum prediction confidence.")
    mine.add_argument("--max-iou", type=float, default=0.10, help="Maximum GT IoU to keep as hard background.")
    mine.add_argument("--out", type=Path, required=True, help="Output candidate CSV.")
    mine.set_defaults(func=cmd_mine)

    evaluate = sub.add_parser("evaluate", help="Evaluate predictions on a mined hard-background candidate list.")
    evaluate.add_argument("--candidates", type=Path, required=True, help="Candidate CSV produced by the mine command.")
    evaluate.add_argument("--predictions", type=Path, required=True, help="JSON list of predictions to evaluate.")
    evaluate.add_argument("--baseline-predictions", type=Path, default=None, help="Optional baseline prediction JSON.")
    evaluate.add_argument("--out", type=Path, required=True, help="Output JSON metrics report.")
    evaluate.add_argument("--high-conf", type=float, default=0.50, help="High-confidence threshold for risky false positives.")
    evaluate.add_argument("--match-iou", type=float, default=0.50, help="IoU threshold for matching predictions to candidates.")
    evaluate.set_defaults(func=cmd_evaluate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
