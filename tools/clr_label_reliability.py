#!/usr/bin/env python3
"""CLR-YOLO label reliability diagnostic."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.idea_diag_common import gate_passes, write_json


@dataclass(frozen=True)
class ReliabilityConfig:
    high_threshold: float = 0.70
    low_threshold: float = 0.45


def _agreement(values: list[Any]) -> float:
    non_null = [value for value in values if value is not None]
    if not non_null:
        return 0.0
    counts = {value: non_null.count(value) for value in set(non_null)}
    return max(counts.values()) / len(values)


def score_label_reliability(view_rows: list[dict[str, Any]], config: ReliabilityConfig) -> dict[str, Any]:
    if not view_rows:
        return {"reliability": 0.0, "bucket": "low", "match_rate": 0.0, "mean_iou": 0.0, "class_agreement": 0.0}
    match_rate = mean(1.0 if row.get("matched") else 0.0 for row in view_rows)
    mean_iou = mean(float(row.get("same_class_iou", 0.0)) for row in view_rows)
    class_agreement = _agreement([row.get("pred_cls") for row in view_rows])
    reliability = 0.45 * match_rate + 0.35 * mean_iou + 0.20 * class_agreement
    if reliability >= config.high_threshold:
        bucket = "high"
    elif reliability < config.low_threshold:
        bucket = "low"
    else:
        bucket = "mid"
    return {
        "reliability": round(float(reliability), 6),
        "bucket": bucket,
        "match_rate": round(float(match_rate), 6),
        "mean_iou": round(float(mean_iou), 6),
        "class_agreement": round(float(class_agreement), 6),
    }


def summarize_reliability(
    rows: list[dict[str, Any]],
    *,
    min_records: int = 100,
    min_error_explained: float = 0.20,
) -> dict[str, Any]:
    low_rows = [row for row in rows if row.get("bucket") == "low"]
    error_rows = [row for row in rows if row.get("baseline_error")]
    low_error_rows = [row for row in low_rows if row.get("baseline_error")]
    low_reliability_error_fraction = len(low_error_rows) / max(len(error_rows), 1)
    checks = {
        "record_count_ge_min": {"value": len(rows), "threshold": min_records, "passed": len(rows) >= min_records},
        "low_reliability_explains_errors": {
            "value": round(low_reliability_error_fraction, 6),
            "threshold": min_error_explained,
            "passed": low_reliability_error_fraction >= min_error_explained,
        },
    }
    return {
        "records": len(rows),
        "low_reliability_count": len(low_rows),
        "baseline_error_count": len(error_rows),
        "low_reliability_error_fraction": round(low_reliability_error_fraction, 6),
        "gate": {"checks": checks, "passes_gate": gate_passes(checks)},
    }


def write_report(summary: dict[str, Any], out: Path) -> None:
    lines = [
        "# CLR-YOLO Label Reliability Diagnostic",
        "",
        f"- Records: `{summary['records']}`",
        f"- Low-reliability labels: `{summary['low_reliability_count']}`",
        f"- Baseline errors: `{summary['baseline_error_count']}`",
        f"- Low-reliability error fraction: `{summary['low_reliability_error_fraction']}`",
        f"- Gate passed: `{summary['gate']['passes_gate']}`",
        "",
        "## Gate Checks",
        "",
        "| Check | Value | Threshold | Passed |",
        "|---|---:|---:|---|",
    ]
    for name, check in summary["gate"]["checks"].items():
        lines.append(f"| {name} | {check['value']} | {check['threshold']} | {check['passed']} |")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate label reliability for CLR-YOLO diagnostics.")
    parser.add_argument("--weights", type=Path, required=False, help="Baseline detector weights.")
    parser.add_argument("--data-yaml", type=Path, required=False, help="YOLO data.yaml path.")
    parser.add_argument("--split", default="val", help="Dataset split to diagnose.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory.")
    parser.add_argument("--views", type=int, default=3, help="Number of augmented prediction views.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--min-records", type=int, default=100)
    parser.add_argument("--min-error-explained", type=float, default=0.20)
    parser.add_argument("--enable-clearml", action="store_true")
    parser.add_argument("--clearml-project", default="yolo-steel-defect")
    parser.add_argument("--clearml-task-name", default=None)
    args = parser.parse_args()

    # Full YOLO inference is added in the next slice. Keep the CLI output shape stable now.
    records: list[dict[str, Any]] = []
    summary = summarize_reliability(
        records,
        min_records=args.min_records,
        min_error_explained=args.min_error_explained,
    )
    write_json(args.out_dir / "summary.json", summary)
    write_json(args.out_dir / "reliability_records.json", records)
    write_report(summary, args.out_dir / "report.md")


if __name__ == "__main__":
    main()
