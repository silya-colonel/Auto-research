#!/usr/bin/env python3
"""FASD-YOLO proxy-teacher quality audit."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.idea_diag_common import Box, gate_passes, write_json


@dataclass(frozen=True)
class MaskQualityConfig:
    min_coverage: float = 0.02
    max_coverage: float = 0.50
    min_quality: float = 0.10


def _box_to_pixels(box: Box, width: int, height: int) -> tuple[int, int, int, int]:
    if max(box.x1, box.y1, box.x2, box.y2) <= 1.0:
        x1 = int(round(box.x1 * width))
        y1 = int(round(box.y1 * height))
        x2 = int(round(box.x2 * width))
        y2 = int(round(box.y2 * height))
    else:
        x1 = int(round(box.x1))
        y1 = int(round(box.y1))
        x2 = int(round(box.x2))
        y2 = int(round(box.y2))

    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return x1, y1, x2, y2


def edge_proxy_mask(image: Image.Image, box: Box, threshold: int = 25) -> list[list[int]]:
    """Return a binary edge/contrast mask for a GT-box crop."""

    gray = image.convert("L")
    x1, y1, x2, y2 = _box_to_pixels(box, gray.width, gray.height)
    crop = gray.crop((x1, y1, x2, y2))
    width, height = crop.size
    pixels = crop.load()
    mask = [[0 for _x in range(width)] for _y in range(height)]

    for y in range(1, max(height - 1, 1)):
        for x in range(1, max(width - 1, 1)):
            center = int(pixels[x, y])
            contrast = max(
                abs(center - int(pixels[x - 1, y])),
                abs(center - int(pixels[x + 1, y])),
                abs(center - int(pixels[x, y - 1])),
                abs(center - int(pixels[x, y + 1])),
            )
            if contrast >= threshold:
                mask[y][x] = 1
    return mask


def score_mask_quality(mask: list[list[int]], config: MaskQualityConfig) -> dict[str, Any]:
    height = len(mask)
    width = len(mask[0]) if height else 0
    total = width * height
    positives = sum(sum(1 for value in row if value) for row in mask)
    coverage = positives / max(total, 1)

    coverage_pass = config.min_coverage <= coverage <= config.max_coverage
    if coverage <= 0:
        quality = 0.0
    elif coverage_pass:
        sparsity_margin = (config.max_coverage - coverage) / max(config.max_coverage - config.min_coverage, 1e-9)
        quality = min(1.0, (coverage / max(config.min_coverage, 1e-9))) * max(0.0, sparsity_margin)
    else:
        quality = min(coverage, 1.0) * 0.25

    usable = coverage_pass and quality >= config.min_quality
    return {
        "pixels": total,
        "positive_pixels": positives,
        "coverage": round(float(coverage), 6),
        "quality": round(float(quality), 6),
        "usable": usable,
        "bucket": "usable" if usable else "unusable",
    }


def summarize_teacher_quality(
    rows: list[dict[str, Any]],
    *,
    min_records: int = 100,
    min_usable_rate: float = 0.30,
) -> dict[str, Any]:
    usable_rows = [row for row in rows if row.get("usable")]
    usable_rate = len(usable_rows) / max(len(rows), 1)
    qualities = [float(row.get("quality", 0.0)) for row in rows]
    checks = {
        "record_count_ge_min": {"value": len(rows), "threshold": min_records, "passed": len(rows) >= min_records},
        "usable_rate_ge_min": {
            "value": round(float(usable_rate), 6),
            "threshold": min_usable_rate,
            "passed": usable_rate >= min_usable_rate,
        },
    }
    return {
        "records": len(rows),
        "usable_count": len(usable_rows),
        "usable_rate": float(usable_rate),
        "mean_quality": round(float(mean(qualities)), 6) if qualities else 0.0,
        "gate": {"checks": checks, "passes_gate": gate_passes(checks)},
    }


def write_report(summary: dict[str, Any], out: Path) -> None:
    lines = [
        "# FASD-YOLO Teacher Quality Audit",
        "",
        f"- Records: `{summary['records']}`",
        f"- Usable masks: `{summary['usable_count']}`",
        f"- Usable rate: `{summary['usable_rate']}`",
        f"- Mean quality: `{summary['mean_quality']}`",
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
    parser = argparse.ArgumentParser(description="Audit FASD-YOLO proxy teacher mask quality.")
    parser.add_argument("--data-yaml", type=Path, required=True, help="YOLO data.yaml path.")
    parser.add_argument("--split", default="val", help="Dataset split to audit.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory.")
    parser.add_argument("--provider", choices=["edge_proxy"], default="edge_proxy")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--min-usable-rate", type=float, default=0.30)
    parser.add_argument("--min-records", type=int, default=100)
    parser.add_argument("--edge-threshold", type=int, default=25)
    parser.add_argument("--device", default=None)
    parser.add_argument("--preview-dir", type=Path, default=None)
    parser.add_argument("--enable-clearml", action="store_true")
    parser.add_argument("--clearml-project", default="yolo-steel-defect")
    parser.add_argument("--clearml-task-name", default=None)
    args = parser.parse_args()

    # Full dataset traversal is intentionally deferred; this slice keeps the diagnostic API stable.
    records: list[dict[str, Any]] = []
    summary = summarize_teacher_quality(
        records,
        min_records=args.min_records,
        min_usable_rate=args.min_usable_rate,
    )
    write_json(args.out_dir / "summary.json", summary)
    write_json(args.out_dir / "teacher_quality_records.json", records)
    write_report(summary, args.out_dir / "report.md")


if __name__ == "__main__":
    main()
