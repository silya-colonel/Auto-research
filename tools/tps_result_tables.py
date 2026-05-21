#!/usr/bin/env python3
"""Aggregate TPS-YOLO11 metrics into paper-ready Markdown and CSV tables."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


NA = "NA"


@dataclass
class RunRecord:
    run_id: str
    phase: str = "unspecified"
    metrics: dict[str, Any] = field(default_factory=dict)
    area_metrics: dict[str, Any] = field(default_factory=dict)
    hard_bg_metrics: dict[str, Any] = field(default_factory=dict)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def flatten_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = payload.get("metrics", payload)
    return metrics if isinstance(metrics, dict) else {}


def infer_phase(run_id: str) -> str:
    if "_e20_" in run_id:
        return "screening"
    if "xray" in run_id:
        return "xray"
    if "_e100_" in run_id:
        return "ablation"
    if "smoke" in run_id:
        return "smoke"
    return "unspecified"


def metric_value(metrics: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in metrics:
            return metrics[name]
    return None


def format_value(value: Any) -> str:
    if value is None:
        return NA
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        text = str(value)
        return text if text else NA


def collect_run_metrics(runs_dir: Path) -> list[RunRecord]:
    records: list[RunRecord] = []
    if not runs_dir.exists():
        return records
    for path in sorted(runs_dir.glob("**/metrics.json")):
        payload = read_json(path)
        run_id = str(payload.get("task_name") or path.parent.name)
        records.append(
            RunRecord(
                run_id=run_id,
                phase=str(payload.get("phase") or infer_phase(run_id)),
                metrics=flatten_metrics(payload),
            )
        )
    return records


def match_metric_file(extra_metrics_dir: Path, run_id: str, suffixes: tuple[str, ...]) -> Path | None:
    for suffix in suffixes:
        direct = extra_metrics_dir / f"{run_id}_{suffix}.json"
        if direct.exists():
            return direct
    for path in sorted(extra_metrics_dir.glob("*.json")):
        if path.stem.startswith(run_id) and any(token in path.stem for token in suffixes):
            return path
    return None


def attach_extra_metrics(records: list[RunRecord], extra_metrics_dir: Path | None) -> None:
    if extra_metrics_dir is None or not extra_metrics_dir.exists():
        return
    for record in records:
        area_path = match_metric_file(extra_metrics_dir, record.run_id, ("area_metrics", "area"))
        if area_path:
            record.area_metrics = flatten_metrics(read_json(area_path))
        hard_bg_path = match_metric_file(extra_metrics_dir, record.run_id, ("hard_bg", "hard_background"))
        if hard_bg_path:
            record.hard_bg_metrics = flatten_metrics(read_json(hard_bg_path))


def infer_components(run_id: str) -> tuple[str, str, str]:
    neck = "FGDC-FPN" if "fgdc" in run_id or "tps" in run_id else "YOLO11"
    loss = "SAHB" if "sahb" in run_id or "tps" in run_id else "CIoU/DFL"
    hard_bg = "off" if "no_hardbg" in run_id else ("on" if loss == "SAHB" else "off")
    if "no_scale" in run_id:
        loss = "SAHB(no-scale)"
    return neck, loss, hard_bg


def ordered_records(records: list[RunRecord]) -> list[RunRecord]:
    def key(record: RunRecord) -> tuple[int, str]:
        priority = 0 if record.run_id.startswith("baseline") else 1
        if record.run_id.startswith("fgdc"):
            priority = 2
        if record.run_id.startswith("sahb"):
            priority = 3
        if record.run_id.startswith("tps"):
            priority = 4
        return priority, record.run_id

    return sorted(records, key=key)


def make_rows(records: list[RunRecord]) -> dict[str, tuple[list[str], list[list[str]]]]:
    main_header = ["Run", "Phase", "mAP50", "mAP50-95", "Precision", "Recall"]
    main_rows = [
        [
            record.run_id,
            record.phase,
            format_value(metric_value(record.metrics, "mAP50", "map50")),
            format_value(metric_value(record.metrics, "mAP50-95", "map", "map50_95")),
            format_value(metric_value(record.metrics, "precision", "mp")),
            format_value(metric_value(record.metrics, "recall", "mr")),
        ]
        for record in ordered_records(records)
    ]

    ablation_header = [
        "Run",
        "Neck",
        "Loss",
        "Hard-bg",
        "mAP50",
        "mAP50-95",
        "AP_tiny",
        "AP75_tiny",
        "FP/Image",
        "FP Reduction",
    ]
    ablation_rows = []
    for record in ordered_records(records):
        neck, loss, hard_bg = infer_components(record.run_id)
        ablation_rows.append(
            [
                record.run_id,
                neck,
                loss,
                hard_bg,
                format_value(metric_value(record.metrics, "mAP50", "map50")),
                format_value(metric_value(record.metrics, "mAP50-95", "map", "map50_95")),
                format_value(record.area_metrics.get("AP_tiny")),
                format_value(record.area_metrics.get("AP75_tiny")),
                format_value(record.hard_bg_metrics.get("fp_per_image")),
                format_value(record.hard_bg_metrics.get("false_positive_reduction_rate")),
            ]
        )

    tiny_header = ["Run", "AP_tiny", "AP75_tiny", "Recall_tiny", "AP_small", "AP75_small", "Recall_small"]
    tiny_rows = [
        [
            record.run_id,
            format_value(record.area_metrics.get("AP_tiny")),
            format_value(record.area_metrics.get("AP75_tiny")),
            format_value(record.area_metrics.get("Recall_tiny")),
            format_value(record.area_metrics.get("AP_small")),
            format_value(record.area_metrics.get("AP75_small")),
            format_value(record.area_metrics.get("Recall_small")),
        ]
        for record in ordered_records(records)
    ]

    hard_header = ["Run", "FP/Image", "High-conf FP/Image", "False Positives", "FP Reduction"]
    hard_rows = [
        [
            record.run_id,
            format_value(record.hard_bg_metrics.get("fp_per_image")),
            format_value(record.hard_bg_metrics.get("high_conf_fp_per_image")),
            format_value(record.hard_bg_metrics.get("false_positives")),
            format_value(record.hard_bg_metrics.get("false_positive_reduction_rate")),
        ]
        for record in ordered_records(records)
    ]

    efficiency_header = ["Run", "Params(M)", "FLOPs(G)", "FPS", "Latency(ms)"]
    efficiency_rows = [
        [
            record.run_id,
            format_value(metric_value(record.metrics, "params_m", "parameters_m", "params")),
            format_value(metric_value(record.metrics, "flops_g", "flops")),
            format_value(metric_value(record.metrics, "fps")),
            format_value(metric_value(record.metrics, "latency_ms", "inference_ms")),
        ]
        for record in ordered_records(records)
    ]

    return {
        "main_comparison": (main_header, main_rows),
        "ablation": (ablation_header, ablation_rows),
        "tiny_object": (tiny_header, tiny_rows),
        "hard_background": (hard_header, hard_rows),
        "efficiency": (efficiency_header, efficiency_rows),
    }


def render_markdown(header: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def write_tables(tables: dict[str, tuple[list[str], list[list[str]]]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, (header, rows) in tables.items():
        (out_dir / f"{name}.md").write_text(render_markdown(header, rows), encoding="utf-8")
        write_csv(out_dir / f"{name}.csv", header, rows)


def build_result_tables(run_metrics: list[dict[str, Any]]) -> dict[str, str]:
    records = [
        RunRecord(
            run_id=str(row.get("task_name") or row.get("run_id")),
            phase=str(row.get("phase") or infer_phase(str(row.get("task_name") or row.get("run_id")))),
            metrics=flatten_metrics(row),
        )
        for row in run_metrics
    ]
    return {name: render_markdown(header, rows) for name, (header, rows) in make_rows(records).items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate TPS-YOLO11 result artifacts into paper-ready tables.")
    parser.add_argument("--runs", type=Path, required=True, help="Directory containing run subdirectories with metrics.json.")
    parser.add_argument("--extra-metrics", type=Path, default=None, help="Directory containing area/hard-background JSON metrics.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for Markdown and CSV tables.")
    args = parser.parse_args()

    records = collect_run_metrics(args.runs)
    attach_extra_metrics(records, args.extra_metrics)
    tables = make_rows(records)
    write_tables(tables, args.out)
    print(f"result tables saved: {args.out} ({len(records)} runs)")


if __name__ == "__main__":
    main()
