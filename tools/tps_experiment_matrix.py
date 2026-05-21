#!/usr/bin/env python3
"""Render TPS-YOLO11 experiment matrix runs as reproducible commands."""

from __future__ import annotations

import argparse
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentRun:
    id: str
    phase: str
    data_yaml: str
    model: str
    imgsz: int
    epochs: int
    seed: int
    batch: str = "-1"
    workers: int = 8
    runs_dir: str = "runs/yolo"
    pretrained_weights: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise SystemExit("PyYAML is required to read experiment matrices.") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("Experiment matrix must be a YAML mapping.")
    return payload


def parse_run(row: dict[str, Any]) -> ExperimentRun:
    return ExperimentRun(
        id=str(row["id"]),
        phase=str(row.get("phase", "unspecified")),
        data_yaml=str(row["data_yaml"]),
        model=str(row["model"]),
        imgsz=int(row.get("imgsz", 640)),
        epochs=int(row["epochs"]),
        seed=int(row.get("seed", 42)),
        batch=str(row.get("batch", "-1")),
        workers=int(row.get("workers", 8)),
        runs_dir=str(row.get("runs_dir", "runs/yolo")),
        pretrained_weights=str(row["pretrained_weights"]) if row.get("pretrained_weights") else None,
        extra=dict(row.get("extra", {}) or {}),
    )


def load_runs(path: Path, phase: str | None = None) -> list[ExperimentRun]:
    payload = load_yaml(path)
    rows = payload.get("runs", [])
    if not isinstance(rows, list):
        raise ValueError("Experiment matrix must contain a list under `runs`.")
    runs = [parse_run(row) for row in rows]
    if phase:
        runs = [run for run in runs if run.phase == phase]
    return runs


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def render_train_command(run: ExperimentRun) -> str:
    parts = [
        "python",
        "train_yolo.py",
        "train",
        "--task-name",
        run.id,
        "--data-yaml",
        run.data_yaml,
        "--model",
        run.model,
    ]
    if run.pretrained_weights:
        parts.extend(["--pretrained-weights", run.pretrained_weights])
    parts.extend(
        [
            "--imgsz",
            str(run.imgsz),
            "--epochs",
            str(run.epochs),
            "--batch",
            run.batch,
            "--workers",
            str(run.workers),
            "--seed",
            str(run.seed),
            "--runs-dir",
            run.runs_dir,
        ]
    )
    if run.extra:
        parts.append("--extra")
        parts.extend(f"{key}={value}" for key, value in run.extra.items())
    return shell_join(parts)


def render_commands(runs: list[ExperimentRun]) -> list[str]:
    return [render_train_command(run) for run in runs]


def main() -> None:
    parser = argparse.ArgumentParser(description="Render TPS-YOLO11 experiment matrix commands.")
    parser.add_argument("--matrix", type=Path, required=True, help="Experiment matrix YAML path.")
    parser.add_argument("--phase", default=None, help="Optional phase filter.")
    parser.add_argument("--print-commands", action="store_true", help="Print commands to stdout.")
    parser.add_argument("--out", type=Path, default=None, help="Optional command list output path.")
    args = parser.parse_args()

    commands = render_commands(load_runs(args.matrix, phase=args.phase))
    text = "\n".join(commands) + ("\n" if commands else "")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    if args.print_commands or not args.out:
        print(text, end="")


if __name__ == "__main__":
    main()
