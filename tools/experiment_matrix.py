#!/usr/bin/env python3
"""Render experiment matrix runs as reproducible commands."""

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
    kind: str
    data_yaml: str
    model: str
    imgsz: int
    epochs: int
    seed: int
    batch: str = "-1"
    workers: int = 8
    runs_dir: str = "runs/yolo"
    extra: dict[str, Any] = field(default_factory=dict)
    weights: str | None = None
    out_dir: str | None = None
    split: str = "val"
    conf: float = 0.05
    max_det: int = 300
    max_images: int = 0
    views: int = 3
    min_error_explained: float = 0.20
    min_records: int = 100
    mode: str = "evaluate"
    unknown_class: list[int] = field(default_factory=list)
    provider: str = "edge_proxy"
    max_samples: int = 100
    min_usable_rate: float = 0.30
    clr_summary: str | None = None
    osd_summary: str | None = None
    fasd_summary: str | None = None
    out: str | None = None


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise SystemExit("PyYAML is required to read experiment matrices.") from exc
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("Experiment matrix must be a YAML mapping.")
    return payload


def parse_run(row: dict[str, Any]) -> ExperimentRun:
    kind = str(row.get("kind", "train"))
    unknown_class = row.get("unknown_class", row.get("unknown_classes", []))
    if unknown_class is None:
        unknown_classes: list[int] = []
    elif isinstance(unknown_class, list):
        unknown_classes = [int(value) for value in unknown_class]
    else:
        unknown_classes = [int(unknown_class)]
    return ExperimentRun(
        id=str(row["id"]),
        phase=str(row.get("phase", "unspecified")),
        kind=kind,
        data_yaml=str(row.get("data_yaml", "")),
        model=str(row.get("model", "")),
        imgsz=int(row.get("imgsz", 640)),
        epochs=int(row.get("epochs", 0)),
        seed=int(row.get("seed", 42)),
        batch=str(row.get("batch", "-1")),
        workers=int(row.get("workers", 8)),
        runs_dir=str(row.get("runs_dir", "runs/yolo")),
        extra=dict(row.get("extra", {}) or {}),
        weights=str(row["weights"]) if row.get("weights") else None,
        out_dir=str(row["out_dir"]) if row.get("out_dir") else None,
        split=str(row.get("split", "val")),
        conf=float(row.get("conf", 0.05)),
        max_det=int(row.get("max_det", 300)),
        max_images=int(row.get("max_images", 0)),
        views=int(row.get("views", 3)),
        min_error_explained=float(row.get("min_error_explained", 0.20)),
        min_records=int(row.get("min_records", 100)),
        mode=str(row.get("mode", "evaluate")),
        unknown_class=unknown_classes,
        provider=str(row.get("provider", "edge_proxy")),
        max_samples=int(row.get("max_samples", 100)),
        min_usable_rate=float(row.get("min_usable_rate", 0.30)),
        clr_summary=str(row["clr_summary"]) if row.get("clr_summary") else None,
        osd_summary=str(row["osd_summary"]) if row.get("osd_summary") else None,
        fasd_summary=str(row["fasd_summary"]) if row.get("fasd_summary") else None,
        out=str(row["out"]) if row.get("out") else None,
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
    if not run.data_yaml or not run.model or run.epochs <= 0:
        raise ValueError(f"Train run {run.id} requires data_yaml, model, and epochs.")
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
    if run.extra:
        parts.append("--extra")
        parts.extend(f"{key}={value}" for key, value in run.extra.items())
    return shell_join(parts)


def render_clr_diag_command(run: ExperimentRun) -> str:
    if not run.weights or not run.data_yaml or not run.out_dir:
        raise ValueError(f"CLR diagnostic run {run.id} requires weights, data_yaml, and out_dir.")
    parts = [
        "python",
        "tools/clr_label_reliability.py",
        "--weights",
        run.weights,
        "--data-yaml",
        run.data_yaml,
        "--split",
        run.split,
        "--out-dir",
        run.out_dir,
        "--views",
        str(run.views),
        "--imgsz",
        str(run.imgsz),
        "--conf",
        str(run.conf),
        "--max-det",
        str(run.max_det),
        "--min-records",
        str(run.min_records),
        "--min-error-explained",
        str(run.min_error_explained),
    ]
    if run.max_images:
        parts.extend(["--max-images", str(run.max_images)])
    return shell_join(parts)


def render_osd_diag_command(run: ExperimentRun) -> str:
    if not run.data_yaml or not run.out_dir or not run.unknown_class:
        raise ValueError(f"OSD diagnostic run {run.id} requires data_yaml, out_dir, and unknown_class.")
    parts = [
        "python",
        "tools/osd_leave_one_class.py",
        "--mode",
        run.mode,
        "--data-yaml",
        run.data_yaml,
        "--out-dir",
        run.out_dir,
    ]
    for unknown in run.unknown_class:
        parts.extend(["--unknown-class", str(unknown)])
    parts.extend(
        [
            "--split",
            run.split,
            "--imgsz",
            str(run.imgsz),
            "--conf",
            str(run.conf),
            "--max-det",
            str(run.max_det),
        ]
    )
    if run.weights:
        parts.extend(["--weights", run.weights])
    return shell_join(parts)


def render_fasd_audit_command(run: ExperimentRun) -> str:
    if not run.data_yaml or not run.out_dir:
        raise ValueError(f"FASD audit run {run.id} requires data_yaml and out_dir.")
    parts = [
        "python",
        "tools/fasd_teacher_audit.py",
        "--data-yaml",
        run.data_yaml,
        "--split",
        run.split,
        "--out-dir",
        run.out_dir,
        "--provider",
        run.provider,
        "--max-samples",
        str(run.max_samples),
        "--min-records",
        str(run.min_records),
        "--min-usable-rate",
        str(run.min_usable_rate),
    ]
    return shell_join(parts)


def render_idea_rank_command(run: ExperimentRun) -> str:
    if not run.clr_summary or not run.osd_summary or not run.fasd_summary or not run.out:
        raise ValueError(f"Idea ranking run {run.id} requires clr_summary, osd_summary, fasd_summary, and out.")
    parts = [
        "python",
        "tools/rank_parallel_ideas.py",
        "--clr-summary",
        run.clr_summary,
        "--osd-summary",
        run.osd_summary,
        "--fasd-summary",
        run.fasd_summary,
        "--out",
        run.out,
    ]
    return shell_join(parts)


def render_run_command(run: ExperimentRun) -> str:
    if run.kind == "train":
        return render_train_command(run)
    if run.kind == "clr_diag":
        return render_clr_diag_command(run)
    if run.kind == "osd_diag":
        return render_osd_diag_command(run)
    if run.kind == "fasd_audit":
        return render_fasd_audit_command(run)
    if run.kind == "idea_rank":
        return render_idea_rank_command(run)
    raise ValueError(f"Unsupported run kind={run.kind!r}. Use train, clr_diag, osd_diag, fasd_audit, or idea_rank.")


def render_commands(runs: list[ExperimentRun]) -> list[str]:
    return [render_run_command(run) for run in runs]


def main() -> None:
    parser = argparse.ArgumentParser(description="Render YOLO experiment matrix commands.")
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
