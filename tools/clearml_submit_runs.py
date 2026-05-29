#!/usr/bin/env python3
"""One-command ClearML submitter for experiment matrix runs."""

from __future__ import annotations

import argparse
import datetime as dt
import shutil
import subprocess
import sys
from pathlib import Path

from experiment_matrix import ExperimentRun, load_runs


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ID = "57e9636c3b404b76991aa1cbff424f18"
DEFAULT_CODE_PROJECT = "yolo-code-bundles"
DEFAULT_TRAIN_PROJECT = "yolo-welding-defect"
DEFAULT_QUEUE = "gpu-any"
CODE_BUNDLE_ITEMS = (
    "train_yolo.py",
    "yolo11n.pt",
    "configs",
    "tools",
)


def copy_item(src: Path, dst: Path) -> None:
    if src.is_dir():
        shutil.copytree(
            src,
            dst,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store", "runs", "artifacts"),
        )
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def build_code_bundle(out_dir: Path, items: tuple[str, ...] = CODE_BUNDLE_ITEMS) -> Path:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    for rel in items:
        src = PROJECT_ROOT / rel
        if not src.exists():
            raise FileNotFoundError(f"code bundle input not found: {src}")
        copy_item(src, out_dir / rel)
    return out_dir


def upload_code_bundle(bundle_dir: Path, project: str, name: str) -> str:
    from clearml import Dataset

    dataset = Dataset.create(dataset_project=project, dataset_name=name)
    dataset.add_files(path=str(bundle_dir))
    dataset.upload()
    dataset.finalize()
    return dataset.id


def launcher_command(
    run: ExperimentRun,
    code_dataset_id: str,
    data_dataset_id: str,
    train_project: str,
    queue: str,
    device: str,
    allow_cpu: bool,
    remote_python_binary: str = "python3.12",
) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "clearml_launcher.py"),
        "--code-dataset-id",
        code_dataset_id,
        "--data-dataset-id",
        data_dataset_id,
        "--task-name",
        run.id,
        "--model",
        run.model,
        "--imgsz",
        str(run.imgsz),
        "--epochs",
        str(run.epochs),
        "--batch",
        run.batch,
        "--device",
        device,
        "--workers",
        str(run.workers),
        "--seed",
        str(run.seed),
        "--runs-dir",
        run.runs_dir,
        "--clearml-remote",
        "--clearml-project",
        train_project,
        "--clearml-queue",
        queue,
        "--remote-python-binary",
        remote_python_binary,
    ]
    if allow_cpu:
        command.append("--allow-cpu")
    if run.pretrained_weights:
        command.extend(["--pretrained-weights", run.pretrained_weights])
    if run.extra:
        command.append("--extra")
        command.extend(f"{key}={value}" for key, value in run.extra.items())
    return command


def select_runs(runs: list[ExperimentRun], run_ids: list[str], exclude_ids: list[str]) -> list[ExperimentRun]:
    selected = runs
    if run_ids:
        wanted = set(run_ids)
        selected = [run for run in selected if run.id in wanted]
        missing = wanted - {run.id for run in selected}
        if missing:
            raise SystemExit(f"run id(s) not found in matrix: {', '.join(sorted(missing))}")
    if exclude_ids:
        excluded = set(exclude_ids)
        selected = [run for run in selected if run.id not in excluded]
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload a minimal TPS code bundle and submit ClearML matrix runs.")
    parser.add_argument("--matrix", type=Path, required=True, help="Experiment matrix YAML path.")
    parser.add_argument("--phase", default=None, help="Optional matrix phase filter, e.g. screening or ablation.")
    parser.add_argument("--run-id", action="append", default=[], help="Submit only this run id. Repeatable.")
    parser.add_argument("--exclude-id", action="append", default=[], help="Exclude this run id. Repeatable.")
    parser.add_argument("--data-dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--reuse-code-dataset-id", default=None, help="Skip upload and reuse an existing code bundle.")
    parser.add_argument("--code-project", default=DEFAULT_CODE_PROJECT)
    parser.add_argument("--code-name", default=None)
    parser.add_argument("--train-project", default=DEFAULT_TRAIN_PROJECT)
    parser.add_argument("--queue", default=DEFAULT_QUEUE)
    parser.add_argument("--device", default="0")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--remote-python-binary", default="python3.12")
    parser.add_argument("--bundle-dir", type=Path, default=Path(".cache/clearml_code_bundle"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    runs = select_runs(load_runs(PROJECT_ROOT / args.matrix, phase=args.phase), args.run_id, args.exclude_id)
    if not runs:
        raise SystemExit("no runs selected")

    code_dataset_id = args.reuse_code_dataset_id
    if not code_dataset_id:
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        code_name = args.code_name or f"tps-yolo11-code-{stamp}"
        bundle_dir = build_code_bundle((PROJECT_ROOT / args.bundle_dir).resolve())
        print(f"code bundle prepared: {bundle_dir}")
        if args.dry_run:
            code_dataset_id = "<uploaded-code-dataset-id>"
            print(f"dry-run: would upload code bundle as {args.code_project}/{code_name}")
        else:
            code_dataset_id = upload_code_bundle(bundle_dir, args.code_project, code_name)
            print(f"code_dataset_id={code_dataset_id}")

    for run in runs:
        command = launcher_command(
            run,
            code_dataset_id,
            args.data_dataset_id,
            args.train_project,
            args.queue,
            args.device,
            args.allow_cpu,
            args.remote_python_binary,
        )
        printable = " ".join(command)
        print(f"\n[{run.id}] {printable}")
        if not args.dry_run:
            subprocess.run(command, cwd=str(PROJECT_ROOT), check=True, timeout=120)


if __name__ == "__main__":
    main()
