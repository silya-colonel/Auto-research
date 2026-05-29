#!/usr/bin/env python3
"""Download a completed ClearML training task and run claim evaluation."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from clearml import Task


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def copy_artifact(task: Task, name: str, out_dir: Path) -> Path:
    artifacts = task.artifacts or {}
    if name not in artifacts:
        raise SystemExit(f"artifact {name!r} not found on task {task.id}; available={list(artifacts)}")
    local = Path(artifacts[name].get_local_copy()).resolve()
    target = out_dir / "artifacts" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(local, target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a completed ClearML YOLO task on TPS paper claims.")
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--data-yaml", default="data/welding-defect-detection-yolo/data.yaml")
    parser.add_argument("--imgsz", type=int, required=True)
    parser.add_argument("--out-root", type=Path, default=Path("idea-stage/artifacts/claim_eval"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument(
        "--allow-unfinished",
        action="store_true",
        help="Allow evaluation before ClearML marks the task completed if required artifacts are already available.",
    )
    args = parser.parse_args()

    task = Task.get_task(task_id=args.task_id)
    print(f"task {args.task_id} status={task.status}")
    if task.status != "completed" and not args.allow_unfinished:
        raise SystemExit(f"task not completed yet: {task.status}")

    out_dir = (PROJECT_ROOT / args.out_root / args.task_name).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = copy_artifact(task, "best.pt", out_dir)
    metrics = copy_artifact(task, "metrics.json", out_dir)
    copy_artifact(task, "results.csv", out_dir)
    copied = {
        "task_id": args.task_id,
        "task_name": args.task_name,
        "status": task.status,
        "weights": str(weights),
        "metrics_json": json.loads(metrics.read_text(encoding="utf-8")),
    }
    (out_dir / "clearml_task.json").write_text(json.dumps(copied, indent=2, ensure_ascii=False), encoding="utf-8")

    command = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "claim_eval.py"),
        "--weights",
        str(weights),
        "--data-yaml",
        str(PROJECT_ROOT / args.data_yaml),
        "--split",
        "val",
        "--imgsz",
        str(args.imgsz),
        "--out-dir",
        str(out_dir),
    ]
    if args.device:
        command.extend(["--device", args.device])
    if args.max_images is not None:
        command.extend(["--max-images", str(args.max_images)])
    subprocess.run(command, cwd=str(PROJECT_ROOT), check=True, timeout=3600)
    print(f"ClearML task claim evaluation saved: {out_dir}")


if __name__ == "__main__":
    main()
