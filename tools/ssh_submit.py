#!/usr/bin/env python3
"""
Submit experiments from Mac to Linux GPU server via SSH.

Replaces the ClearML-agent bundle flow with direct SSH execution.
ClearML is used only for logging (--enable-clearml), not for task queuing.

Only runs with kind=train are submitted; kind=edlr runs are local-only (Mac).

Usage:
    python tools/ssh_submit.py \\
        --matrix research/yolo11_welding_defect/experiments/dsa_edlr_ablation_matrix.yaml \\
        --phase dsa_edlr_ablation \\
        --run-id baseline_yolo11n_steel_mixed_640_e100_s42

Requirements:
    - SSH key-based auth from Mac to Linux server
    - Linux: ~/train-venv with torch + ultralytics + clearml installed
    - Linux: ~/ar/run_experiments.sh entrypoint script
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_HOST = "server03"
DEFAULT_REMOTE_ROOT = "~/ar"


def build_remote_command(run: object, args: argparse.Namespace, device: int) -> str:
    """Build the remote command line to pass via SSH."""
    # Use --default-data-yaml to override the dataset path for Linux
    # (the matrix may reference Mac-local paths)
    data_yaml = args.default_data_yaml
    parts = [
        f"{args.remote_root}/run_experiments.sh",
        run.id,
        "--data-yaml", data_yaml,
        "--model", run.model,
        "--imgsz", str(run.imgsz),
        "--epochs", str(run.epochs),
        "--batch", str(run.batch),
        "--device", str(device),
        "--workers", str(run.workers),
        "--seed", str(run.seed),
        "--runs-dir", getattr(run, "runs_dir", "runs/yolo"),
        "--clearml-project", args.clearml_project,
    ]

    if run.extra:
        extra_parts = []
        for k, v in run.extra.items():
            extra_parts.append(f"{k}={v}")
        parts.extend(["--extra"] + extra_parts)

    return " ".join(shlex.quote(p) for p in parts)


def ssh_run(host: str, remote_cmd: str, dry_run: bool = False) -> int:
    """Execute a command on the remote host via SSH."""
    ssh_cmd = ["ssh", host, remote_cmd]

    if dry_run:
        print(f"[DRY-RUN] {' '.join(ssh_cmd)}")
        return 0

    print(f"[{host}] {remote_cmd[:200]}{'...' if len(remote_cmd) > 200 else ''}")
    result = subprocess.run(ssh_cmd)
    return result.returncode


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Submit experiments via SSH to Linux GPU")
    parser.add_argument("--matrix", required=True, help="Path to experiment matrix YAML")
    parser.add_argument("--phase", default=None, help="Filter runs by phase")
    parser.add_argument("--run-id", action="append", dest="run_ids", help="Specific run IDs to submit")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"SSH host (default: {DEFAULT_HOST})")
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT, help="Path to ~/ar on remote")
    parser.add_argument("--clearml-project", default="yolo-steel-defect")
    parser.add_argument("--default-data-yaml", default="data/steel-defect-mixed/data.yaml",
                        help="Dataset data.yaml path on Linux")
    parser.add_argument("--gpus", type=int, default=4, help="Number of GPUs available")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    args = parser.parse_args(argv)

    from experiment_matrix import load_runs

    runs = load_runs(PROJECT_ROOT / args.matrix, phase=args.phase)

    if args.run_ids:
        run_ids = set(args.run_ids)
        runs = [r for r in runs if r.id in run_ids]
        if not runs:
            print(f"No runs matched --run-id filters: {args.run_ids}", file=sys.stderr)
            return 1

    # Only submit training runs; EDLR/calibration runs stay on Mac
    train_runs = [r for r in runs if getattr(r, "kind", "train") == "train"]
    skipped = [r for r in runs if getattr(r, "kind", "train") != "train"]

    if skipped:
        print(f"Skipping {len(skipped)} non-training run(s) (stay on Mac):")
        for r in skipped:
            print(f"  {r.id}: kind={r.kind}")
        print()

    if not train_runs:
        print("No training runs to submit.", file=sys.stderr)
        return 1

    print(f"Submitting {len(train_runs)} training run(s) to {args.host}:")
    for r in train_runs:
        print(f"  {r.id}: model={r.model}, epochs={r.epochs}, extra={r.extra}")
    print()

    for i, r in enumerate(train_runs):
        device = i % args.gpus
        cmd = build_remote_command(r, args, device)
        rc = ssh_run(args.host, cmd, dry_run=args.dry_run)
        if rc != 0:
            print(f"[FAILED] {r.id} returned {rc}")
            return rc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
