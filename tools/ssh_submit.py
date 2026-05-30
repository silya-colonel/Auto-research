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
    # --default-data-yaml overrides the matrix data_yaml for Linux paths.
    # Log the override so it's auditable.
    matrix_data_yaml = getattr(run, "data_yaml", None)
    data_yaml = args.default_data_yaml
    if matrix_data_yaml and matrix_data_yaml != data_yaml:
        print(f"  [override] {run.id}: data_yaml={matrix_data_yaml!r} -> {data_yaml!r}")

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


def check_remote_log_exists(host: str, run_id: str) -> bool:
    """Check if a training log for this run already exists on the remote host."""
    log_file = f"/tmp/{run_id}_ssh.log"
    result = subprocess.run(
        ["ssh", host, f"test -f {shlex.quote(log_file)} && echo EXISTS"],
        capture_output=True, text=True, timeout=10,
    )
    return b"EXISTS" in result.stdout.encode() or "EXISTS" in result.stdout


def check_gpu_available(host: str, device: int) -> bool:
    """Quick pre-flight: check if the target GPU has enough free memory (~2GiB)."""
    cmd = f"nvidia-smi --query-gpu=memory.free --format=csv,noheader -i {device}"
    result = subprocess.run(
        ["ssh", host, cmd],
        capture_output=True, text=True, timeout=10,
    )
    try:
        free_mib = int(result.stdout.strip().split()[0])
        return free_mib > 2000
    except (ValueError, IndexError):
        return True  # can't check, don't block


def ssh_run(host: str, remote_cmd: str, run: object, dry_run: bool = False) -> int:
    """Execute a command on the remote host via SSH, return immediately."""
    log_file = f"/tmp/{run.id}_ssh.log"
    background_cmd = f"nohup {remote_cmd} > {log_file} 2>&1 & echo PID=$!"
    ssh_cmd = ["ssh", host, background_cmd]

    if dry_run:
        print(f"[DRY-RUN] {remote_cmd[:150]}...")
        return 0

    print(f"[{host}] {run.id}: {remote_cmd[:180]}...")
    result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        print(f"[FAILED] {result.stderr.strip()}")
        return result.returncode
    print(f"  -> submitted, log: {log_file}")
    if result.stdout.strip():
        print(f"  -> {result.stdout.strip()}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Submit experiments via SSH to Linux GPU")
    parser.add_argument("--matrix", required=True, help="Path to experiment matrix YAML")
    parser.add_argument("--phase", default=None, help="Filter runs by phase")
    parser.add_argument("--run-id", action="append", dest="run_ids", help="Specific run IDs to submit")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"SSH host (default: {DEFAULT_HOST})")
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT, help="Path to ~/ar on remote")
    parser.add_argument("--clearml-project", default="yolo-steel-defect")
    parser.add_argument("--default-data-yaml", default="data/steel-defect-mixed/data.yaml",
                        help="Dataset data.yaml path on Linux (overrides matrix)")
    parser.add_argument("--gpus", type=int, default=4, help="Number of GPUs available")
    parser.add_argument("--force", action="store_true", help="Allow re-submitting already-submitted runs")
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

    if not args.dry_run and not args.force:
        # Pre-flight: detect duplicate submissions
        duplicates = [r for r in train_runs if check_remote_log_exists(args.host, r.id)]
        if duplicates:
            print("These runs already have remote logs (use --force to re-submit):")
            for r in duplicates:
                print(f"  {r.id}: /tmp/{r.id}_ssh.log exists on {args.host}")
            print()
            train_runs = [r for r in train_runs if r not in duplicates]
            if not train_runs:
                print("All runs already submitted. Nothing to do.")
                return 0

    for i, r in enumerate(train_runs):
        device = i % args.gpus

        if not args.dry_run and not check_gpu_available(args.host, device):
            print(f"  [warn] GPU {device} low on memory, run may OOM")

        cmd = build_remote_command(r, args, device)
        rc = ssh_run(args.host, cmd, r, dry_run=args.dry_run)
        if rc != 0:
            print(f"[FAILED] {r.id} returned {rc}")
            return rc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
