#!/usr/bin/env python3
"""
Submit experiments from Mac to Linux GPU server via SSH.

Replaces the ClearML-agent bundle flow with direct SSH execution.
ClearML is used only for logging (--enable-clearml), not for task queuing.

Runs with kind=train, kind=clr_diag, kind=osd_diag, and kind=fasd_audit are
submitted to the Linux host. Ranking remains a local post-processing step.

Usage:
    python tools/ssh_submit.py \\
        --matrix research/yolo11_welding_defect/experiments/parallel_s_idea_diagnostics.yaml \\
        --phase parallel_s_idea_diagnostics \\
        --dry-run

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


def quote_remote_word(value: object) -> str:
    """Shell-quote a remote word while preserving leading tilde expansion."""
    text = str(value)
    if text == "~":
        return "~"
    if text.startswith("~/"):
        return "~/" + shlex.quote(text[2:])
    if text == "PYTHONPATH=~":
        return text
    if text.startswith("PYTHONPATH=~/"):
        return "PYTHONPATH=~/" + shlex.quote(text[len("PYTHONPATH=~/"):])
    return shlex.quote(text)


def _remote_path(remote_root: str, path: str | None) -> str:
    if not path:
        return ""
    if path.startswith("/") or path.startswith("~"):
        return path
    return f"{remote_root}/{path}"


def build_train_remote_command(run: object, args: argparse.Namespace, device: int) -> str:
    """Build the remote training command line to pass via SSH."""
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

    return " ".join(quote_remote_word(p) for p in parts)


def _build_python_diag_command(
    run: object,
    args: argparse.Namespace,
    device: int,
    script: str,
    parts: list[str],
) -> str:
    clearml_project = getattr(args, "clearml_project", "yolo-steel-defect")
    command = [
        "cd", args.remote_root,
        "&&",
        f"PYTHONPATH={args.remote_root}",
        args.remote_python,
        script,
        *parts,
        "--device", str(device),
        "--enable-clearml",
        "--clearml-project", clearml_project,
        "--clearml-task-name", run.id,
    ]
    return " ".join(quote_remote_word(p) if p != "&&" else p for p in command)


def build_clr_diag_remote_command(run: object, args: argparse.Namespace, device: int) -> str:
    """Build the remote CLR label-reliability diagnostic command."""
    weights = _remote_path(args.remote_root, getattr(run, "weights", None))
    data_yaml = _remote_path(args.remote_root, getattr(run, "data_yaml", None) or args.default_data_yaml)
    out_dir = _remote_path(args.remote_root, getattr(run, "out_dir", None))
    if not weights or not data_yaml or not out_dir:
        raise ValueError(f"CLR diagnostic run {run.id} requires weights, data_yaml, and out_dir")

    parts = [
        "--weights", weights,
        "--data-yaml", data_yaml,
        "--split", getattr(run, "split", "val"),
        "--out-dir", out_dir,
        "--views", str(getattr(run, "views", 3)),
        "--imgsz", str(getattr(run, "imgsz", 640)),
        "--conf", str(getattr(run, "conf", 0.05)),
        "--max-det", str(getattr(run, "max_det", 300)),
        "--min-records", str(getattr(run, "min_records", 100)),
        "--min-error-explained", str(getattr(run, "min_error_explained", 0.20)),
    ]
    max_images = int(getattr(run, "max_images", 0) or 0)
    if max_images:
        parts.extend(["--max-images", str(max_images)])
    return _build_python_diag_command(run, args, device, "tools/clr_label_reliability.py", parts)


def build_osd_diag_remote_command(run: object, args: argparse.Namespace, device: int) -> str:
    """Build the remote OSD leave-one-class diagnostic command."""
    data_yaml = _remote_path(args.remote_root, getattr(run, "data_yaml", None) or args.default_data_yaml)
    out_dir = _remote_path(args.remote_root, getattr(run, "out_dir", None))
    unknown_classes = list(getattr(run, "unknown_class", []) or [])
    if not data_yaml or not out_dir or not unknown_classes:
        raise ValueError(f"OSD diagnostic run {run.id} requires data_yaml, out_dir, and unknown_class")

    parts = [
        "--mode", getattr(run, "mode", "evaluate"),
        "--data-yaml", data_yaml,
        "--out-dir", out_dir,
    ]
    for unknown in unknown_classes:
        parts.extend(["--unknown-class", str(unknown)])
    parts.extend(
        [
            "--split", getattr(run, "split", "val"),
            "--imgsz", str(getattr(run, "imgsz", 640)),
            "--conf", str(getattr(run, "conf", 0.05)),
            "--max-det", str(getattr(run, "max_det", 300)),
        ]
    )
    weights = _remote_path(args.remote_root, getattr(run, "weights", None))
    if weights:
        parts.extend(["--weights", weights])
    return _build_python_diag_command(run, args, device, "tools/osd_leave_one_class.py", parts)


def build_fasd_audit_remote_command(run: object, args: argparse.Namespace, device: int) -> str:
    """Build the remote FASD proxy-teacher audit command."""
    data_yaml = _remote_path(args.remote_root, getattr(run, "data_yaml", None) or args.default_data_yaml)
    out_dir = _remote_path(args.remote_root, getattr(run, "out_dir", None))
    if not data_yaml or not out_dir:
        raise ValueError(f"FASD audit run {run.id} requires data_yaml and out_dir")

    parts = [
        "--data-yaml", data_yaml,
        "--split", getattr(run, "split", "val"),
        "--out-dir", out_dir,
        "--provider", getattr(run, "provider", "edge_proxy"),
        "--max-samples", str(getattr(run, "max_samples", 100)),
        "--min-records", str(getattr(run, "min_records", 100)),
        "--min-usable-rate", str(getattr(run, "min_usable_rate", 0.30)),
    ]
    return _build_python_diag_command(run, args, device, "tools/fasd_teacher_audit.py", parts)


def build_remote_command(run: object, args: argparse.Namespace, device: int) -> str:
    """Build the remote command line to pass via SSH."""
    kind = getattr(run, "kind", "train")
    if kind == "train":
        return build_train_remote_command(run, args, device)
    if kind == "clr_diag":
        return build_clr_diag_remote_command(run, args, device)
    if kind == "osd_diag":
        return build_osd_diag_remote_command(run, args, device)
    if kind == "fasd_audit":
        return build_fasd_audit_remote_command(run, args, device)
    raise ValueError("Unsupported run kind={!r}. Use train, clr_diag, osd_diag, or fasd_audit.".format(kind))


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


def build_background_command(remote_cmd: str, log_file: str) -> str:
    """Wrap a remote shell command for background execution with logging."""
    script = "set -e; " + remote_cmd
    return f"nohup bash -c {shlex.quote(script)} > {shlex.quote(log_file)} 2>&1 & echo PID=$!"


def ssh_run(host: str, remote_cmd: str, run: object, dry_run: bool = False) -> int:
    """Execute a command on the remote host via SSH, return immediately."""
    log_file = f"/tmp/{run.id}_ssh.log"
    background_cmd = build_background_command(remote_cmd, log_file)
    ssh_cmd = ["ssh", host, background_cmd]

    if dry_run:
        print(f"[DRY-RUN] {remote_cmd}")
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
    parser.add_argument("--remote-python", default="~/train-venv/bin/python", help="Python executable on remote")
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

    supported_kinds = {"train", "clr_diag", "osd_diag", "fasd_audit"}
    unsupported = [r for r in runs if getattr(r, "kind", "train") not in supported_kinds]
    if unsupported:
        print(f"Warning: skipping {len(unsupported)} unsupported run kind(s):", file=sys.stderr)
        for r in unsupported:
            print(f"  {r.id}: kind={r.kind}", file=sys.stderr)
        runs = [r for r in runs if r not in unsupported]

    if not runs:
        print("No runs to submit.", file=sys.stderr)
        return 1

    print(f"Submitting {len(runs)} run(s) to {args.host}:")
    for r in runs:
        kind = getattr(r, "kind", "train")
        if kind == "train":
            print(f"  {r.id}: kind=train, model={r.model}, epochs={r.epochs}, extra={r.extra}")
        elif kind == "clr_diag":
            print(f"  {r.id}: kind=clr_diag, weights={r.weights}, out_dir={r.out_dir}")
        elif kind == "osd_diag":
            print(f"  {r.id}: kind=osd_diag, unknown_class={getattr(r, 'unknown_class', [])}, out_dir={r.out_dir}")
        else:
            print(f"  {r.id}: kind=fasd_audit, provider={getattr(r, 'provider', None)}, out_dir={r.out_dir}")
    print()

    if not args.dry_run and not args.force:
        # Pre-flight: detect duplicate submissions
        duplicates = [r for r in runs if check_remote_log_exists(args.host, r.id)]
        if duplicates:
            print("These runs already have remote logs (use --force to re-submit):")
            for r in duplicates:
                print(f"  {r.id}: /tmp/{r.id}_ssh.log exists on {args.host}")
            print()
            runs = [r for r in runs if r not in duplicates]
            if not runs:
                print("All runs already submitted. Nothing to do.")
                return 0

    for i, r in enumerate(runs):
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
