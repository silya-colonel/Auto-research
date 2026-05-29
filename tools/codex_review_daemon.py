#!/usr/bin/env python3
"""Codex Review Daemon — watches for Codex completion signals and triggers Claude review.

Polls .codex_review_queue/ for pending review signals. When a signal is detected,
launches Claude Code to audit Codex's work and writes a review report.

Usage:
    python tools/codex_review_daemon.py --once
    python tools/codex_review_daemon.py --poll 60
    nohup python tools/codex_review_daemon.py --poll 120 --log daemon.log > /dev/null 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
QUEUE_DIR = REPO_ROOT / ".codex_review_queue"
REVIEW_DIR = REPO_ROOT / "results" / "codex_reviews"
PROMPT_TEMPLATE = REPO_ROOT / "tools" / "codex_review_prompt.md"

# State machine: pending → reviewing → reviewed
STATE_PENDING = "pending"
STATE_REVIEWING = "reviewing"
STATE_REVIEWED = "reviewed"
STATE_FAILED = "failed"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_prompt_template() -> str:
    if PROMPT_TEMPLATE.exists():
        return PROMPT_TEMPLATE.read_text(encoding="utf-8")
    return "Review the following Codex work. Be thorough and critical."


def build_review_prompt(signal_data: dict) -> str:
    template = load_prompt_template()
    ctx = signal_data.get("context", {})
    ctx_summary = json.dumps(ctx, indent=2, ensure_ascii=False)
    task_id = signal_data.get("task_id", "unknown")
    timestamp = signal_data.get("timestamp", now_iso())
    trigger = signal_data.get("trigger", "codex_post_exec")

    prompt = template.replace("{task_id}", task_id)
    prompt = prompt.replace("{timestamp}", timestamp)
    prompt = prompt.replace("{trigger}", trigger)
    prompt = prompt.replace("{signal_context}", ctx_summary)

    focus = signal_data.get("review_focus", [])
    if focus:
        prompt += f"\n\n## 特别关注领域\n\n{', '.join(focus)}\n"

    prompt += f"""

## 工作目录

项目根目录: {REPO_ROOT}

请使用 Read 工具直接读取相关文件进行检查。重点检查信号中 listed 的 changed_files 和 experiment_outputs。
如果 changed_files 中有 Python 脚本，请阅读它们并检查正确性。
如果 experiment_outputs 中有 metrics.json，请读取并检查数值合理性。

完成审查后，将完整报告保存到: {REVIEW_DIR}/{task_id}_{signal_data.get('signal_id', 'unknown')}.md
"""

    return prompt


def get_pending_signals() -> list[Path]:
    if not QUEUE_DIR.is_dir():
        return []
    signals = sorted(QUEUE_DIR.glob("*.json"))
    return signals


def move_to_reviewing(signal_path: Path) -> Path:
    reviewing_name = signal_path.name.replace(".json", ".reviewing")
    reviewing_path = signal_path.with_name(reviewing_name)
    signal_path.rename(reviewing_path)
    return reviewing_path


def move_to_reviewed(signal_path: Path) -> Path:
    reviewed_name = signal_path.name.replace(".reviewing", ".reviewed")
    reviewed_path = signal_path.with_name(reviewed_name)
    signal_path.rename(reviewed_path)
    return reviewed_path


def mark_failed(signal_path: Path) -> Path:
    failed_name = signal_path.name.replace(".reviewing", ".failed").replace(".json", ".failed")
    failed_path = signal_path.with_name(failed_name)
    signal_path.rename(failed_path)
    return failed_path


def invoke_claude_review(prompt: str, timeout: int = 600) -> tuple[str, int]:
    """Invoke Claude Code CLI to perform the review. Returns (output, returncode)."""
    env = os.environ.copy()
    env["CLAUDE_CODE_EFFORT_LEVEL"] = "max"

    r = subprocess.run(
        ["claude", "-p", prompt],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(REPO_ROOT),
        env=env,
    )
    return r.stdout.strip(), r.returncode


def save_review_report(signal_data: dict, review_output: str) -> Path:
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    task_id = signal_data.get("task_id", "unknown")
    signal_id = signal_data.get("signal_id", "unknown")

    report_path = REVIEW_DIR / f"{task_id}_{signal_id}.md"
    header = f"""# Codex 审查报告

- **任务**: {task_id}
- **审查时间**: {now_iso()}
- **触发**: {signal_data.get('trigger', 'codex_post_exec')}
- **关注领域**: {', '.join(signal_data.get('review_focus', []))}

---

"""
    report_path.write_text(header + review_output, encoding="utf-8")
    return report_path


def process_signal(signal_path: Path, timeout: int = 600) -> bool:
    """Process a single review signal. Returns True on success."""
    try:
        signal_data = json.loads(signal_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"[{now_iso()}] ERROR: Cannot read signal {signal_path}: {e}")
        return False

    task_id = signal_data.get("task_id", signal_path.stem)
    print(f"[{now_iso()}] Reviewing: {task_id}")

    # Move to reviewing state
    reviewing_path = move_to_reviewing(signal_path)

    try:
        prompt = build_review_prompt(signal_data)
        review_output, rc = invoke_claude_review(prompt, timeout=timeout)

        if rc != 0:
            print(f"[{now_iso()}] WARNING: Claude exited with code {rc}")
            if not review_output:
                print(f"[{now_iso()}] ERROR: No output from Claude review")
                mark_failed(reviewing_path)
                return False

        report_path = save_review_report(signal_data, review_output)
        move_to_reviewed(reviewing_path)
        print(f"[{now_iso()}] Review saved: {report_path}")
        return True

    except subprocess.TimeoutExpired:
        print(f"[{now_iso()}] ERROR: Review timed out for {task_id}")
        mark_failed(reviewing_path)
        return False
    except Exception as e:
        print(f"[{now_iso()}] ERROR: Review failed for {task_id}: {e}")
        mark_failed(reviewing_path)
        return False


def cleanup_stale_reviewing(timeout_seconds: int = 3600):
    """Reset .reviewing files older than timeout back to .json for retry."""
    if not QUEUE_DIR.is_dir():
        return
    for path in QUEUE_DIR.glob("*.reviewing"):
        age = time.time() - path.stat().st_mtime
        if age > timeout_seconds:
            original = path.with_name(path.name.replace(".reviewing", ".json"))
            path.rename(original)
            print(f"[{now_iso()}] Reset stale review: {path.name}")


def is_inside_claude_session() -> bool:
    """Detect if we're running inside a Claude Code session."""
    return bool(os.environ.get("CLAUDE_CODE_SESSION_ID") or
                os.environ.get("ANTHROPIC_BASE_URL"))


def run_once(timeout: int = 600, max_reviews: int = 5):
    """Process pending signals once and exit."""
    signals = get_pending_signals()
    if not signals:
        print(f"[{now_iso()}] No pending reviews")
        return

    # Auto-degrade if inside a Claude session
    if is_inside_claude_session():
        print(f"[{now_iso()}] Detected Claude session — using write-prompt mode")
        write_prompt_file()
        return

    print(f"[{now_iso()}] Found {len(signals)} pending review(s)")
    count = 0
    for signal_path in signals[:max_reviews]:
        process_signal(signal_path, timeout=timeout)
        count += 1
    print(f"[{now_iso()}] Processed {count} review(s)")


def write_prompt_file(output_path: str | None = None):
    """Build a combined review prompt from all pending signals and write to a file.

    This mode works from within a Claude Code session — instead of spawning a
    subprocess, it writes the prompt for the current session to act on.
    """
    signals = get_pending_signals()
    if not signals:
        print(f"[{now_iso()}] No pending reviews")
        return None

    REVIEW_DIR.mkdir(parents=True, exist_ok=True)

    prompts = []
    for signal_path in signals:
        try:
            signal_data = json.loads(signal_path.read_text(encoding="utf-8"))
        except Exception:
            print(f"[{now_iso()}] WARNING: cannot read signal {signal_path}", file=sys.stderr)
            continue
        prompts.append(build_review_prompt(signal_data))

    if not prompts:
        print(f"[{now_iso()}] No valid signals to review")
        return None

    combined = "\n\n---\n\n".join(prompts)

    out_path = Path(output_path) if output_path else REVIEW_DIR / f"review_prompt_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(combined, encoding="utf-8")

    # Move all signals to reviewing state
    for signal_path in signals:
        try:
            move_to_reviewing(signal_path)
        except Exception:
            pass

    print(f"[{now_iso()}] Review prompt written: {out_path}")
    print(f"  Signals: {len(signals)}")
    print(f"  请在 Claude Code 中执行: claude -p \"$(cat {out_path})\"")
    print(f"  或在当前会话中继续审查")

    return out_path


def run_daemon(poll_interval: int = 60, timeout: int = 600):
    """Run as a background daemon, polling for new signals."""
    print(f"[{now_iso()}] Daemon started (poll={poll_interval}s, review_timeout={timeout}s)")
    print(f"[{now_iso()}] Queue dir: {QUEUE_DIR}")
    print(f"[{now_iso()}] Reports dir: {REVIEW_DIR}")
    sys.stdout.flush()

    QUEUE_DIR.mkdir(parents=True, exist_ok=True)

    stop_flag = False

    def handle_signal(sig, frame):
        nonlocal stop_flag
        print(f"\n[{now_iso()}] Shutting down...")
        stop_flag = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    while not stop_flag:
        cleanup_stale_reviewing()
        signals = get_pending_signals()
        if signals:
            print(f"[{now_iso()}] {len(signals)} pending review(s)")
            for signal_path in signals[:3]:  # max 3 per cycle
                process_signal(signal_path, timeout=timeout)

        time.sleep(poll_interval)

    print(f"[{now_iso()}] Daemon stopped")


def main():
    ap = argparse.ArgumentParser(description="Codex Review Daemon")
    ap.add_argument("--once", action="store_true",
                    help="Process pending signals once and exit")
    ap.add_argument("--write-prompt", default=None, const="auto", nargs="?",
                    help="Write combined review prompt to file (path optional). "
                         "Use this within a Claude session instead of --once.")
    ap.add_argument("--poll", type=int, default=None,
                    help="Poll interval in seconds (daemon mode)")
    ap.add_argument("--max-reviews", type=int, default=5,
                    help="Max reviews per cycle (default: 5)")
    ap.add_argument("--timeout", type=int, default=600,
                    help="Per-review timeout in seconds (default: 600)")
    ap.add_argument("--log", default=None, help="Log file path")
    args = ap.parse_args()

    if args.log:
        log_path = Path(args.log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        sys.stdout = open(args.log, "a")
        sys.stderr = sys.stdout

    if args.once:
        run_once(timeout=args.timeout, max_reviews=args.max_reviews)
    elif args.write_prompt is not None:
        output = None if args.write_prompt == "auto" else args.write_prompt
        write_prompt_file(output)
    elif args.poll:
        run_daemon(poll_interval=args.poll, timeout=args.timeout)
    else:
        print("Usage:")
        print("  python tools/codex_review_daemon.py --once          # process and invoke claude")
        print("  python tools/codex_review_daemon.py --write-prompt   # write prompt for current session")
        print("  python tools/codex_review_daemon.py --poll 60        # daemon mode")


if __name__ == "__main__":
    main()
