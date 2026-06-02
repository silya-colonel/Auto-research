from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SSHSubmitTests(unittest.TestCase):
    def test_build_train_remote_command_keeps_existing_override_behavior(self) -> None:
        sys.path.insert(0, str(PROJECT_ROOT / "tools"))
        from experiment_matrix import ExperimentRun
        from ssh_submit import build_remote_command

        run = ExperimentRun(
            id="baseline_train",
            phase="screening",
            kind="train",
            data_yaml="data/steel-defect-mixed-r2/data.yaml",
            model="yolo11n.pt",
            imgsz=640,
            epochs=20,
            seed=42,
            batch="-1",
            extra={"patience": 20},
        )
        args = argparse.Namespace(
            remote_root="~/ar",
            default_data_yaml="data/steel-defect-mixed/data.yaml",
            clearml_project="yolo-steel-defect",
        )

        command = build_remote_command(run, args, device=0)

        self.assertIn("~/ar/run_experiments.sh baseline_train", command)
        self.assertIn("--data-yaml data/steel-defect-mixed/data.yaml", command)
        self.assertIn("--extra patience=20", command)
        self.assertNotIn("steel-defect-mixed-r2", command)

    def test_build_clr_diag_remote_command(self) -> None:
        sys.path.insert(0, str(PROJECT_ROOT / "tools"))
        from experiment_matrix import ExperimentRun
        from ssh_submit import build_remote_command

        run = ExperimentRun(
            id="clr_label_reliability",
            phase="parallel_s_idea_diagnostics",
            kind="clr_diag",
            data_yaml="data/steel-defect-mixed/data.yaml",
            model="",
            imgsz=640,
            epochs=0,
            seed=42,
            weights="runs/yolo/baseline/weights/best.pt",
            out_dir="idea-stage/artifacts/parallel_s/clr",
            views=3,
            min_error_explained=0.2,
        )
        args = argparse.Namespace(
            remote_root="~/ar",
            remote_python="~/train-venv/bin/python",
            default_data_yaml="data/steel-defect-mixed/data.yaml",
            clearml_project="yolo-steel-defect",
        )

        command = build_remote_command(run, args, device=0)

        self.assertIn("~/train-venv/bin/python tools/clr_label_reliability.py", command)
        self.assertIn("--weights ~/ar/runs/yolo/baseline/weights/best.pt", command)
        self.assertIn("--data-yaml ~/ar/data/steel-defect-mixed/data.yaml", command)
        self.assertIn("--views 3", command)
        self.assertIn("--device 0", command)
        self.assertIn("--enable-clearml", command)
        self.assertIn("--clearml-task-name clr_label_reliability", command)

    def test_build_osd_diag_remote_command(self) -> None:
        sys.path.insert(0, str(PROJECT_ROOT / "tools"))
        from experiment_matrix import ExperimentRun
        from ssh_submit import build_remote_command

        run = ExperimentRun(
            id="osd_holdout_spalling",
            phase="parallel_s_idea_diagnostics",
            kind="osd_diag",
            data_yaml="data/steel-defect-mixed/data.yaml",
            model="",
            imgsz=640,
            epochs=0,
            seed=42,
            weights="runs/yolo/baseline/weights/best.pt",
            out_dir="idea-stage/artifacts/parallel_s/osd_spalling",
            unknown_class=[2, 5],
            mode="evaluate",
        )
        args = argparse.Namespace(
            remote_root="~/ar",
            remote_python="~/train-venv/bin/python",
            default_data_yaml="data/steel-defect-mixed/data.yaml",
            clearml_project="yolo-steel-defect",
        )

        command = build_remote_command(run, args, device=1)

        self.assertIn("~/train-venv/bin/python tools/osd_leave_one_class.py", command)
        self.assertIn("--mode evaluate", command)
        self.assertIn("--unknown-class 2 --unknown-class 5", command)
        self.assertIn("--weights ~/ar/runs/yolo/baseline/weights/best.pt", command)
        self.assertIn("--device 1", command)
        self.assertIn("--enable-clearml", command)
        self.assertIn("--clearml-task-name osd_holdout_spalling", command)

    def test_build_fasd_audit_remote_command(self) -> None:
        sys.path.insert(0, str(PROJECT_ROOT / "tools"))
        from experiment_matrix import ExperimentRun
        from ssh_submit import build_remote_command

        run = ExperimentRun(
            id="fasd_edge_proxy",
            phase="parallel_s_idea_diagnostics",
            kind="fasd_audit",
            data_yaml="data/steel-defect-mixed/data.yaml",
            model="",
            imgsz=640,
            epochs=0,
            seed=42,
            out_dir="idea-stage/artifacts/parallel_s/fasd",
            provider="edge_proxy",
            max_samples=64,
            min_usable_rate=0.3,
        )
        args = argparse.Namespace(
            remote_root="~/ar",
            remote_python="~/train-venv/bin/python",
            default_data_yaml="data/steel-defect-mixed/data.yaml",
            clearml_project="yolo-steel-defect",
        )

        command = build_remote_command(run, args, device=2)

        self.assertIn("~/train-venv/bin/python tools/fasd_teacher_audit.py", command)
        self.assertIn("--data-yaml ~/ar/data/steel-defect-mixed/data.yaml", command)
        self.assertIn("--provider edge_proxy", command)
        self.assertIn("--max-samples 64", command)
        self.assertIn("--device 2", command)
        self.assertIn("--enable-clearml", command)
        self.assertIn("--clearml-task-name fasd_edge_proxy", command)

    def test_dry_run_submits_parallel_s_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            matrix = Path(tmp) / "matrix.yaml"
            matrix.write_text(
                """
runs:
  - id: clr_label_reliability
    phase: parallel_s_idea_diagnostics
    kind: clr_diag
    data_yaml: data/steel-defect-mixed/data.yaml
    weights: runs/yolo/baseline/weights/best.pt
    out_dir: idea-stage/artifacts/parallel_s/clr
  - id: osd_holdout_spalling
    phase: parallel_s_idea_diagnostics
    kind: osd_diag
    data_yaml: data/steel-defect-mixed/data.yaml
    weights: runs/yolo/baseline/weights/best.pt
    out_dir: idea-stage/artifacts/parallel_s/osd_spalling
    unknown_class: [2]
  - id: fasd_edge_proxy
    phase: parallel_s_idea_diagnostics
    kind: fasd_audit
    data_yaml: data/steel-defect-mixed/data.yaml
    out_dir: idea-stage/artifacts/parallel_s/fasd
  - id: rank_parallel_s_ideas
    phase: parallel_s_idea_diagnostics
    kind: idea_rank
    clr_summary: idea-stage/artifacts/parallel_s/clr/summary.json
    osd_summary: idea-stage/artifacts/parallel_s/osd_spalling/summary.json
    fasd_summary: idea-stage/artifacts/parallel_s/fasd/summary.json
    out: idea-stage/artifacts/parallel_s/ranking.md
""",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "tools" / "ssh_submit.py"),
                    "--matrix",
                    str(matrix),
                    "--phase",
                    "parallel_s_idea_diagnostics",
                    "--remote-root",
                    "~/ar",
                    "--dry-run",
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=True,
            )

        self.assertIn("Submitting 3 run(s)", result.stdout)
        self.assertIn("kind=clr_diag", result.stdout)
        self.assertIn("kind=osd_diag", result.stdout)
        self.assertIn("kind=fasd_audit", result.stdout)
        self.assertIn("tools/clr_label_reliability.py", result.stdout)
        self.assertIn("tools/osd_leave_one_class.py", result.stdout)
        self.assertIn("tools/fasd_teacher_audit.py", result.stdout)
        self.assertIn("rank_parallel_s_ideas: kind=idea_rank", result.stderr)

    def test_background_command_wraps_shell_builtins(self) -> None:
        sys.path.insert(0, str(PROJECT_ROOT / "tools"))
        from ssh_submit import build_background_command

        command = build_background_command("cd ~/ar && echo ok", "/tmp/run.log")

        self.assertTrue(command.startswith("nohup bash -c "))
        self.assertIn("set -e; cd ~/ar && echo ok", command)
        self.assertNotIn("exec cd", command)
        self.assertIn("> /tmp/run.log 2>&1 & echo PID=$!", command)


if __name__ == "__main__":
    unittest.main()
