from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ExperimentMatrixCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp())
        self.matrix = self.tmpdir / "matrix.yaml"

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_generates_deterministic_training_commands(self) -> None:
        self.matrix.write_text(
            """
runs:
  - id: baseline
    phase: screening
    data_yaml: data/welding-defect-detection-yolo/data.yaml
    model: yolo11n.pt
    imgsz: 640
    epochs: 20
    seed: 42
  - id: full_variant
    phase: screening
    data_yaml: data/welding-defect-detection-yolo/data.yaml
    model: configs/yolo11n_variant.yaml
    imgsz: 640
    epochs: 20
    seed: 42
    extra:
      custom_iou_loss: sahb
      sahb_scale_weight: 1.0
""",
            encoding="utf-8",
        )

        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "tools" / "experiment_matrix.py"),
                "--matrix",
                str(self.matrix),
                "--print-commands",
            ],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=True,
        )

        lines = [line for line in result.stdout.splitlines() if line.startswith("python train_yolo.py")]

        self.assertEqual(len(lines), 2)
        self.assertIn("--task-name baseline", lines[0])
        self.assertIn("--model yolo11n.pt", lines[0])
        self.assertIn("--epochs 20", lines[0])
        self.assertIn("--seed 42", lines[0])
        self.assertIn("--task-name full_variant", lines[1])
        self.assertIn("--model configs/yolo11n_variant.yaml", lines[1])
        self.assertIn("--extra custom_iou_loss=sahb sahb_scale_weight=1.0", lines[1])

    def test_generates_parallel_s_idea_diagnostic_commands(self) -> None:
        self.matrix.write_text(
            """
runs:
  - id: clr_label_reliability
    phase: parallel_s_idea_diagnostics
    kind: clr_diag
    data_yaml: data/steel-defect-mixed/data.yaml
    weights: runs/yolo/baseline/weights/best.pt
    out_dir: idea-stage/artifacts/parallel_s/clr
    split: val
    imgsz: 640
    views: 3
    min_error_explained: 0.2
  - id: osd_holdout_spalling
    phase: parallel_s_idea_diagnostics
    kind: osd_diag
    mode: evaluate
    data_yaml: data/steel-defect-mixed/data.yaml
    weights: runs/yolo/baseline/weights/best.pt
    out_dir: idea-stage/artifacts/parallel_s/osd_spalling
    unknown_class: [2, 5]
    split: val
    imgsz: 640
  - id: fasd_edge_proxy
    phase: parallel_s_idea_diagnostics
    kind: fasd_audit
    data_yaml: data/steel-defect-mixed/data.yaml
    out_dir: idea-stage/artifacts/parallel_s/fasd
    provider: edge_proxy
    max_samples: 64
    min_usable_rate: 0.3
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
                str(PROJECT_ROOT / "tools" / "experiment_matrix.py"),
                "--matrix",
                str(self.matrix),
                "--phase",
                "parallel_s_idea_diagnostics",
                "--print-commands",
            ],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=True,
        )

        lines = [line for line in result.stdout.splitlines() if line]

        self.assertEqual(len(lines), 4)
        self.assertTrue(lines[0].startswith("python tools/clr_label_reliability.py"))
        self.assertIn("--weights runs/yolo/baseline/weights/best.pt", lines[0])
        self.assertIn("--views 3", lines[0])
        self.assertIn("--min-error-explained 0.2", lines[0])
        self.assertTrue(lines[1].startswith("python tools/osd_leave_one_class.py"))
        self.assertIn("--mode evaluate", lines[1])
        self.assertIn("--unknown-class 2 --unknown-class 5", lines[1])
        self.assertTrue(lines[2].startswith("python tools/fasd_teacher_audit.py"))
        self.assertIn("--provider edge_proxy", lines[2])
        self.assertIn("--max-samples 64", lines[2])
        self.assertTrue(lines[3].startswith("python tools/rank_parallel_ideas.py"))
        self.assertIn("--clr-summary idea-stage/artifacts/parallel_s/clr/summary.json", lines[3])
        self.assertIn("--out idea-stage/artifacts/parallel_s/ranking.md", lines[3])


if __name__ == "__main__":
    unittest.main()
