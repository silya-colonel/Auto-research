from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TPSExperimentMatrixCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp())
        self.matrix = self.tmpdir / "matrix.yaml"
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
  - id: full_tps
    phase: screening
    data_yaml: data/welding-defect-detection-yolo/data.yaml
    model: configs/tps_yolo11n_fgdc.yaml
    pretrained_weights: yolo11n.pt
    imgsz: 640
    epochs: 20
    seed: 42
    extra:
      custom_iou_loss: sahb
      sahb_scale_weight: 1.0
      sahb_hard_bg_weight: 0.5
""",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_generates_deterministic_training_commands(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "tools" / "tps_experiment_matrix.py"),
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
        self.assertIn("--task-name full_tps", lines[1])
        self.assertIn("--model configs/tps_yolo11n_fgdc.yaml", lines[1])
        self.assertIn("--pretrained-weights yolo11n.pt", lines[1])
        self.assertIn("--extra custom_iou_loss=sahb sahb_scale_weight=1.0 sahb_hard_bg_weight=0.5", lines[1])


if __name__ == "__main__":
    unittest.main()
