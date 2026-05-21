from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class YoloAreaMetricsCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp())
        self.labels = self.tmpdir / "labels"
        self.labels.mkdir()
        self.predictions = self.tmpdir / "predictions.json"
        self.output = self.tmpdir / "area_metrics.json"

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_reports_tiny_success_and_small_miss_from_public_cli(self) -> None:
        (self.labels / "img_a.txt").write_text("0 0.500 0.500 0.010 0.010\n", encoding="utf-8")
        (self.labels / "img_b.txt").write_text("0 0.500 0.500 0.040 0.040\n", encoding="utf-8")
        self.predictions.write_text(
            json.dumps(
                [
                    {"image_id": "img_a", "class": 0, "confidence": 0.95, "bbox": [0.5, 0.5, 0.01, 0.01]},
                    {"image_id": "img_b", "class": 0, "confidence": 0.80, "bbox": [0.1, 0.1, 0.04, 0.04]},
                ]
            ),
            encoding="utf-8",
        )

        subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "tools" / "yolo_area_metrics.py"),
                "--labels",
                str(self.labels),
                "--predictions",
                str(self.predictions),
                "--out",
                str(self.output),
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )

        report = json.loads(self.output.read_text(encoding="utf-8"))

        self.assertEqual(report["summary"]["ground_truth"], 2)
        self.assertEqual(report["bins"]["tiny"]["ground_truth"], 1)
        self.assertEqual(report["bins"]["tiny"]["true_positives"], 1)
        self.assertEqual(report["bins"]["tiny"]["recall"], 1.0)
        self.assertEqual(report["bins"]["tiny"]["AP"], 1.0)
        self.assertEqual(report["bins"]["small"]["ground_truth"], 1)
        self.assertEqual(report["bins"]["small"]["true_positives"], 0)
        self.assertEqual(report["bins"]["small"]["recall"], 0.0)

    def test_reports_ap75_separately_from_ap50(self) -> None:
        (self.labels / "img_a.txt").write_text("0 0.500 0.500 0.090 0.090\n", encoding="utf-8")
        self.predictions.write_text(
            json.dumps(
                [
                    {"image_id": "img_a", "class": 0, "confidence": 0.95, "bbox": [0.515, 0.50, 0.09, 0.09]},
                ]
            ),
            encoding="utf-8",
        )

        subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "tools" / "yolo_area_metrics.py"),
                "--labels",
                str(self.labels),
                "--predictions",
                str(self.predictions),
                "--out",
                str(self.output),
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )

        report = json.loads(self.output.read_text(encoding="utf-8"))

        self.assertEqual(report["bins"]["medium"]["AP50"], 1.0)
        self.assertEqual(report["bins"]["medium"]["AP75"], 0.0)
        self.assertEqual(report["metrics"]["AP_medium"], 1.0)
        self.assertEqual(report["metrics"]["AP75_medium"], 0.0)


if __name__ == "__main__":
    unittest.main()
