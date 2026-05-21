from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class YoloHardBackgroundCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp())
        self.labels = self.tmpdir / "labels"
        self.labels.mkdir()
        self.predictions = self.tmpdir / "predictions.json"
        self.output = self.tmpdir / "hard_background.csv"

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_mines_only_high_confidence_low_iou_false_positives(self) -> None:
        (self.labels / "img_a.txt").write_text("0 0.500 0.500 0.100 0.100\n", encoding="utf-8")
        self.predictions.write_text(
            json.dumps(
                [
                    {"image_id": "img_a", "class": 0, "confidence": 0.95, "bbox": [0.5, 0.5, 0.10, 0.10]},
                    {"image_id": "img_a", "class": 0, "confidence": 0.80, "bbox": [0.1, 0.1, 0.10, 0.10]},
                    {"image_id": "img_a", "class": 0, "confidence": 0.10, "bbox": [0.8, 0.8, 0.10, 0.10]},
                ]
            ),
            encoding="utf-8",
        )

        subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "tools" / "yolo_hard_background.py"),
                "mine",
                "--labels",
                str(self.labels),
                "--predictions",
                str(self.predictions),
                "--conf",
                "0.25",
                "--max-iou",
                "0.10",
                "--out",
                str(self.output),
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )

        with self.output.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["image_id"], "img_a")
        self.assertEqual(rows[0]["class"], "0")
        self.assertEqual(rows[0]["confidence"], "0.800000")
        self.assertEqual(rows[0]["max_iou"], "0.000000")

    def test_evaluates_false_positive_reduction_on_hard_background_candidates(self) -> None:
        candidates = self.tmpdir / "candidates.csv"
        baseline_predictions = self.tmpdir / "baseline_predictions.json"
        candidate_predictions = self.tmpdir / "candidate_predictions.json"
        metrics_out = self.tmpdir / "hard_background_metrics.json"

        candidates.write_text(
            "\n".join(
                [
                    "image_id,class,confidence,x,y,w,h,max_iou",
                    "img_a,0,0.900000,0.200000,0.200000,0.100000,0.100000,0.000000",
                    "img_b,0,0.850000,0.700000,0.700000,0.100000,0.100000,0.000000",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        baseline_predictions.write_text(
            json.dumps(
                [
                    {"image_id": "img_a", "class": 0, "confidence": 0.90, "bbox": [0.2, 0.2, 0.10, 0.10]},
                    {"image_id": "img_b", "class": 0, "confidence": 0.80, "bbox": [0.7, 0.7, 0.10, 0.10]},
                ]
            ),
            encoding="utf-8",
        )
        candidate_predictions.write_text(
            json.dumps(
                [
                    {"image_id": "img_a", "class": 0, "confidence": 0.60, "bbox": [0.2, 0.2, 0.10, 0.10]},
                ]
            ),
            encoding="utf-8",
        )

        subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "tools" / "yolo_hard_background.py"),
                "evaluate",
                "--candidates",
                str(candidates),
                "--predictions",
                str(candidate_predictions),
                "--baseline-predictions",
                str(baseline_predictions),
                "--out",
                str(metrics_out),
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )

        report = json.loads(metrics_out.read_text(encoding="utf-8"))

        self.assertEqual(report["candidate_regions"], 2)
        self.assertEqual(report["candidate_images"], 2)
        self.assertEqual(report["false_positives"], 1)
        self.assertEqual(report["fp_per_image"], 0.5)
        self.assertEqual(report["high_conf_false_positives"], 1)
        self.assertEqual(report["baseline_false_positives"], 2)
        self.assertEqual(report["false_positive_reduction_rate"], 0.5)

    def test_evaluate_does_not_require_baseline_predictions(self) -> None:
        candidates = self.tmpdir / "candidates.csv"
        candidate_predictions = self.tmpdir / "candidate_predictions.json"
        metrics_out = self.tmpdir / "hard_background_metrics.json"

        candidates.write_text(
            "\n".join(
                [
                    "image_id,class,confidence,x,y,w,h,max_iou",
                    "img_a,0,0.900000,0.200000,0.200000,0.100000,0.100000,0.000000",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        candidate_predictions.write_text(json.dumps([]), encoding="utf-8")

        subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "tools" / "yolo_hard_background.py"),
                "evaluate",
                "--candidates",
                str(candidates),
                "--predictions",
                str(candidate_predictions),
                "--out",
                str(metrics_out),
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )

        report = json.loads(metrics_out.read_text(encoding="utf-8"))

        self.assertEqual(report["candidate_regions"], 1)
        self.assertEqual(report["false_positives"], 0)
        self.assertNotIn("baseline_false_positives", report)


if __name__ == "__main__":
    unittest.main()
