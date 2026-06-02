from __future__ import annotations

import subprocess
import sys
import tempfile
import json
import unittest
from pathlib import Path

from tools.clr_label_reliability import (
    ReliabilityConfig,
    score_label_reliability,
    summarize_reliability,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class CLRLabelReliabilityTests(unittest.TestCase):
    def _write_tiny_dataset(self, root: Path) -> Path:
        image_dir = root / "images" / "val"
        label_dir = root / "labels" / "val"
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        (image_dir / "sample.jpg").write_bytes(b"fake")
        (label_dir / "sample.txt").write_text("1 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        data_yaml = root / "data.yaml"
        data_yaml.write_text("path: .\nval: images/val\nnames: ['a', 'b']\n", encoding="utf-8")
        return data_yaml

    def test_stable_high_iou_label_scores_high(self) -> None:
        row = score_label_reliability(
            [
                {"matched": True, "same_class_iou": 0.82, "pred_cls": 1, "conf": 0.80},
                {"matched": True, "same_class_iou": 0.78, "pred_cls": 1, "conf": 0.76},
                {"matched": True, "same_class_iou": 0.84, "pred_cls": 1, "conf": 0.79},
            ],
            ReliabilityConfig(),
        )

        self.assertGreaterEqual(row["reliability"], 0.75)
        self.assertEqual(row["bucket"], "high")

    def test_unstable_low_iou_label_scores_low(self) -> None:
        row = score_label_reliability(
            [
                {"matched": False, "same_class_iou": 0.05, "pred_cls": None, "conf": 0.0},
                {"matched": True, "same_class_iou": 0.31, "pred_cls": 2, "conf": 0.42},
                {"matched": False, "same_class_iou": 0.08, "pred_cls": None, "conf": 0.0},
            ],
            ReliabilityConfig(),
        )

        self.assertLess(row["reliability"], 0.40)
        self.assertEqual(row["bucket"], "low")

    def test_summary_gate_passes_when_low_reliability_explains_errors(self) -> None:
        summary = summarize_reliability(
            [
                {"bucket": "low", "baseline_error": True},
                {"bucket": "low", "baseline_error": True},
                {"bucket": "high", "baseline_error": False},
                {"bucket": "high", "baseline_error": False},
            ],
            min_records=4,
            min_error_explained=0.40,
        )

        self.assertTrue(summary["gate"]["passes_gate"])
        self.assertEqual(summary["low_reliability_error_fraction"], 1.0)

    def test_cli_help_lists_diagnostic_inputs(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "tools" / "clr_label_reliability.py"),
                "--help",
            ],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=True,
        )

        self.assertIn("--weights", result.stdout)
        self.assertIn("--data-yaml", result.stdout)
        self.assertIn("--out-dir", result.stdout)
        self.assertIn("--views", result.stdout)
        self.assertIn("--min-error-explained", result.stdout)

    def test_cli_uses_predictions_json_to_write_nonempty_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_yaml = self._write_tiny_dataset(root / "dataset")
            predictions_json = root / "predictions.json"
            predictions_json.write_text(
                json.dumps([
                    {"image_id": "sample", "cls": 1, "conf": 0.9, "box": [0.4, 0.4, 0.6, 0.6]},
                ]),
                encoding="utf-8",
            )
            out_dir = root / "clr"

            subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "tools" / "clr_label_reliability.py"),
                    "--data-yaml",
                    str(data_yaml),
                    "--out-dir",
                    str(out_dir),
                    "--predictions-json",
                    str(predictions_json),
                    "--min-records",
                    "1",
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=True,
            )

            records = json.loads((out_dir / "reliability_records.json").read_text(encoding="utf-8"))
            summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["bucket"], "high")
            self.assertEqual(summary["records"], 1)


if __name__ == "__main__":
    unittest.main()
