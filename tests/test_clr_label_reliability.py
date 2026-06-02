from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

from tools.clr_label_reliability import (
    ReliabilityConfig,
    score_label_reliability,
    summarize_reliability,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class CLRLabelReliabilityTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
