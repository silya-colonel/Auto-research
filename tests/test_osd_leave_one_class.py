from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

from tools.idea_diag_common import Box
from tools.osd_leave_one_class import (
    evaluate_unknown_proposals,
    filter_train_labels,
    select_candidate_unknown_classes,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class OSDLeaveOneClassTests(unittest.TestCase):
    def test_filter_train_labels_removes_unknown_classes(self) -> None:
        rows = ["0 0.5 0.5 0.1 0.1", "2 0.5 0.5 0.1 0.1", "4 0.5 0.5 0.1 0.1"]

        filtered = filter_train_labels(rows, unknown_classes={2})

        self.assertEqual(filtered, ["0 0.5 0.5 0.1 0.1", "4 0.5 0.5 0.1 0.1"])

    def test_select_candidate_unknown_classes_requires_enough_validation_boxes(self) -> None:
        counts = {0: {"train": 100, "val": 40}, 1: {"train": 80, "val": 3}, 2: {"train": 60, "val": 20}}

        selected = select_candidate_unknown_classes(counts, min_val_boxes=10)

        self.assertEqual(selected, [0, 2])

    def test_evaluate_unknown_proposals_is_class_agnostic(self) -> None:
        labels = [
            {"image_id": "im1", "gt_index": 0, "cls": 9, "box": Box(0.1, 0.1, 0.3, 0.3)},
            {"image_id": "im2", "gt_index": 0, "cls": 1, "box": Box(0.1, 0.1, 0.3, 0.3)},
        ]
        predictions = [
            {"image_id": "im1", "cls": 1, "conf": 0.8, "box": Box(0.1, 0.1, 0.3, 0.3)},
            {"image_id": "im2", "cls": 1, "conf": 0.8, "box": Box(0.1, 0.1, 0.3, 0.3)},
        ]

        summary = evaluate_unknown_proposals(predictions, labels, unknown_classes={9}, iou_threshold=0.5)

        self.assertEqual(summary["unknown_gt"], 1)
        self.assertEqual(summary["unknown_recalled"], 1)
        self.assertEqual(summary["unknown_recall"], 1.0)

    def test_cli_help_lists_protocol_modes(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "tools" / "osd_leave_one_class.py"),
                "--help",
            ],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=True,
        )

        self.assertIn("--data-yaml", result.stdout)
        self.assertIn("--unknown-class", result.stdout)
        self.assertIn("--output-root", result.stdout)
        self.assertIn("--mode", result.stdout)
        self.assertIn("build-split", result.stdout)
        self.assertIn("evaluate", result.stdout)


if __name__ == "__main__":
    unittest.main()
