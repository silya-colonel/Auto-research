from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
import json
from pathlib import Path

from tools.idea_diag_common import Box
from tools.osd_leave_one_class import (
    evaluate_unknown_proposals,
    filter_train_labels,
    select_candidate_unknown_classes,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class OSDLeaveOneClassTests(unittest.TestCase):
    def _write_tiny_dataset(self, root: Path) -> Path:
        for split in ("train", "val"):
            (root / "images" / split).mkdir(parents=True)
            (root / "labels" / split).mkdir(parents=True)
            (root / "images" / split / "sample.jpg").write_bytes(b"fake")
        (root / "labels" / "train" / "sample.txt").write_text(
            "0 0.5 0.5 0.2 0.2\n2 0.5 0.5 0.2 0.2\n",
            encoding="utf-8",
        )
        (root / "labels" / "val" / "sample.txt").write_text(
            "2 0.5 0.5 0.2 0.2\n",
            encoding="utf-8",
        )
        data_yaml = root / "data.yaml"
        data_yaml.write_text("path: .\ntrain: images/train\nval: images/val\nnames: ['a', 'b', 'c']\n", encoding="utf-8")
        return data_yaml

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

    def test_cli_build_split_filters_train_unknowns_and_preserves_val_unknowns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_yaml = self._write_tiny_dataset(root / "dataset")
            out_dir = root / "osd_split"

            subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "tools" / "osd_leave_one_class.py"),
                    "--mode",
                    "build-split",
                    "--data-yaml",
                    str(data_yaml),
                    "--unknown-class",
                    "2",
                    "--out-dir",
                    str(out_dir),
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=True,
            )

            train_label = (out_dir / "labels" / "train" / "sample.txt").read_text(encoding="utf-8")
            val_label = (out_dir / "labels" / "val" / "sample.txt").read_text(encoding="utf-8")
            summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))

            self.assertNotIn("2 0.5", train_label)
            self.assertIn("2 0.5", val_label)
            self.assertEqual(summary["removed_unknown_labels"], 1)

    def test_cli_evaluate_reads_predictions_json_and_recalls_unknowns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_yaml = self._write_tiny_dataset(root / "dataset")
            out_dir = root / "osd_eval"
            predictions_json = root / "predictions.json"
            predictions_json.write_text(
                json.dumps([
                    {"image_id": "sample", "cls": 0, "conf": 0.9, "box": [0.4, 0.4, 0.6, 0.6]},
                ]),
                encoding="utf-8",
            )

            subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "tools" / "osd_leave_one_class.py"),
                    "--mode",
                    "evaluate",
                    "--data-yaml",
                    str(data_yaml),
                    "--unknown-class",
                    "2",
                    "--out-dir",
                    str(out_dir),
                    "--predictions-json",
                    str(predictions_json),
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=True,
            )

            summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))

            self.assertEqual(summary["unknown_gt"], 1)
            self.assertEqual(summary["unknown_recalled"], 1)
            self.assertEqual(summary["unknown_recall"], 1.0)
            self.assertTrue(summary["gate"]["passes_gate"])


if __name__ == "__main__":
    unittest.main()
