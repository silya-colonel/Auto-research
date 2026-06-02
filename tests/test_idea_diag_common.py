from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.idea_diag_common import (
    Box,
    LabelKey,
    class_agnostic_best_iou,
    gate_passes,
    label_key,
    write_json,
)


class IdeaDiagCommonTests(unittest.TestCase):
    def test_label_key_is_stable_across_extra_fields(self) -> None:
        label = {"image_id": "img_001", "gt_index": 7, "cls": 3, "ignored": "x"}

        self.assertEqual(label_key(label), LabelKey("img_001", 7, 3))

    def test_class_agnostic_best_iou_ignores_prediction_class(self) -> None:
        label = {"image_id": "img_001", "gt_index": 0, "cls": 2, "box": Box(0.1, 0.1, 0.3, 0.3)}
        predictions = [
            {"image_id": "img_001", "cls": 8, "conf": 0.7, "box": Box(0.1, 0.1, 0.3, 0.3)},
            {"image_id": "img_001", "cls": 2, "conf": 0.9, "box": Box(0.6, 0.6, 0.8, 0.8)},
        ]

        best = class_agnostic_best_iou(label, predictions)

        self.assertEqual(best["cls"], 8)
        self.assertAlmostEqual(best["iou"], 1.0)

    def test_gate_passes_requires_every_check_to_pass(self) -> None:
        checks = {
            "enough_records": {"passed": True},
            "not_too_noisy": {"passed": False},
        }

        self.assertFalse(gate_passes(checks))

    def test_write_json_creates_parent_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "summary.json"
            write_json(path, {"ok": True})

            self.assertEqual(path.read_text(encoding="utf-8").strip(), '{\n  "ok": true\n}')


if __name__ == "__main__":
    unittest.main()
