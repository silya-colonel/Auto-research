from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tools.idea_diag_common import Box
from tools.fasd_teacher_audit import (
    MaskQualityConfig,
    edge_proxy_mask,
    score_mask_quality,
    summarize_teacher_quality,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FASDTeacherAuditTests(unittest.TestCase):
    def test_edge_proxy_mask_finds_boundary_inside_gt_box(self) -> None:
        image = Image.new("L", (32, 32), 20)
        for x in range(8, 24):
            for y in range(8, 24):
                image.putpixel((x, y), 230)

        mask = edge_proxy_mask(image.convert("RGB"), Box(0.0, 0.0, 1.0, 1.0), threshold=30)

        self.assertEqual(len(mask), 32)
        self.assertEqual(len(mask[0]), 32)
        self.assertGreater(sum(sum(row) for row in mask), 0)
        self.assertEqual(mask[16][16], 0)

    def test_score_mask_quality_rejects_empty_mask(self) -> None:
        mask = [[0 for _x in range(12)] for _y in range(12)]

        row = score_mask_quality(mask, MaskQualityConfig(min_coverage=0.02))

        self.assertEqual(row["bucket"], "unusable")
        self.assertFalse(row["usable"])
        self.assertEqual(row["coverage"], 0.0)

    def test_score_mask_quality_accepts_sparse_boundary_mask(self) -> None:
        mask = [[0 for _x in range(12)] for _y in range(12)]
        for i in range(12):
            mask[0][i] = 1
            mask[11][i] = 1
            mask[i][0] = 1
            mask[i][11] = 1

        row = score_mask_quality(mask, MaskQualityConfig(min_coverage=0.05, max_coverage=0.60))

        self.assertEqual(row["bucket"], "usable")
        self.assertTrue(row["usable"])
        self.assertGreater(row["coverage"], 0.05)

    def test_summary_gate_uses_usable_rate(self) -> None:
        summary = summarize_teacher_quality(
            [
                {"usable": True, "quality": 0.60},
                {"usable": True, "quality": 0.40},
                {"usable": False, "quality": 0.05},
            ],
            min_records=3,
            min_usable_rate=0.50,
        )

        self.assertTrue(summary["gate"]["passes_gate"])
        self.assertAlmostEqual(summary["usable_rate"], 2 / 3)

    def test_cli_writes_stable_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "fasd"
            data_yaml = Path(tmp) / "data.yaml"
            data_yaml.write_text("path: .\n", encoding="utf-8")

            subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "tools" / "fasd_teacher_audit.py"),
                    "--data-yaml",
                    str(data_yaml),
                    "--out-dir",
                    str(out_dir),
                    "--provider",
                    "edge_proxy",
                    "--max-samples",
                    "0",
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=True,
            )

            self.assertTrue((out_dir / "summary.json").exists())
            self.assertTrue((out_dir / "teacher_quality_records.json").exists())
            self.assertTrue((out_dir / "report.md").exists())

    def test_cli_help_lists_diagnostic_inputs(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "tools" / "fasd_teacher_audit.py"),
                "--help",
            ],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=True,
        )

        self.assertIn("--data-yaml", result.stdout)
        self.assertIn("--split", result.stdout)
        self.assertIn("--out-dir", result.stdout)
        self.assertIn("--provider", result.stdout)
        self.assertIn("--max-samples", result.stdout)
        self.assertIn("--min-usable-rate", result.stdout)


if __name__ == "__main__":
    unittest.main()
