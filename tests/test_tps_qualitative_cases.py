from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TPSQualitativeCasesCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp())
        self.labels = self.tmpdir / "labels"
        self.labels.mkdir()
        (self.labels / "img_a.txt").write_text("0 0.50 0.50 0.10 0.10\n", encoding="utf-8")
        (self.labels / "img_b.txt").write_text("0 0.70 0.70 0.08 0.08\n", encoding="utf-8")
        self.baseline = self.tmpdir / "baseline.json"
        self.tps = self.tmpdir / "tps.json"
        self.out = self.tmpdir / "cases.json"
        self.baseline.write_text(
            json.dumps(
                [
                    {"image_id": "img_a", "class": 0, "confidence": 0.82, "bbox": [0.18, 0.18, 0.08, 0.08]},
                    {"image_id": "img_b", "class": 0, "confidence": 0.40, "bbox": [0.20, 0.70, 0.08, 0.08]},
                ]
            ),
            encoding="utf-8",
        )
        self.tps.write_text(
            json.dumps(
                [
                    {"image_id": "img_a", "class": 0, "confidence": 0.91, "bbox": [0.50, 0.50, 0.10, 0.10]}
                ]
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_categorizes_improvements_suppressed_fps_and_remaining_failures(self) -> None:
        subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "tools" / "tps_qualitative_cases.py"),
                "--labels",
                str(self.labels),
                "--baseline",
                str(self.baseline),
                "--tps",
                str(self.tps),
                "--out",
                str(self.out),
            ],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=True,
        )

        payload = json.loads(self.out.read_text(encoding="utf-8"))

        self.assertEqual(payload["summary"]["improved_detections"], 1)
        self.assertEqual(payload["summary"]["suppressed_false_positives"], 2)
        self.assertEqual(payload["summary"]["remaining_misses"], 1)
        self.assertEqual(payload["cases"]["improved_detections"][0]["image_id"], "img_a")
        self.assertEqual(payload["cases"]["remaining_misses"][0]["image_id"], "img_b")
        self.assertEqual(payload["cases"]["suppressed_false_positives"][0]["case_type"], "suppressed_false_positive")


if __name__ == "__main__":
    unittest.main()
