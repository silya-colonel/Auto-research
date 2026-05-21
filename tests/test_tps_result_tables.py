from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TPSResultTablesCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp())
        self.runs = self.tmpdir / "runs" / "yolo"
        self.extra_metrics = self.tmpdir / "metrics"
        self.out = self.tmpdir / "tables"
        self._write_run_metrics(
            "baseline_yolo11n_welding_640_e100_s42",
            {
                "mAP50": 0.7812,
                "mAP50-95": 0.4211,
                "precision": 0.742,
                "recall": 0.695,
                "params_m": 2.6,
                "flops_g": 6.5,
            },
        )
        self._write_run_metrics(
            "tps_yolo11n_welding_640_e100_s42",
            {
                "mAP50": 0.8234,
                "mAP50-95": 0.4678,
                "precision": 0.802,
                "recall": 0.724,
                "fps": 118.4,
            },
        )
        self._write_json(
            self.extra_metrics / "baseline_yolo11n_welding_640_e100_s42_area_metrics.json",
            {
                "metrics": {
                    "AP_tiny": 0.31,
                    "AP75_tiny": 0.18,
                    "Recall_tiny": 0.44,
                    "AP_small": 0.52,
                    "AP75_small": 0.39,
                    "Recall_small": 0.61,
                }
            },
        )
        self._write_json(
            self.extra_metrics / "tps_yolo11n_welding_640_e100_s42_area_metrics.json",
            {
                "metrics": {
                    "AP_tiny": 0.43,
                    "AP75_tiny": 0.29,
                    "Recall_tiny": 0.57,
                    "AP_small": 0.58,
                    "AP75_small": 0.46,
                    "Recall_small": 0.68,
                }
            },
        )
        self._write_json(
            self.extra_metrics / "tps_yolo11n_welding_640_e100_s42_hard_bg.json",
            {
                "fp_per_image": 0.27,
                "high_conf_fp_per_image": 0.08,
                "false_positives": 7,
                "false_positive_reduction_rate": 0.4615,
            },
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _write_run_metrics(self, run_id: str, metrics: dict) -> None:
        self._write_json(
            self.runs / run_id / "metrics.json",
            {
                "task_name": run_id,
                "timestamp": "2026-05-21T12:00:00",
                "metrics": metrics,
            },
        )

    def test_generates_paper_ready_markdown_and_csv_tables(self) -> None:
        subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "tools" / "tps_result_tables.py"),
                "--runs",
                str(self.runs),
                "--extra-metrics",
                str(self.extra_metrics),
                "--out",
                str(self.out),
            ],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=True,
        )

        main_md = (self.out / "main_comparison.md").read_text(encoding="utf-8")
        tiny_md = (self.out / "tiny_object.md").read_text(encoding="utf-8")
        hard_md = (self.out / "hard_background.md").read_text(encoding="utf-8")
        efficiency_csv = (self.out / "efficiency.csv").read_text(encoding="utf-8")

        self.assertIn("baseline_yolo11n_welding_640_e100_s42", main_md)
        self.assertIn("tps_yolo11n_welding_640_e100_s42", main_md)
        self.assertIn("| mAP50 | mAP50-95 | Precision | Recall |", main_md)
        self.assertIn("| 0.4300 | 0.2900 | 0.5700 |", tiny_md)
        self.assertIn("| baseline_yolo11n_welding_640_e100_s42 | NA | NA | NA | NA |", hard_md)
        self.assertIn("tps_yolo11n_welding_640_e100_s42,NA,NA,118.4000,NA", efficiency_csv)

        expected_files = {
            "main_comparison.md",
            "main_comparison.csv",
            "ablation.md",
            "ablation.csv",
            "tiny_object.md",
            "tiny_object.csv",
            "hard_background.md",
            "hard_background.csv",
            "efficiency.md",
            "efficiency.csv",
        }
        self.assertTrue(expected_files.issubset({path.name for path in self.out.iterdir()}))


if __name__ == "__main__":
    unittest.main()
