from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class YoloSplitDatasetCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp())
        self.dataset = self.tmpdir / "dataset"
        self.output = self.tmpdir / "split"
        self.second_output = self.tmpdir / "split_again"
        (self.dataset / "images" / "train").mkdir(parents=True)
        (self.dataset / "labels" / "train").mkdir(parents=True)
        (self.dataset / "classes.txt").write_text("defect\n", encoding="utf-8")
        (self.dataset / "data.yaml").write_text(
            'path: .\ntrain: images/train\nval: images/val\nnc: 1\nnames: ["defect"]\n',
            encoding="utf-8",
        )
        for idx in range(10):
            stem = f"img_{idx:02d}"
            (self.dataset / "images" / "train" / f"{stem}.jpg").write_bytes(b"fake-image")
            (self.dataset / "labels" / "train" / f"{stem}.txt").write_text(
                "0 0.5 0.5 0.1 0.1\n",
                encoding="utf-8",
            )

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_splits_yolo_dataset_deterministically_and_preserves_label_pairs(self) -> None:
        command = [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "yolo_split_dataset.py"),
            str(self.dataset),
            "--out",
            str(self.output),
            "--train",
            "0.70",
            "--val",
            "0.15",
            "--test",
            "0.15",
            "--seed",
            "42",
        ]
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)

        rerun = command.copy()
        rerun[rerun.index(str(self.output))] = str(self.second_output)
        subprocess.run(rerun, cwd=PROJECT_ROOT, check=True)

        self.assertEqual(len(list((self.output / "images" / "train").glob("*.jpg"))), 7)
        self.assertEqual(len(list((self.output / "images" / "val").glob("*.jpg"))), 1)
        self.assertEqual(len(list((self.output / "images" / "test").glob("*.jpg"))), 2)
        for split in ("train", "val", "test"):
            image_stems = sorted(path.stem for path in (self.output / "images" / split).glob("*.jpg"))
            label_stems = sorted(path.stem for path in (self.output / "labels" / split).glob("*.txt"))
            self.assertEqual(image_stems, label_stems)
            second_stems = sorted(path.stem for path in (self.second_output / "images" / split).glob("*.jpg"))
            self.assertEqual(image_stems, second_stems)

        data_yaml = (self.output / "data.yaml").read_text(encoding="utf-8")
        self.assertIn("train: images/train", data_yaml)
        self.assertIn("val: images/val", data_yaml)
        self.assertIn("test: images/test", data_yaml)
        self.assertIn('names: ["defect"]', data_yaml)

    def test_rejects_ratios_that_do_not_sum_to_one(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "tools" / "yolo_split_dataset.py"),
                str(self.dataset),
                "--out",
                str(self.output),
                "--train",
                "0.50",
                "--val",
                "0.30",
                "--test",
                "0.30",
            ],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must sum to 1.0", result.stderr)


if __name__ == "__main__":
    unittest.main()
