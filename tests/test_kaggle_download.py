#!/usr/bin/env python3
"""Unit tests for tools/kaggle_download.py."""

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import kaggle_download


class TestKaggleHubDownload(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_uses_output_dir_not_dataset_internal_path(self):
        calls = []

        def dataset_download(dataset, output_dir=None):
            calls.append((dataset, output_dir))
            return output_dir

        fake = SimpleNamespace(dataset_download=dataset_download)
        out_dir = self.tmpdir / "dataset"
        with patch.dict(sys.modules, {"kagglehub": fake}):
            result = kaggle_download.download_kagglehub("owner/name", out_dir)

        self.assertEqual(result, out_dir)
        self.assertEqual(calls, [("owner/name", str(out_dir))])

    def test_copies_cache_download_when_output_dir_is_unsupported(self):
        cache_dir = self.tmpdir / "cache"
        cache_dir.mkdir()
        (cache_dir / "sample.txt").write_text("ok", encoding="utf-8")

        def dataset_download(dataset):
            return str(cache_dir)

        fake = SimpleNamespace(dataset_download=dataset_download)
        out_dir = self.tmpdir / "dataset"
        with patch.dict(sys.modules, {"kagglehub": fake}):
            kaggle_download.download_kagglehub("owner/name", out_dir)

        self.assertEqual((out_dir / "sample.txt").read_text(encoding="utf-8"), "ok")


class TestCocoConversion(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_remaps_sparse_category_ids_to_contiguous_yolo_ids(self):
        for name in ("a.jpg", "b.jpg"):
            (self.tmpdir / name).write_bytes(b"image")

        coco = {
            "images": [
                {"id": 10, "file_name": "a.jpg", "width": 100, "height": 100},
                {"id": 20, "file_name": "b.jpg", "width": 100, "height": 100},
            ],
            "categories": [
                {"id": 3, "name": "scratch"},
                {"id": 7, "name": "dent"},
            ],
            "annotations": [
                {"image_id": 10, "category_id": 3, "bbox": [0, 0, 10, 10]},
                {"image_id": 20, "category_id": 7, "bbox": [10, 10, 20, 20]},
            ],
        }
        (self.tmpdir / "annotations.json").write_text(json.dumps(coco), encoding="utf-8")

        class_names = kaggle_download.convert_coco_to_yolo(self.tmpdir)

        self.assertEqual(class_names, {0: "scratch", 1: "dent"})
        self.assertTrue((self.tmpdir / "labels" / "a.txt").read_text().startswith("0 "))
        self.assertTrue((self.tmpdir / "labels" / "b.txt").read_text().startswith("1 "))


class TestDataYaml(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        (self.tmpdir / "images").mkdir()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_flat_images_use_existing_images_dir_for_train_and_val(self):
        yaml_path = kaggle_download.generate_data_yaml(self.tmpdir, {0: "defect"})
        text = yaml_path.read_text(encoding="utf-8")

        self.assertIn("train: images\n", text)
        self.assertIn("val: images\n", text)

    def test_setup_auth_does_not_require_dataset(self):
        with patch.object(sys, "argv", ["kaggle_download.py", "--setup-auth"]):
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                kaggle_download.main()

        self.assertIn("Kaggle API", stdout.getvalue())


class TestDatasetInput(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_parses_plain_dataset_ref(self):
        dataset = kaggle_download.parse_dataset_input("aadityadeth/yolov5-weld")

        self.assertEqual(dataset, "aadityadeth/yolov5-weld")

    def test_parses_kagglehub_snippet(self):
        snippet = '''
import kagglehub
path = kagglehub.dataset_download("aadityadeth/yolov5-weld")
print("Path to dataset files:", path)
'''
        dataset = kaggle_download.parse_dataset_input(snippet)

        self.assertEqual(dataset, "aadityadeth/yolov5-weld")

    def test_parses_dataset_ref_from_file(self):
        snippet_path = self.tmpdir / "download.py"
        snippet_path.write_text(
            'path = kagglehub.dataset_download("aadityadeth/yolov5-weld")\n',
            encoding="utf-8",
        )

        dataset = kaggle_download.parse_dataset_input(str(snippet_path))

        self.assertEqual(dataset, "aadityadeth/yolov5-weld")

    def test_parses_kaggle_url(self):
        dataset = kaggle_download.parse_dataset_input(
            "https://www.kaggle.com/datasets/aadityadeth/yolov5-weld"
        )

        self.assertEqual(dataset, "aadityadeth/yolov5-weld")

    def test_missing_dataset_prompts_before_list_mode(self):
        with patch.object(sys, "argv", ["kaggle_download.py", "--list"]):
            with patch.object(kaggle_download, "prompt_dataset_input") as prompt:
                with patch.object(kaggle_download, "_has_kagglehub", return_value=False):
                    with patch.object(kaggle_download, "list_files_kaggle_cli") as list_files:
                        prompt.return_value = 'kagglehub.dataset_download("aadityadeth/yolov5-weld")'
                        kaggle_download.main()

        list_files.assert_called_once_with("aadityadeth/yolov5-weld")

    def test_list_files_kaggle_cli_handles_missing_command(self):
        with patch.object(kaggle_download.shutil, "which", return_value=None):
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                kaggle_download.list_files_kaggle_cli("aadityadeth/yolov5-weld")

        self.assertIn("未找到 kaggle CLI", stdout.getvalue())

    def test_prompt_accepts_multiline_kagglehub_snippet(self):
        lines = [
            "import kagglehub",
            'path = kagglehub.dataset_download("aadityadeth/yolov5-weld")',
            'print("Path to dataset files:", path)',
            "",
        ]
        with patch("builtins.input", side_effect=lines):
            with contextlib.redirect_stdout(io.StringIO()):
                text = kaggle_download.prompt_dataset_input()

        self.assertEqual(kaggle_download.parse_dataset_input(text), "aadityadeth/yolov5-weld")


if __name__ == "__main__":
    unittest.main()
