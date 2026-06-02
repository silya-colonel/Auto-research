from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.idea_diag_common import (
    Box,
    LabelKey,
    class_agnostic_best_iou,
    finish_clearml_task,
    flatten_numeric_metrics,
    gate_passes,
    label_key,
    label_path_for_image,
    load_yolo_labels,
    write_json,
    yolo_line_to_box,
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

    def test_yolo_line_to_box_converts_center_width_height_to_xyxy(self) -> None:
        parsed = yolo_line_to_box("2 0.50 0.50 0.20 0.40")

        self.assertIsNotNone(parsed)
        cls, box = parsed
        self.assertEqual(cls, 2)
        self.assertEqual(box, Box(0.4, 0.3, 0.6, 0.7))

    def test_load_yolo_labels_resolves_standard_images_labels_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "images" / "val"
            label_dir = root / "labels" / "val"
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            image_path = image_dir / "sample.jpg"
            image_path.write_bytes(b"not-a-real-image")
            (label_dir / "sample.txt").write_text("1 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            data_yaml = root / "data.yaml"
            data_yaml.write_text("path: .\nval: images/val\nnames: ['a', 'b']\n", encoding="utf-8")

            labels = load_yolo_labels(data_yaml, "val")

            self.assertEqual(len(labels), 1)
            self.assertEqual(labels[0]["image_id"], "sample")
            self.assertEqual(labels[0]["cls"], 1)
            self.assertEqual(labels[0]["box"], Box(0.4, 0.4, 0.6, 0.6))
            self.assertEqual(label_path_for_image(image_path, data_yaml).resolve(), (label_dir / "sample.txt").resolve())

    def test_flatten_numeric_metrics_recurses_and_converts_bools(self) -> None:
        metrics = flatten_numeric_metrics({"records": 3, "gate": {"passes_gate": True}, "text": "x"})

        self.assertEqual(metrics["records"], 3.0)
        self.assertEqual(metrics["gate.passes_gate"], 1.0)
        self.assertNotIn("text", metrics)

    def test_finish_clearml_task_reports_scalars_and_uploads_artifacts(self) -> None:
        class FakeLogger:
            def __init__(self) -> None:
                self.scalars: list[tuple[str, str, float, int]] = []

            def report_scalar(self, title: str, series: str, value: float, iteration: int) -> None:
                self.scalars.append((title, series, value, iteration))

        class FakeTask:
            def __init__(self) -> None:
                self.logger = FakeLogger()
                self.artifacts: dict[str, str] = {}
                self.closed = False

            def get_logger(self) -> FakeLogger:
                return self.logger

            def upload_artifact(self, name: str, artifact_object: str) -> None:
                self.artifacts[name] = artifact_object

            def close(self) -> None:
                self.closed = True

        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "summary.json"
            artifact.write_text("{}", encoding="utf-8")
            created: list[FakeTask] = []

            def factory(**_kwargs: object) -> FakeTask:
                task = FakeTask()
                created.append(task)
                return task

            finish_clearml_task(
                enabled=True,
                project_name="project",
                task_name="task",
                summary={"records": 1, "gate": {"passes_gate": True}},
                artifacts={"summary": artifact, "missing": Path(tmp) / "missing.json"},
                task_factory=factory,
            )

            self.assertEqual(len(created), 1)
            self.assertIn(("summary", "records", 1.0, 0), created[0].logger.scalars)
            self.assertIn(("summary", "gate.passes_gate", 1.0, 0), created[0].logger.scalars)
            self.assertEqual(created[0].artifacts["summary"], str(artifact))
            self.assertTrue(created[0].closed)

    def test_finish_clearml_task_disabled_does_not_create_task(self) -> None:
        called = False

        def factory(**_kwargs: object) -> object:
            nonlocal called
            called = True
            return object()

        finish_clearml_task(
            enabled=False,
            project_name="project",
            task_name="task",
            summary={},
            artifacts={},
            task_factory=factory,
        )

        self.assertFalse(called)


if __name__ == "__main__":
    unittest.main()
