from __future__ import annotations

import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class SAHBLossTests(unittest.TestCase):
    def test_scale_aware_weights_prioritize_tiny_boxes(self) -> None:
        from tools.yolo_custom_modules import scale_aware_box_weight

        target_bboxes = torch.tensor(
            [
                [[0.0, 0.0, 4.0, 4.0], [0.0, 0.0, 80.0, 80.0]],
            ],
            dtype=torch.float32,
        )
        fg_mask = torch.tensor([[True, True]])
        base_weight = torch.ones(2, 1)

        weights = scale_aware_box_weight(
            target_bboxes=target_bboxes,
            fg_mask=fg_mask,
            base_weight=base_weight,
            imgsz=torch.tensor([640.0, 640.0]),
            scale_weight=2.0,
        )

        self.assertGreater(float(weights[0]), float(weights[1]))
        self.assertGreaterEqual(float(weights[0]), 1.0)
        self.assertAlmostEqual(float(weights[1]), 1.0, places=5)

    def test_train_entry_consumes_sahb_extra_options_before_ultralytics_train(self) -> None:
        import train_yolo

        captured: dict[str, object] = {}
        tmpdir_obj = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir_obj.cleanup)
        tmpdir = Path(tmpdir_obj.name)

        class FakeYOLO:
            def __init__(self, model: str):
                captured["model"] = model

            def train(self, **kwargs: object) -> object:
                captured["train_kwargs"] = kwargs
                return Namespace(save_dir=tmpdir / "runs" / "fake", box=Namespace(map50=0.0))

            def val(self, **kwargs: object) -> object:
                box = Namespace(map50=0.0, map=0.0, mp=0.0, mr=0.0, class_result=lambda _cls: (0.0, 0.0, 0.0, 0.0))
                return Namespace(box=box, names={})

        args = Namespace(
            task_name="fake",
            data_yaml="data/welding-defect-detection-yolo/data.yaml",
            model="yolo11n.pt",
            pretrained_weights=None,
            imgsz=640,
            epochs=1,
            batch="-1",
            device=None,
            workers=0,
            seed=42,
            runs_dir=str(tmpdir / "runs"),
            resume=False,
            max_retries=0,
            enable_clearml=False,
            disable_clearml=True,
            clearml_remote=False,
            clearml_queue="gpu-any",
            clearml_project="yolo-welding-defect",
            enable_mlflow=False,
            extra=["custom_iou_loss=sahb", "sahb_scale_weight=1.25", "sahb_hard_bg_weight=0.5"],
        )

        with patch.object(train_yolo, "load_yolo", return_value=FakeYOLO):
            with patch.object(train_yolo, "configure_ultralytics_integrations"):
                with patch("tools.yolo_custom_modules.register_yolo_modules"):
                    with patch("tools.yolo_custom_modules.patch_detection_cls_loss") as patch_cls:
                        with patch("tools.yolo_custom_modules.patch_detection_iou_loss") as patch_iou:
                            train_yolo.cmd_train(args)

        patch_iou.assert_called_once_with("sahb", scale_weight=1.25, hard_bg_weight=0.5)
        patch_cls.assert_called_once_with("focal", gamma=2.0, alpha=0.25)
        self.assertNotIn("sahb_scale_weight", captured["train_kwargs"])
        self.assertNotIn("sahb_hard_bg_weight", captured["train_kwargs"])


if __name__ == "__main__":
    unittest.main()
