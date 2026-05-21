from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "torch is required for YOLO custom module tensor tests")
class FGDCCustomModuleTests(unittest.TestCase):
    def test_fgdc_preserves_feature_shape_and_allows_gradients(self) -> None:
        import torch

        from tools.yolo_custom_modules import FGDC

        module = FGDC(16, reduction=4)
        x = torch.randn(2, 16, 12, 10, requires_grad=True)
        y = module(x)
        y.mean().backward()

        self.assertEqual(tuple(y.shape), tuple(x.shape))
        self.assertIsNotNone(x.grad)


class FGDCConfigTests(unittest.TestCase):
    def test_fgdc_config_declares_three_detect_feature_blocks(self) -> None:
        config = (PROJECT_ROOT / "configs" / "tps_yolo11n_fgdc.yaml").read_text(encoding="utf-8")
        fgdc_lines = [line for line in config.splitlines() if line.strip().startswith("- [-1, 1, FGDC")]

        self.assertEqual(len(fgdc_lines), 3)
        self.assertIn("[-1, 1, FGDC, [64]]", config)
        self.assertIn("[-1, 1, FGDC, [128]]", config)
        self.assertIn("[-1, 1, FGDC, [256]]", config)
        self.assertIn("[[17, 21, 25], 1, Detect, [nc]]", config)


if __name__ == "__main__":
    unittest.main()
