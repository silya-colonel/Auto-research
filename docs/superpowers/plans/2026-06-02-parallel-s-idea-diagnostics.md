# Parallel S-Idea Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a test-first diagnostic harness for three new steel-defect research ideas: CLR-YOLO, OSD-YOLO, and FASD-YOLO.

**Architecture:** The first phase is diagnosis, not full training. Each idea gets a small, independently runnable CLI that outputs `summary.json`, `records.json`, and `report.md`; a shared ranking tool then decides which idea deserves GPU training. Experiment matrices and SSH submission support are added only after each CLI has behavior tests.

**Tech Stack:** Python 3.10+, pytest/unittest, Ultralytics YOLO via existing project helpers, YAML experiment matrices, existing `tools/ssh_submit.py` remote runner, ClearML-compatible artifact directories.

---

## Scope

Run these three ideas in parallel at the diagnostic level:

- **CLR-YOLO:** estimate label reliability from prediction consistency and error correlation.
- **OSD-YOLO:** build an open-set defect protocol using leave-one-class-out or leave-one-group-out splits.
- **FASD-YOLO:** audit whether foundation or proxy region teachers can provide usable boundary/attribute supervision inside GT boxes.

Do not start multi-seed training until the ranking task produces a pass/fail decision. The failed historical families remain out of scope: MCA/DSA/QDH/HPR/MRS/RST/HNC/TPS.

## File Structure

- Create `tools/idea_diag_common.py`: shared label keys, class-agnostic matching, gate summaries, JSON/Markdown writing.
- Create `tests/test_idea_diag_common.py`: behavior tests for the shared diagnostic helpers.
- Create `tools/clr_label_reliability.py`: CLR reliability scoring and report CLI.
- Create `tests/test_clr_label_reliability.py`: stable/unstable/noisy-label behavior tests.
- Create `tools/osd_leave_one_class.py`: OSD split protocol and class-agnostic unknown proposal evaluator.
- Create `tests/test_osd_leave_one_class.py`: held-out class protocol tests.
- Create `tools/fasd_teacher_audit.py`: FASD teacher-quality audit with a built-in edge/contrast proxy teacher and optional future provider hook.
- Create `tests/test_fasd_teacher_audit.py`: synthetic image mask-quality behavior tests.
- Create `tools/rank_parallel_ideas.py`: reads the three diagnostic summaries and writes the S-idea ranking report.
- Create `tests/test_rank_parallel_ideas.py`: ranking behavior tests.
- Modify `tools/experiment_matrix.py`: add run kinds `clr_diag`, `osd_diag`, `fasd_audit`, and `idea_rank`.
- Modify `tools/ssh_submit.py`: add remote commands for the new diagnostic run kinds.
- Modify `tests/test_experiment_matrix.py`: command rendering tests for new run kinds.
- Modify `tests/test_ssh_submit.py`: dry-run remote command tests for new run kinds.
- Create `research/yolo11_welding_defect/experiments/parallel_s_idea_diagnostics.yaml`: diagnostic matrix.

---

### Task 1: Shared Diagnostic Helpers

**Files:**
- Create: `tools/idea_diag_common.py`
- Test: `tests/test_idea_diag_common.py`

- [ ] **Step 1: Write the failing tests**

```python
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.mrs_sliced_teacher import Box
from tools.idea_diag_common import (
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
pytest tests/test_idea_diag_common.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'tools.idea_diag_common'`.

- [ ] **Step 3: Write the minimal implementation**

Create `tools/idea_diag_common.py`:

```python
#!/usr/bin/env python3
"""Shared helpers for steel-defect idea diagnostics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.mrs_sliced_teacher import box_iou


@dataclass(frozen=True)
class LabelKey:
    image_id: str
    gt_index: int
    cls: int


def label_key(label: dict[str, Any]) -> LabelKey:
    return LabelKey(str(label["image_id"]), int(label["gt_index"]), int(label["cls"]))


def class_agnostic_best_iou(label: dict[str, Any], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    best = {"iou": 0.0, "conf": 0.0, "cls": None, "box": None}
    for prediction in predictions:
        if str(prediction["image_id"]) != str(label["image_id"]):
            continue
        iou = box_iou(label["box"], prediction["box"])
        confidence = float(prediction.get("conf", 0.0))
        if (iou, confidence) > (float(best["iou"]), float(best["conf"])):
            best = {
                "iou": float(iou),
                "conf": confidence,
                "cls": int(prediction["cls"]),
                "box": prediction["box"],
            }
    return best


def gate_passes(checks: dict[str, dict[str, Any]]) -> bool:
    return all(bool(row.get("passed", False)) for row in checks.values())


def write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
```

- [ ] **Step 4: Run the test to verify it passes**

Run:

```bash
pytest tests/test_idea_diag_common.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/idea_diag_common.py tests/test_idea_diag_common.py
git commit -m "test: add shared idea diagnostic helpers"
```

---

### Task 2: CLR-YOLO Label Reliability Diagnostic

**Files:**
- Create: `tools/clr_label_reliability.py`
- Test: `tests/test_clr_label_reliability.py`

- [ ] **Step 1: Write the failing tests**

```python
from __future__ import annotations

import unittest

from tools.clr_label_reliability import (
    ReliabilityConfig,
    score_label_reliability,
    summarize_reliability,
)


class CLRLabelReliabilityTests(unittest.TestCase):
    def test_stable_high_iou_label_scores_high(self) -> None:
        row = score_label_reliability(
            [
                {"matched": True, "same_class_iou": 0.82, "pred_cls": 1, "conf": 0.80},
                {"matched": True, "same_class_iou": 0.78, "pred_cls": 1, "conf": 0.76},
                {"matched": True, "same_class_iou": 0.84, "pred_cls": 1, "conf": 0.79},
            ],
            ReliabilityConfig(),
        )

        self.assertGreaterEqual(row["reliability"], 0.75)
        self.assertEqual(row["bucket"], "high")

    def test_unstable_low_iou_label_scores_low(self) -> None:
        row = score_label_reliability(
            [
                {"matched": False, "same_class_iou": 0.05, "pred_cls": None, "conf": 0.0},
                {"matched": True, "same_class_iou": 0.31, "pred_cls": 2, "conf": 0.42},
                {"matched": False, "same_class_iou": 0.08, "pred_cls": None, "conf": 0.0},
            ],
            ReliabilityConfig(),
        )

        self.assertLess(row["reliability"], 0.40)
        self.assertEqual(row["bucket"], "low")

    def test_summary_gate_passes_when_low_reliability_explains_errors(self) -> None:
        summary = summarize_reliability(
            [
                {"bucket": "low", "baseline_error": True},
                {"bucket": "low", "baseline_error": True},
                {"bucket": "high", "baseline_error": False},
                {"bucket": "high", "baseline_error": False},
            ],
            min_records=4,
            min_error_explained=0.40,
        )

        self.assertTrue(summary["gate"]["passes_gate"])
        self.assertEqual(summary["low_reliability_error_fraction"], 0.5)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
pytest tests/test_clr_label_reliability.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'tools.clr_label_reliability'`.

- [ ] **Step 3: Implement reliability scoring**

Create `tools/clr_label_reliability.py`:

```python
#!/usr/bin/env python3
"""CLR-YOLO label reliability diagnostic."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any

from tools.idea_diag_common import gate_passes


@dataclass(frozen=True)
class ReliabilityConfig:
    high_threshold: float = 0.70
    low_threshold: float = 0.45


def _agreement(values: list[Any]) -> float:
    non_null = [value for value in values if value is not None]
    if not non_null:
        return 0.0
    counts = {value: non_null.count(value) for value in set(non_null)}
    return max(counts.values()) / len(values)


def score_label_reliability(view_rows: list[dict[str, Any]], config: ReliabilityConfig) -> dict[str, Any]:
    if not view_rows:
        return {"reliability": 0.0, "bucket": "low", "match_rate": 0.0, "mean_iou": 0.0, "class_agreement": 0.0}
    match_rate = mean(1.0 if row.get("matched") else 0.0 for row in view_rows)
    mean_iou = mean(float(row.get("same_class_iou", 0.0)) for row in view_rows)
    class_agreement = _agreement([row.get("pred_cls") for row in view_rows])
    reliability = 0.45 * match_rate + 0.35 * mean_iou + 0.20 * class_agreement
    if reliability >= config.high_threshold:
        bucket = "high"
    elif reliability < config.low_threshold:
        bucket = "low"
    else:
        bucket = "mid"
    return {
        "reliability": round(float(reliability), 6),
        "bucket": bucket,
        "match_rate": round(float(match_rate), 6),
        "mean_iou": round(float(mean_iou), 6),
        "class_agreement": round(float(class_agreement), 6),
    }


def summarize_reliability(
    rows: list[dict[str, Any]],
    *,
    min_records: int = 100,
    min_error_explained: float = 0.20,
) -> dict[str, Any]:
    low_rows = [row for row in rows if row.get("bucket") == "low"]
    error_rows = [row for row in rows if row.get("baseline_error")]
    low_error_rows = [row for row in low_rows if row.get("baseline_error")]
    low_reliability_error_fraction = len(low_error_rows) / max(len(error_rows), 1)
    checks = {
        "record_count_ge_min": {"value": len(rows), "threshold": min_records, "passed": len(rows) >= min_records},
        "low_reliability_explains_errors": {
            "value": round(low_reliability_error_fraction, 6),
            "threshold": min_error_explained,
            "passed": low_reliability_error_fraction >= min_error_explained,
        },
    }
    return {
        "records": len(rows),
        "low_reliability_count": len(low_rows),
        "baseline_error_count": len(error_rows),
        "low_reliability_error_fraction": round(low_reliability_error_fraction, 6),
        "gate": {"checks": checks, "passes_gate": gate_passes(checks)},
    }
```

- [ ] **Step 4: Run the tests**

Run:

```bash
pytest tests/test_clr_label_reliability.py -v
```

Expected: PASS.

- [ ] **Step 5: Add the CLI in a second TDD slice**

Add one failing CLI smoke test that runs:

```bash
python tools/clr_label_reliability.py --help
```

Expected after implementation: help text includes `--weights`, `--data-yaml`, `--out-dir`, `--views`, and `--min-error-explained`.

Implement the CLI after the failing test. Use existing helpers from `tools.mrs_sliced_teacher` for reading `data.yaml`, loading labels, collecting image sizes, and running YOLO predictions. The CLI writes:

- `summary.json`
- `reliability_records.json`
- `report.md`

- [ ] **Step 6: Commit**

```bash
git add tools/clr_label_reliability.py tests/test_clr_label_reliability.py
git commit -m "feat: add CLR label reliability diagnostic"
```

---

### Task 3: OSD-YOLO Open-Set Protocol Diagnostic

**Files:**
- Create: `tools/osd_leave_one_class.py`
- Test: `tests/test_osd_leave_one_class.py`

- [ ] **Step 1: Write the failing tests**

```python
from __future__ import annotations

import unittest

from tools.mrs_sliced_teacher import Box
from tools.osd_leave_one_class import (
    evaluate_unknown_proposals,
    filter_train_labels,
    select_candidate_unknown_classes,
)


class OSDLeaveOneClassTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
pytest tests/test_osd_leave_one_class.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'tools.osd_leave_one_class'`.

- [ ] **Step 3: Implement the protocol helpers**

Create `tools/osd_leave_one_class.py`:

```python
#!/usr/bin/env python3
"""OSD-YOLO leave-one-class open-set protocol helpers."""

from __future__ import annotations

from typing import Any

from tools.idea_diag_common import class_agnostic_best_iou


def filter_train_labels(lines: list[str], unknown_classes: set[int]) -> list[str]:
    kept: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        cls = int(float(stripped.split()[0]))
        if cls not in unknown_classes:
            kept.append(stripped)
    return kept


def select_candidate_unknown_classes(counts: dict[int, dict[str, int]], min_val_boxes: int = 10) -> list[int]:
    return [
        cls
        for cls, split_counts in sorted(counts.items())
        if int(split_counts.get("val", 0)) >= min_val_boxes and int(split_counts.get("train", 0)) > 0
    ]


def evaluate_unknown_proposals(
    predictions: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    *,
    unknown_classes: set[int],
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    unknown_labels = [label for label in labels if int(label["cls"]) in unknown_classes]
    recalled = 0
    for label in unknown_labels:
        best = class_agnostic_best_iou(label, predictions)
        if float(best["iou"]) >= iou_threshold:
            recalled += 1
    unknown_gt = len(unknown_labels)
    return {
        "unknown_gt": unknown_gt,
        "unknown_recalled": recalled,
        "unknown_recall": round(recalled / max(unknown_gt, 1), 6),
    }
```

- [ ] **Step 4: Run the tests**

Run:

```bash
pytest tests/test_osd_leave_one_class.py -v
```

Expected: PASS.

- [ ] **Step 5: Add CLI and dataset split generation in the next TDD slice**

Add CLI tests for:

```bash
python tools/osd_leave_one_class.py --help
```

Expected after implementation: help text includes `--data-yaml`, `--unknown-class`, `--output-root`, `--mode build-split`, and `--mode evaluate`.

Implement two modes:

- `build-split`: create a YOLO dataset variant where unknown-class labels are removed from train labels and preserved in metadata for evaluation.
- `evaluate`: run class-agnostic unknown proposal metrics against a trained leave-one-class model.

Outputs:

- `summary.json`
- `unknown_proposals.json`
- `report.md`

- [ ] **Step 6: Commit**

```bash
git add tools/osd_leave_one_class.py tests/test_osd_leave_one_class.py
git commit -m "feat: add OSD leave-one-class protocol diagnostic"
```

---

### Task 4: FASD-YOLO Teacher Quality Audit

**Files:**
- Create: `tools/fasd_teacher_audit.py`
- Test: `tests/test_fasd_teacher_audit.py`

- [ ] **Step 1: Write the failing tests**

```python
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from tools.fasd_teacher_audit import (
    MaskQualityConfig,
    edge_proxy_mask,
    score_mask_quality,
)
from tools.mrs_sliced_teacher import Box


class FASDTeacherAuditTests(unittest.TestCase):
    def test_edge_proxy_mask_finds_synthetic_box_edges(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.png"
            image = Image.new("L", (100, 100), color=20)
            draw = ImageDraw.Draw(image)
            draw.rectangle([30, 30, 70, 70], fill=220)
            image.save(path)

            mask = edge_proxy_mask(path, Box(0.25, 0.25, 0.75, 0.75))

            self.assertGreater(mask["foreground_ratio"], 0.01)
            self.assertLess(mask["foreground_ratio"], 0.80)

    def test_score_mask_quality_rejects_empty_mask(self) -> None:
        score = score_mask_quality({"foreground_ratio": 0.0, "edge_density": 0.0}, MaskQualityConfig())

        self.assertFalse(score["usable"])
        self.assertEqual(score["reason"], "foreground_ratio_out_of_range")

    def test_score_mask_quality_accepts_moderate_edge_mask(self) -> None:
        score = score_mask_quality({"foreground_ratio": 0.25, "edge_density": 0.15}, MaskQualityConfig())

        self.assertTrue(score["usable"])
        self.assertEqual(score["reason"], "usable")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
pytest tests/test_fasd_teacher_audit.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'tools.fasd_teacher_audit'`.

- [ ] **Step 3: Implement the proxy teacher and quality scoring**

Create `tools/fasd_teacher_audit.py`:

```python
#!/usr/bin/env python3
"""FASD-YOLO region-teacher quality audit."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from tools.mrs_sliced_teacher import Box


@dataclass(frozen=True)
class MaskQualityConfig:
    min_foreground_ratio: float = 0.02
    max_foreground_ratio: float = 0.75
    min_edge_density: float = 0.03


def edge_proxy_mask(image_path: Path, box: Box) -> dict[str, float]:
    with Image.open(image_path) as image:
        gray = image.convert("L")
        width, height = gray.size
        crop = gray.crop((
            int(box.x1 * width),
            int(box.y1 * height),
            int(box.x2 * width),
            int(box.y2 * height),
        ))
    arr = np.asarray(crop, dtype=np.float32) / 255.0
    if arr.size == 0:
        return {"foreground_ratio": 0.0, "edge_density": 0.0}
    gx = np.zeros_like(arr)
    gy = np.zeros_like(arr)
    if arr.shape[1] > 1:
        gx[:, 1:] = np.abs(arr[:, 1:] - arr[:, :-1])
    if arr.shape[0] > 1:
        gy[1:, :] = np.abs(arr[1:, :] - arr[:-1, :])
    edge = gx + gy
    threshold = max(float(edge.mean() + edge.std()), 0.05)
    mask = edge >= threshold
    return {
        "foreground_ratio": round(float(mask.mean()), 6),
        "edge_density": round(float(edge.mean()), 6),
    }


def score_mask_quality(mask_stats: dict[str, float], config: MaskQualityConfig) -> dict[str, Any]:
    foreground_ratio = float(mask_stats.get("foreground_ratio", 0.0))
    edge_density = float(mask_stats.get("edge_density", 0.0))
    if foreground_ratio < config.min_foreground_ratio or foreground_ratio > config.max_foreground_ratio:
        return {"usable": False, "reason": "foreground_ratio_out_of_range"}
    if edge_density < config.min_edge_density:
        return {"usable": False, "reason": "edge_density_too_low"}
    return {"usable": True, "reason": "usable"}
```

- [ ] **Step 4: Run the tests**

Run:

```bash
pytest tests/test_fasd_teacher_audit.py -v
```

Expected: PASS.

- [ ] **Step 5: Add CLI in the next TDD slice**

Add CLI help test for:

```bash
python tools/fasd_teacher_audit.py --help
```

Expected after implementation: help text includes `--data-yaml`, `--split`, `--out-dir`, `--provider`, `--max-samples`, and `--min-usable-rate`.

Implement provider `edge_proxy` first. Future `sam` or `dino` providers can be added behind the same provider interface after the edge-proxy audit proves the reporting path.

Outputs:

- `summary.json`
- `teacher_quality_records.json`
- `report.md`
- `preview/` with sampled crop overlays when `--save-previews` is passed.

- [ ] **Step 6: Commit**

```bash
git add tools/fasd_teacher_audit.py tests/test_fasd_teacher_audit.py
git commit -m "feat: add FASD teacher quality audit"
```

---

### Task 5: Ranking Report Across CLR, OSD, and FASD

**Files:**
- Create: `tools/rank_parallel_ideas.py`
- Test: `tests/test_rank_parallel_ideas.py`

- [ ] **Step 1: Write the failing tests**

```python
from __future__ import annotations

import unittest

from tools.rank_parallel_ideas import rank_ideas


class RankParallelIdeasTests(unittest.TestCase):
    def test_rank_prefers_passing_high_score_idea(self) -> None:
        rows = [
            {"idea": "CLR-YOLO", "passes_gate": True, "score": 0.72, "risk": "medium"},
            {"idea": "OSD-YOLO", "passes_gate": False, "score": 0.90, "risk": "high"},
            {"idea": "FASD-YOLO", "passes_gate": True, "score": 0.60, "risk": "high"},
        ]

        ranked = rank_ideas(rows)

        self.assertEqual([row["idea"] for row in ranked], ["CLR-YOLO", "FASD-YOLO", "OSD-YOLO"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
pytest tests/test_rank_parallel_ideas.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'tools.rank_parallel_ideas'`.

- [ ] **Step 3: Implement the ranking helper**

Create `tools/rank_parallel_ideas.py`:

```python
#!/usr/bin/env python3
"""Rank parallel S-level steel-defect idea diagnostics."""

from __future__ import annotations

from typing import Any


RISK_PENALTY = {"low": 0.0, "medium": 0.05, "high": 0.10}


def rank_ideas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> tuple[int, float]:
        passes = 1 if row.get("passes_gate") else 0
        score = float(row.get("score", 0.0)) - RISK_PENALTY.get(str(row.get("risk", "medium")), 0.05)
        return passes, score

    return sorted(rows, key=key, reverse=True)
```

- [ ] **Step 4: Run the tests**

Run:

```bash
pytest tests/test_rank_parallel_ideas.py -v
```

Expected: PASS.

- [ ] **Step 5: Add CLI in the next TDD slice**

Add a CLI test that writes three small summary files into a temp directory and runs:

```bash
python tools/rank_parallel_ideas.py --clr-summary <path> --osd-summary <path> --fasd-summary <path> --out <path>
```

Expected after implementation: output Markdown contains a ranked table and a clear recommendation line beginning with `Recommended next training route:`.

- [ ] **Step 6: Commit**

```bash
git add tools/rank_parallel_ideas.py tests/test_rank_parallel_ideas.py
git commit -m "feat: add parallel idea ranking report"
```

---

### Task 6: Experiment Matrix Command Rendering

**Files:**
- Modify: `tools/experiment_matrix.py`
- Modify: `tests/test_experiment_matrix.py`

- [ ] **Step 1: Add failing command-rendering tests**

Append tests for these matrix rows:

```yaml
runs:
  - id: clr_diag_val
    phase: parallel_s_idea_diagnostics
    kind: clr_diag
    data_yaml: data/steel-defect-mixed/data.yaml
    weights: runs/yolo/baseline_yolo11n_steel_mixed_640_e100_s42_v2/weights/best.pt
    out_dir: idea-stage/artifacts/parallel_s_ideas/clr_diag_val
    split: val
    imgsz: 640
  - id: osd_diag_scratches
    phase: parallel_s_idea_diagnostics
    kind: osd_diag
    data_yaml: data/steel-defect-mixed/data.yaml
    out_dir: idea-stage/artifacts/parallel_s_ideas/osd_scratches
    split: val
    extra:
      unknown_class: 9
  - id: fasd_audit_val
    phase: parallel_s_idea_diagnostics
    kind: fasd_audit
    data_yaml: data/steel-defect-mixed/data.yaml
    out_dir: idea-stage/artifacts/parallel_s_ideas/fasd_audit_val
    split: val
    max_images: 200
```

Expected command starts:

- `python tools/clr_label_reliability.py`
- `python tools/osd_leave_one_class.py`
- `python tools/fasd_teacher_audit.py`

- [ ] **Step 2: Run the focused test to verify it fails**

Run:

```bash
pytest tests/test_experiment_matrix.py -v
```

Expected: FAIL with `Unsupported run kind`.

- [ ] **Step 3: Add renderers**

Modify `tools/experiment_matrix.py`:

```python
def render_clr_diag_command(run: ExperimentRun) -> str:
    if not run.weights or not run.data_yaml or not run.out_dir:
        raise ValueError(f"CLR diagnostic run {run.id} requires weights, data_yaml, and out_dir.")
    parts = [
        "python", "tools/clr_label_reliability.py",
        "--weights", run.weights,
        "--data-yaml", run.data_yaml,
        "--split", run.split,
        "--out-dir", run.out_dir,
        "--imgsz", str(run.imgsz),
        "--conf", str(run.conf),
        "--max-det", str(run.max_det),
    ]
    if run.max_images:
        parts.extend(["--max-images", str(run.max_images)])
    return shell_join(parts)


def render_osd_diag_command(run: ExperimentRun) -> str:
    if not run.data_yaml or not run.out_dir:
        raise ValueError(f"OSD diagnostic run {run.id} requires data_yaml and out_dir.")
    unknown_class = run.extra.get("unknown_class")
    if unknown_class is None:
        raise ValueError(f"OSD diagnostic run {run.id} requires extra.unknown_class.")
    parts = [
        "python", "tools/osd_leave_one_class.py",
        "--mode", "evaluate",
        "--data-yaml", run.data_yaml,
        "--split", run.split,
        "--out-dir", run.out_dir,
        "--unknown-class", str(unknown_class),
        "--imgsz", str(run.imgsz),
    ]
    if run.weights:
        parts.extend(["--weights", run.weights])
    return shell_join(parts)


def render_fasd_audit_command(run: ExperimentRun) -> str:
    if not run.data_yaml or not run.out_dir:
        raise ValueError(f"FASD audit run {run.id} requires data_yaml and out_dir.")
    parts = [
        "python", "tools/fasd_teacher_audit.py",
        "--data-yaml", run.data_yaml,
        "--split", run.split,
        "--out-dir", run.out_dir,
        "--provider", str(run.extra.get("provider", "edge_proxy")),
    ]
    if run.max_images:
        parts.extend(["--max-samples", str(run.max_images)])
    return shell_join(parts)
```

Add branches in `render_run_command` for `clr_diag`, `osd_diag`, and `fasd_audit`.

- [ ] **Step 4: Run the tests**

Run:

```bash
pytest tests/test_experiment_matrix.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/experiment_matrix.py tests/test_experiment_matrix.py
git commit -m "feat: render parallel S-idea diagnostic commands"
```

---

### Task 7: SSH Dry-Run Support

**Files:**
- Modify: `tools/ssh_submit.py`
- Modify: `tests/test_ssh_submit.py`

- [ ] **Step 1: Add failing dry-run tests**

Add tests that instantiate `ExperimentRun(kind="clr_diag")`, `ExperimentRun(kind="osd_diag")`, and `ExperimentRun(kind="fasd_audit")`, then call `build_remote_command`.

Expected remote command snippets:

```text
~/train-venv/bin/python tools/clr_label_reliability.py
~/train-venv/bin/python tools/osd_leave_one_class.py
~/train-venv/bin/python tools/fasd_teacher_audit.py
--enable-clearml
--clearml-task-name <run-id>
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run:

```bash
pytest tests/test_ssh_submit.py -v
```

Expected: FAIL with `Unsupported run kind='clr_diag'` or equivalent.

- [ ] **Step 3: Add remote command builders**

Add builder functions following the existing `build_qdh_diag_remote_command` pattern. Each command must:

- `cd` into `args.remote_root`
- set `PYTHONPATH`
- call `args.remote_python`
- resolve relative `data_yaml`, `weights`, and `out_dir` through `_remote_path`
- pass `--enable-clearml`, `--clearml-project`, and `--clearml-task-name`

- [ ] **Step 4: Run the tests**

Run:

```bash
pytest tests/test_ssh_submit.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/ssh_submit.py tests/test_ssh_submit.py
git commit -m "feat: submit parallel S-idea diagnostics by SSH"
```

---

### Task 8: Diagnostic Matrix

**Files:**
- Create: `research/yolo11_welding_defect/experiments/parallel_s_idea_diagnostics.yaml`

- [ ] **Step 1: Add the matrix**

```yaml
project: yolo_welding_defect
matrix: parallel_s_idea_diagnostics
created: "2026-06-02"

description: >
  Parallel diagnostic matrix for CLR-YOLO, OSD-YOLO, and FASD-YOLO.
  These runs decide which S-level idea deserves training. They do not make
  final paper claims.

defaults:
  data_yaml: data/steel-defect-mixed/data.yaml
  imgsz: 640
  seed: 42
  runs_dir: runs/yolo

runs:
  - id: clr_diag_steel_mixed_val_s42
    phase: parallel_s_idea_diagnostics
    kind: clr_diag
    data_yaml: data/steel-defect-mixed/data.yaml
    weights: runs/yolo/baseline_yolo11n_steel_mixed_640_e100_s42_v2/weights/best.pt
    out_dir: idea-stage/artifacts/parallel_s_ideas/clr_diag_steel_mixed_val_s42
    split: val
    imgsz: 640
    conf: 0.05
    max_det: 300

  - id: osd_diag_scratches_unknown_val_s42
    phase: parallel_s_idea_diagnostics
    kind: osd_diag
    data_yaml: data/steel-defect-mixed/data.yaml
    out_dir: idea-stage/artifacts/parallel_s_ideas/osd_diag_scratches_unknown_val_s42
    split: val
    imgsz: 640
    extra:
      unknown_class: 9

  - id: fasd_audit_edge_proxy_val_s42
    phase: parallel_s_idea_diagnostics
    kind: fasd_audit
    data_yaml: data/steel-defect-mixed/data.yaml
    out_dir: idea-stage/artifacts/parallel_s_ideas/fasd_audit_edge_proxy_val_s42
    split: val
    max_images: 300
    extra:
      provider: edge_proxy
```

- [ ] **Step 2: Render commands locally**

Run:

```bash
python tools/experiment_matrix.py \
  --matrix research/yolo11_welding_defect/experiments/parallel_s_idea_diagnostics.yaml \
  --phase parallel_s_idea_diagnostics \
  --print-commands
```

Expected: three commands, one for each idea.

- [ ] **Step 3: Dry-run SSH submission**

Run:

```bash
python tools/ssh_submit.py \
  --matrix research/yolo11_welding_defect/experiments/parallel_s_idea_diagnostics.yaml \
  --phase parallel_s_idea_diagnostics \
  --dry-run
```

Expected: three remote commands and no actual SSH execution.

- [ ] **Step 4: Commit**

```bash
git add research/yolo11_welding_defect/experiments/parallel_s_idea_diagnostics.yaml
git commit -m "chore: add parallel S-idea diagnostic matrix"
```

---

### Task 9: Issue Breakdown Approval

**Files:**
- Create only after user approval: `issues/P45-parallel-s-idea-diagnostics.md`
- Create only after user approval: `issues/P45-1-common-diagnostic-helpers.md`
- Create only after user approval: `issues/P45-2-clr-diagnostic.md`
- Create only after user approval: `issues/P45-3-osd-protocol.md`
- Create only after user approval: `issues/P45-4-fasd-teacher-audit.md`
- Create only after user approval: `issues/P45-5-ranking-report.md`
- Create only after user approval: `issues/P45-6-matrix-and-ssh.md`

Draft vertical slices:

1. **P45.1 Shared Diagnostic Helpers**  
   Type: AFK  
   Blocked by: None  
   Acceptance: `tools/idea_diag_common.py` and tests pass.

2. **P45.2 CLR Label Reliability Diagnostic**  
   Type: AFK  
   Blocked by: P45.1  
   Acceptance: CLR CLI writes `summary.json`, `reliability_records.json`, and `report.md`.

3. **P45.3 OSD Leave-One-Class Protocol**  
   Type: AFK  
   Blocked by: P45.1  
   Acceptance: OSD CLI can build/evaluate a held-out unknown-class protocol.

4. **P45.4 FASD Teacher Audit**  
   Type: AFK  
   Blocked by: P45.1  
   Acceptance: FASD CLI audits edge-proxy teacher quality and writes previewable artifacts.

5. **P45.5 Parallel Idea Ranking Report**  
   Type: AFK  
   Blocked by: P45.2, P45.3, P45.4  
   Acceptance: ranking report recommends one training route or rejects all three.

6. **P45.6 Matrix and SSH Submission Support**  
   Type: AFK  
   Blocked by: P45.2, P45.3, P45.4  
   Acceptance: experiment matrix renders and SSH dry-run prints all three diagnostic commands.

7. **P45.7 Human Route Selection**  
   Type: HITL  
   Blocked by: P45.5, P45.6  
   Acceptance: user selects CLR, OSD, FASD, or a new idea for training.

Do not publish these issue files until the user approves the granularity and dependencies.

---

## Self-Review

- Spec coverage: CLR, OSD, and FASD each have a diagnostic CLI, tests, outputs, and a gate before training.
- TDD coverage: every implementation task starts with behavior tests and only then adds code.
- Issue coverage: the proposed P45 slices are vertical and independently verifiable.
- Scope control: no full training, no revival of failed families, and no foundation model dependency is required for the first FASD audit.
