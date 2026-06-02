#!/usr/bin/env python3
"""Shared helpers for steel-defect idea diagnostics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LabelKey:
    image_id: str
    gt_index: int
    cls: int


@dataclass(frozen=True)
class Box:
    """Normalized or absolute xyxy box."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    def clipped(self) -> "Box":
        return Box(
            max(0.0, min(1.0, self.x1)),
            max(0.0, min(1.0, self.y1)),
            max(0.0, min(1.0, self.x2)),
            max(0.0, min(1.0, self.y2)),
        )


def label_key(label: dict[str, Any]) -> LabelKey:
    return LabelKey(str(label["image_id"]), int(label["gt_index"]), int(label["cls"]))


def box_iou(a: Box, b: Box) -> float:
    inter_x1 = max(a.x1, b.x1)
    inter_y1 = max(a.y1, b.y1)
    inter_x2 = min(a.x2, b.x2)
    inter_y2 = min(a.y2, b.y2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    union = a.area + b.area - intersection
    if union <= 0:
        return 0.0
    return intersection / union


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
