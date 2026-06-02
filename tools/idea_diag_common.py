#!/usr/bin/env python3
"""Shared helpers for steel-defect idea diagnostics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


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


def _json_default(value: Any) -> Any:
    if isinstance(value, Box):
        return {"x1": value.x1, "y1": value.y1, "x2": value.x2, "y2": value.y2}
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, default=_json_default, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_data_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise SystemExit("PyYAML is required to read YOLO data.yaml files.") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return payload


def dataset_root(data_yaml: Path, payload: dict[str, Any] | None = None) -> Path:
    payload = payload if payload is not None else load_data_yaml(data_yaml)
    root_value = payload.get("path", ".")
    root = Path(str(root_value)).expanduser()
    if not root.is_absolute():
        root = data_yaml.parent / root
    return root.resolve()


def _as_path_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def resolve_split_sources(data_yaml: Path, split: str) -> list[Path]:
    payload = load_data_yaml(data_yaml)
    root = dataset_root(data_yaml, payload)
    sources: list[Path] = []
    for value in _as_path_list(payload.get(split)):
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = root / path
        sources.append(path)
    return sources


def iter_image_paths(data_yaml: Path, split: str, max_images: int = 0) -> list[Path]:
    paths: list[Path] = []
    for source in resolve_split_sources(data_yaml, split):
        if source.is_file() and source.suffix.lower() == ".txt":
            for line in source.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped:
                    paths.append(Path(stripped).expanduser())
        elif source.is_file() and source.suffix.lower() in IMAGE_SUFFIXES:
            paths.append(source)
        elif source.is_dir():
            paths.extend(path for path in sorted(source.rglob("*")) if path.suffix.lower() in IMAGE_SUFFIXES)
    if max_images > 0:
        paths = paths[:max_images]
    return paths


def image_id_for_path(path: Path) -> str:
    return path.stem


def label_path_for_image(image_path: Path, data_yaml: Path) -> Path:
    root = dataset_root(data_yaml)
    try:
        rel = image_path.resolve().relative_to(root)
    except ValueError:
        return image_path.with_suffix(".txt")
    parts = list(rel.parts)
    if parts and parts[0] == "images":
        parts[0] = "labels"
    return (root / Path(*parts)).with_suffix(".txt")


def yolo_line_to_box(line: str) -> tuple[int, Box] | None:
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    cls = int(float(parts[0]))
    cx, cy, width, height = (float(value) for value in parts[1:5])
    box = Box(cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2).clipped()
    return cls, box


def load_yolo_labels(data_yaml: Path, split: str, max_images: int = 0) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    for image_path in iter_image_paths(data_yaml, split, max_images=max_images):
        label_path = label_path_for_image(image_path, data_yaml)
        if not label_path.exists():
            continue
        rows = label_path.read_text(encoding="utf-8").splitlines()
        for gt_index, line in enumerate(rows):
            parsed = yolo_line_to_box(line)
            if parsed is None:
                continue
            cls, box = parsed
            labels.append(
                {
                    "image_id": image_id_for_path(image_path),
                    "image_path": str(image_path),
                    "label_path": str(label_path),
                    "gt_index": gt_index,
                    "cls": cls,
                    "box": box,
                }
            )
    return labels


def parse_prediction_box(value: Any) -> Box:
    if isinstance(value, Box):
        return value
    if isinstance(value, dict):
        return Box(float(value["x1"]), float(value["y1"]), float(value["x2"]), float(value["y2"]))
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        values = list(value)
        if len(values) >= 4:
            return Box(float(values[0]), float(values[1]), float(values[2]), float(values[3]))
    raise ValueError(f"Unsupported prediction box format: {value!r}")


def normalize_predictions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for row in rows:
        prediction = dict(row)
        prediction["image_id"] = str(prediction["image_id"])
        prediction["cls"] = int(prediction["cls"])
        prediction["conf"] = float(prediction.get("conf", 1.0))
        prediction["box"] = parse_prediction_box(prediction["box"]).clipped()
        predictions.append(prediction)
    return predictions


def yolo_result_to_predictions(results: Any) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for result in results:
        image_id = image_id_for_path(Path(str(result.path)))
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        xyxyn = boxes.xyxyn.cpu().tolist()
        classes = boxes.cls.cpu().tolist()
        confidences = boxes.conf.cpu().tolist()
        for box, cls, conf in zip(xyxyn, classes, confidences):
            predictions.append(
                {
                    "image_id": image_id,
                    "cls": int(cls),
                    "conf": float(conf),
                    "box": Box(float(box[0]), float(box[1]), float(box[2]), float(box[3])).clipped(),
                }
            )
    return predictions


def flatten_numeric_metrics(payload: dict[str, Any], prefix: str = "") -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in payload.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, bool):
            metrics[name] = 1.0 if value else 0.0
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            metrics[name] = float(value)
        elif isinstance(value, dict):
            metrics.update(flatten_numeric_metrics(value, prefix=name))
    return metrics


def finish_clearml_task(
    *,
    enabled: bool,
    project_name: str,
    task_name: str | None,
    summary: dict[str, Any],
    artifacts: dict[str, Path],
    task_factory: Callable[..., Any] | None = None,
) -> None:
    if not enabled:
        return
    if task_factory is None:
        from clearml import Task  # type: ignore

        task_factory = Task.init

    task = task_factory(
        project_name=project_name,
        task_name=task_name or "parallel_s_idea_diagnostic",
        auto_connect_frameworks=False,
    )
    logger = task.get_logger()
    for metric_name, value in flatten_numeric_metrics(summary).items():
        logger.report_scalar(title="summary", series=metric_name, value=value, iteration=0)
    for name, path in artifacts.items():
        if path.exists():
            task.upload_artifact(name=name, artifact_object=str(path))
    task.close()
