#!/usr/bin/env python3
"""Convert a Roboflow-style COCO export into a clean YOLO dataset.

The converter is intentionally non-destructive: it never moves files from the
source dataset. Images are hard-linked by default when possible and copied as a
fallback. The output layout is:

  output/
    images/train
    images/val
    labels/train
    labels/val
    data.yaml
    classes.txt
    conversion_report.json
    conversion_report.md
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class SplitSpec:
    source_name: str
    yolo_name: str


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "images" not in data or "annotations" not in data:
        raise ValueError(f"not a COCO annotation file: {path}")
    return data


def image_path_for(split_dir: Path, file_name: str) -> Path | None:
    direct = split_dir / file_name
    if direct.exists():
        return direct
    name = Path(file_name).name
    candidate = split_dir / name
    if candidate.exists():
        return candidate
    matches = [p for p in split_dir.rglob(name) if p.suffix.lower() in IMAGE_EXTS]
    return matches[0] if matches else None


def link_or_copy(src: Path, dst: Path, mode: str) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "exists"
    if mode == "copy":
        shutil.copy2(src, dst)
        return "copy"
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def yolo_line(
    class_id: int,
    bbox: list[float],
    image_width: float,
    image_height: float,
) -> tuple[str | None, dict[str, float]]:
    x, y, w, h = [float(v) for v in bbox]
    x1 = clip(x, 0.0, image_width)
    y1 = clip(y, 0.0, image_height)
    x2 = clip(x + w, 0.0, image_width)
    y2 = clip(y + h, 0.0, image_height)
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0 or bh <= 0 or image_width <= 0 or image_height <= 0:
        return None, {}
    cx = (x1 + bw / 2.0) / image_width
    cy = (y1 + bh / 2.0) / image_height
    nw = bw / image_width
    nh = bh / image_height
    line = f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"
    return line, {
        "area_ratio": nw * nh,
        "aspect_ratio": max(nw / nh, nh / nw) if nw > 0 and nh > 0 else math.inf,
        "width_ratio": nw,
        "height_ratio": nh,
    }


def size_bin(area_ratio: float) -> str:
    if area_ratio < 0.0005:
        return "tiny"
    if area_ratio < 0.0025:
        return "small"
    if area_ratio < 0.01:
        return "medium"
    return "large"


def aspect_bin(aspect_ratio: float) -> str:
    if aspect_ratio >= 8:
        return "extreme_elongated"
    if aspect_ratio >= 4:
        return "elongated"
    if aspect_ratio >= 2:
        return "moderate"
    return "near_square"


def build_categories(coco_by_split: dict[str, dict[str, Any]], drop_empty: bool) -> tuple[dict[int, int], list[str], list[str]]:
    categories_by_id: dict[int, str] = {}
    annotation_counts: Counter[int] = Counter()
    for coco in coco_by_split.values():
        for cat in coco.get("categories", []):
            categories_by_id[int(cat["id"])] = str(cat["name"])
        for ann in coco.get("annotations", []):
            annotation_counts[int(ann["category_id"])] += 1

    kept_original_ids: list[int] = []
    dropped: list[str] = []
    for cat_id in sorted(categories_by_id):
        if drop_empty and annotation_counts[cat_id] == 0:
            dropped.append(f"{cat_id}:{categories_by_id[cat_id]}")
            continue
        kept_original_ids.append(cat_id)

    category_id_to_yolo = {cat_id: idx for idx, cat_id in enumerate(kept_original_ids)}
    names = [categories_by_id[cat_id] for cat_id in kept_original_ids]
    return category_id_to_yolo, names, dropped


def convert_split(
    source_root: Path,
    output_root: Path,
    spec: SplitSpec,
    category_id_to_yolo: dict[int, int],
    names: list[str],
    copy_mode: str,
) -> dict[str, Any]:
    split_dir = source_root / spec.source_name
    ann_path = split_dir / "_annotations.coco.json"
    coco = load_json(ann_path)
    images = {int(img["id"]): img for img in coco.get("images", [])}
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        annotations_by_image[int(ann["image_id"])].append(ann)

    image_out_dir = output_root / "images" / spec.yolo_name
    label_out_dir = output_root / "labels" / spec.yolo_name
    image_out_dir.mkdir(parents=True, exist_ok=True)
    label_out_dir.mkdir(parents=True, exist_ok=True)

    class_counts: Counter[int] = Counter()
    size_counts: Counter[str] = Counter()
    aspect_counts: Counter[str] = Counter()
    per_class_size_counts: dict[str, Counter[str]] = {name: Counter() for name in names}
    per_class_aspect_counts: dict[str, Counter[str]] = {name: Counter() for name in names}
    missing_images: list[str] = []
    invalid_boxes = 0
    copied = 0
    linked = 0

    for image_id, image_info in sorted(images.items()):
        file_name = str(image_info["file_name"])
        src_image = image_path_for(split_dir, file_name)
        if src_image is None:
            missing_images.append(file_name)
            continue

        dst_image = image_out_dir / Path(file_name).name
        transfer = link_or_copy(src_image, dst_image, copy_mode)
        if transfer == "hardlink":
            linked += 1
        elif transfer == "copy":
            copied += 1

        image_width = float(image_info["width"])
        image_height = float(image_info["height"])
        label_lines: list[str] = []
        for ann in annotations_by_image.get(image_id, []):
            original_cat = int(ann["category_id"])
            if original_cat not in category_id_to_yolo:
                continue
            class_id = category_id_to_yolo[original_cat]
            line, box_stats = yolo_line(class_id, ann["bbox"], image_width, image_height)
            if line is None:
                invalid_boxes += 1
                continue
            label_lines.append(line)
            class_counts[class_id] += 1
            s_bin = size_bin(box_stats["area_ratio"])
            a_bin = aspect_bin(box_stats["aspect_ratio"])
            size_counts[s_bin] += 1
            aspect_counts[a_bin] += 1
            per_class_size_counts[names[class_id]][s_bin] += 1
            per_class_aspect_counts[names[class_id]][a_bin] += 1

        label_path = label_out_dir / (dst_image.stem + ".txt")
        label_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")

    return {
        "source_split": spec.source_name,
        "yolo_split": spec.yolo_name,
        "images_declared": len(images),
        "images_written": len(list(image_out_dir.glob("*"))),
        "linked_images": linked,
        "copied_images": copied,
        "missing_images": missing_images,
        "invalid_boxes": invalid_boxes,
        "class_counts": {names[k]: v for k, v in sorted(class_counts.items())},
        "size_counts": dict(sorted(size_counts.items())),
        "aspect_counts": dict(sorted(aspect_counts.items())),
        "per_class_size_counts": {name: dict(counter) for name, counter in per_class_size_counts.items()},
        "per_class_aspect_counts": {name: dict(counter) for name, counter in per_class_aspect_counts.items()},
    }


def write_data_yaml(output_root: Path, names: list[str]) -> None:
    lines = [
        f"path: {output_root.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(names)}",
        "names:",
    ]
    lines.extend(f"  {idx}: {name}" for idx, name in enumerate(names))
    (output_root / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_root / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")


def report_to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Roboflow COCO to YOLO Conversion Report",
        "",
        f"- **source**: `{report['source_root']}`",
        f"- **output**: `{report['output_root']}`",
        f"- **classes**: `{report['nc']}`",
        f"- **names**: {', '.join(report['names'])}",
        f"- **dropped empty categories**: {', '.join(report['dropped_empty_categories']) or 'none'}",
        "",
        "## Splits",
        "",
        "| Split | Images Declared | Images Written | Linked | Copied | Missing Images | Invalid Boxes |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split in report["splits"]:
        lines.append(
            f"| {split['yolo_split']} | {split['images_declared']} | {split['images_written']} | "
            f"{split['linked_images']} | {split['copied_images']} | {len(split['missing_images'])} | {split['invalid_boxes']} |"
        )

    lines.extend(["", "## Class Counts", ""])
    for split in report["splits"]:
        lines.append(f"### {split['yolo_split']}")
        lines.append("")
        lines.append("| Class | Count |")
        lines.append("|---|---:|")
        for name in report["names"]:
            lines.append(f"| {name} | {split['class_counts'].get(name, 0)} |")
        lines.append("")

    lines.extend(["## Object Size Bins", ""])
    lines.append("Bins use normalized box area: tiny < 0.0005, small < 0.0025, medium < 0.01, large otherwise.")
    lines.append("")
    for split in report["splits"]:
        lines.append(f"- **{split['yolo_split']}**: `{split['size_counts']}`")

    lines.extend(["", "## Aspect Ratio Bins", ""])
    lines.append("Bins use max(width/height, height/width): moderate >= 2, elongated >= 4, extreme >= 8.")
    lines.append("")
    for split in report["splits"]:
        lines.append(f"- **{split['yolo_split']}**: `{split['aspect_counts']}`")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Empty labels are expected for images with no annotations.",
            "- Inspect heavily imbalanced classes before long training.",
            "- Track per-class recall for minority and elongated defect classes.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_split_specs(values: list[str]) -> list[SplitSpec]:
    specs: list[SplitSpec] = []
    for value in values:
        if ":" not in value:
            raise ValueError(f"split mapping must be source:yolo, got {value}")
        source_name, yolo_name = value.split(":", 1)
        specs.append(SplitSpec(source_name=source_name, yolo_name=yolo_name))
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Roboflow COCO folders to YOLO format.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", action="append", default=["train:train", "valid:val"])
    parser.add_argument("--copy-mode", choices=["hardlink", "copy"], default="hardlink")
    parser.add_argument("--keep-empty-categories", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_root = args.source.resolve()
    output_root = args.output.resolve()
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise SystemExit(f"output is not empty, use --overwrite to replace files: {output_root}")
    if args.overwrite and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    specs = parse_split_specs(args.split)
    coco_by_split = {
        spec.source_name: load_json(source_root / spec.source_name / "_annotations.coco.json")
        for spec in specs
    }
    category_id_to_yolo, names, dropped = build_categories(coco_by_split, not args.keep_empty_categories)
    if not names:
        raise SystemExit("no categories remain after filtering")

    split_reports = [
        convert_split(source_root, output_root, spec, category_id_to_yolo, names, args.copy_mode)
        for spec in specs
    ]
    write_data_yaml(output_root, names)

    report = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "nc": len(names),
        "names": names,
        "dropped_empty_categories": dropped,
        "copy_mode": args.copy_mode,
        "splits": split_reports,
    }
    (output_root / "conversion_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_root / "conversion_report.md").write_text(report_to_markdown(report), encoding="utf-8")
    print(f"Wrote {output_root / 'data.yaml'}")
    print(f"Wrote {output_root / 'conversion_report.md'}")


if __name__ == "__main__":
    main()
