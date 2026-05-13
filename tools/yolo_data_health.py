#!/usr/bin/env python3
"""Lightweight YOLO data.yaml and label health checker.

The script intentionally avoids project-specific assumptions. It checks the
common Ultralytics layout and writes Markdown/JSON reports suitable for ARIS.
"""

from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from pathlib import Path
from typing import Any


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore

        with path.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        if not isinstance(loaded, dict):
            raise ValueError("data.yaml must contain a mapping at top level")
        return loaded
    except ImportError:
        return load_simple_yaml(path)


def load_simple_yaml(path: Path) -> dict[str, Any]:
    """Small fallback parser for common Ultralytics data.yaml files."""
    result: dict[str, Any] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip() or ":" not in line:
            i += 1
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value:
            result[key] = parse_scalar(value)
            i += 1
            continue
        items: list[str] = []
        i += 1
        while i < len(lines) and lines[i].startswith((" ", "\t", "-")):
            item = lines[i].strip()
            if item.startswith("-"):
                items.append(item[1:].strip())
            i += 1
        result[key] = items
    return result


def parse_scalar(value: str) -> Any:
    if value.startswith(("[", "{")):
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value
    if value.isdigit():
        return int(value)
    return value.strip("'\"")


def resolve_path(value: Any, data_yaml: Path, root: Path | None) -> Path | None:
    if value is None:
        return None
    if isinstance(value, list):
        return None
    text = str(value)
    path = Path(text)
    if path.is_absolute():
        return path
    base = root if root is not None else data_yaml.parent
    return (base / path).resolve()


def names_from_yaml(cfg: dict[str, Any]) -> list[str]:
    names = cfg.get("names", [])
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names)]
    if isinstance(names, list):
        return [str(x) for x in names]
    return []


def collect_images(path: Path | None) -> list[Path]:
    if path is None or not path.exists():
        return []
    if path.is_file():
        return [Path(p.strip()) for p in path.read_text(encoding="utf-8").splitlines() if p.strip()]
    return [p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTS]


def label_path_for_image(image: Path) -> Path:
    parts = list(image.parts)
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image.with_suffix(".txt")


def inspect_labels(images: list[Path], nc: int | None, max_files: int) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "images_checked": min(len(images), max_files),
        "missing_labels": 0,
        "empty_labels": 0,
        "bad_rows": 0,
        "out_of_range_boxes": 0,
        "unknown_classes": 0,
        "class_counts": Counter(),
        "tiny_boxes": 0,
    }
    for image in images[:max_files]:
        label = label_path_for_image(image)
        if not label.exists():
            stats["missing_labels"] += 1
            continue
        rows = [row.strip() for row in label.read_text(encoding="utf-8", errors="ignore").splitlines() if row.strip()]
        if not rows:
            stats["empty_labels"] += 1
            continue
        for row in rows:
            parts = row.split()
            if len(parts) != 5:
                stats["bad_rows"] += 1
                continue
            try:
                cls = int(float(parts[0]))
                x, y, w, h = [float(v) for v in parts[1:]]
            except ValueError:
                stats["bad_rows"] += 1
                continue
            if nc is not None and (cls < 0 or cls >= nc):
                stats["unknown_classes"] += 1
            stats["class_counts"][cls] += 1
            if not all(0.0 <= v <= 1.0 for v in (x, y, w, h)):
                stats["out_of_range_boxes"] += 1
            if w * h < 0.0005:
                stats["tiny_boxes"] += 1
    stats["class_counts"] = dict(sorted(stats["class_counts"].items()))
    return stats


def build_report(data_yaml: Path, cfg: dict[str, Any], max_files: int) -> dict[str, Any]:
    root = resolve_path(cfg.get("path"), data_yaml, data_yaml.parent)
    names = names_from_yaml(cfg)
    nc = int(cfg["nc"]) if "nc" in cfg and str(cfg["nc"]).isdigit() else len(names) or None
    splits: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        split_path = resolve_path(cfg.get(split), data_yaml, root)
        images = collect_images(split_path)
        splits[split] = {
            "path": str(split_path) if split_path else None,
            "exists": bool(split_path and split_path.exists()),
            "images": len(images),
            "label_stats": inspect_labels(images, nc, max_files),
        }
    return {
        "data_yaml": str(data_yaml),
        "root": str(root) if root else None,
        "nc": nc,
        "names": names,
        "splits": splits,
    }


def to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# YOLO Data Health Report",
        "",
        f"- **data.yaml**: `{report['data_yaml']}`",
        f"- **root**: `{report['root']}`",
        f"- **classes**: `{report['nc']}`",
        f"- **names**: {', '.join(report['names']) if report['names'] else '[missing]'}",
        "",
        "## Splits",
        "",
        "| Split | Exists | Images | Missing Labels | Empty Labels | Bad Rows | Unknown Classes | Out-of-range Boxes | Tiny Boxes |",
        "|-------|--------|--------|----------------|--------------|----------|-----------------|--------------------|------------|",
    ]
    for split, info in report["splits"].items():
        stats = info["label_stats"]
        lines.append(
            f"| {split} | {info['exists']} | {info['images']} | {stats['missing_labels']} | "
            f"{stats['empty_labels']} | {stats['bad_rows']} | {stats['unknown_classes']} | "
            f"{stats['out_of_range_boxes']} | {stats['tiny_boxes']} |"
        )
    lines.extend(["", "## Class Counts", ""])
    for split, info in report["splits"].items():
        counts = info["label_stats"]["class_counts"]
        lines.append(f"- **{split}**: `{counts}`")
    lines.extend(
        [
            "",
            "## Suggested Next Step",
            "",
            "- Fix missing labels, malformed rows, or path issues before training.",
            "- If minority classes are sparse, track per-class recall in metrics.",
            "- If tiny boxes are common, include an image-size sweep such as 640 vs 960.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Check YOLO data.yaml and labels.")
    parser.add_argument("data_yaml", type=Path)
    parser.add_argument("--max-files", type=int, default=5000)
    parser.add_argument("--markdown-out", type=Path, default=Path("results/DATA_HEALTH_REPORT.md"))
    parser.add_argument("--json-out", type=Path, default=Path("results/DATA_HEALTH_REPORT.json"))
    args = parser.parse_args()

    cfg = load_yaml(args.data_yaml)
    report = build_report(args.data_yaml.resolve(), cfg, args.max_files)

    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.write_text(to_markdown(report), encoding="utf-8")
    args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {args.markdown_out} and {args.json_out}")


if __name__ == "__main__":
    main()
