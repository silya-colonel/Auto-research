#!/usr/bin/env python3
"""Create a deterministic train/val/test split for a YOLO dataset."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class SplitRatios:
    train: float
    val: float
    test: float


@dataclass(frozen=True)
class ImageLabelPair:
    image: Path
    label: Path


def load_simple_yaml(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if not path.exists():
        return result
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        if value.startswith("["):
            try:
                result[key.strip()] = json.loads(value.replace("'", '"'))
            except json.JSONDecodeError:
                result[key.strip()] = value
        elif value.isdigit():
            result[key.strip()] = int(value)
        else:
            result[key.strip()] = value.strip("'\"")
    return result


def names_from_dataset(root: Path) -> list[str]:
    data_yaml = load_simple_yaml(root / "data.yaml")
    names = data_yaml.get("names")
    if isinstance(names, list):
        return [str(name) for name in names]
    classes_txt = root / "classes.txt"
    if classes_txt.exists():
        return [line.strip() for line in classes_txt.read_text(encoding="utf-8").splitlines() if line.strip()]
    return []


def label_for_image(root: Path, image: Path) -> Path:
    rel = image.relative_to(root)
    parts = list(rel.parts)
    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        return root.joinpath(*parts).with_suffix(".txt")
    return image.with_suffix(".txt")


def collect_pairs(root: Path) -> list[ImageLabelPair]:
    images_root = root / "images"
    pairs: list[ImageLabelPair] = []
    for image in sorted(path for path in images_root.rglob("*") if path.suffix.lower() in IMAGE_EXTS):
        label = label_for_image(root, image)
        if label.exists():
            pairs.append(ImageLabelPair(image=image, label=label))
    return pairs


def split_counts(total: int, ratios: SplitRatios) -> tuple[int, int, int]:
    if total == 0:
        return 0, 0, 0
    train_count = int(total * ratios.train)
    val_count = int(total * ratios.val)
    test_count = total - train_count - val_count
    if ratios.val > 0 and total >= 3 and val_count == 0:
        val_count = 1
        if test_count > 1:
            test_count -= 1
        else:
            train_count -= 1
    if ratios.test > 0 and total >= 3 and test_count == 0:
        test_count = 1
        train_count -= 1
    return train_count, val_count, test_count


def reset_split_dirs(out: Path) -> None:
    for split in ("train", "val", "test"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)


def copy_pair(pair: ImageLabelPair, out: Path, split: str) -> None:
    shutil.copy2(pair.image, out / "images" / split / pair.image.name)
    shutil.copy2(pair.label, out / "labels" / split / pair.label.name)


def write_data_yaml(out: Path, names: list[str]) -> None:
    quoted_names = ", ".join(json.dumps(name, ensure_ascii=False) for name in names)
    text = "\n".join(
        [
            "path: .",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            f"nc: {len(names)}",
            f"names: [{quoted_names}]",
            "",
        ]
    )
    (out / "data.yaml").write_text(text, encoding="utf-8")
    if names:
        (out / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")


def split_yolo_dataset(root: Path, out: Path, ratios: SplitRatios, seed: int) -> dict[str, int]:
    pairs = collect_pairs(root)
    shuffled = pairs[:]
    random.Random(seed).shuffle(shuffled)
    train_count, val_count, _ = split_counts(len(shuffled), ratios)
    buckets = {
        "train": shuffled[:train_count],
        "val": shuffled[train_count : train_count + val_count],
        "test": shuffled[train_count + val_count :],
    }
    reset_split_dirs(out)
    for split, split_pairs in buckets.items():
        for pair in split_pairs:
            copy_pair(pair, out, split)
    write_data_yaml(out, names_from_dataset(root))
    summary = {split: len(split_pairs) for split, split_pairs in buckets.items()}
    (out / "split_summary.json").write_text(
        json.dumps({"source": str(root), "seed": seed, "splits": summary}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a deterministic train/val/test split for a YOLO dataset.")
    parser.add_argument("dataset", type=Path, help="Source YOLO dataset root.")
    parser.add_argument("--out", type=Path, required=True, help="Output dataset root.")
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    total = args.train + args.val + args.test
    if not 0.999 <= total <= 1.001:
        raise SystemExit("--train + --val + --test must sum to 1.0")
    summary = split_yolo_dataset(args.dataset, args.out, SplitRatios(args.train, args.val, args.test), args.seed)
    print(f"split dataset saved: {args.out} ({summary})")


if __name__ == "__main__":
    main()
