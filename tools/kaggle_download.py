#!/usr/bin/env python3
"""从 Kaggle 下载数据集，自动转换 YOLO 格式并分割 train/val。

用法:
  python kaggle_download.py
  python kaggle_download.py ultralytics/coco8
  python kaggle_download.py huazai/visdrone2019 --path data/visdrone --split-test
  python kaggle_download.py alessiocorrado99/animals10 --path data/animals
  python kaggle_download.py ultralytics/coco8 --no-convert --no-split
  python kaggle_download.py <dataset> --list

管道:
  下载 → 检测格式 → 转换(COCO/VOC/分类) → 分割(8:2 或 8:1:1) → data.yaml
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

# ── 图片扩展名 ────────────────────────────────────────────────────

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

DATASET_REF_RE = re.compile(
    r"(?P<owner>[A-Za-z0-9][A-Za-z0-9_-]*)/(?P<slug>[A-Za-z0-9][A-Za-z0-9_-]*)"
)
KAGGLE_URL_RE = re.compile(
    r"kaggle\.com/(?:datasets/)?(?P<owner>[A-Za-z0-9][A-Za-z0-9_-]*)/"
    r"(?P<slug>[A-Za-z0-9][A-Za-z0-9_-]*)"
)
KAGGLEHUB_CALL_RE = re.compile(
    r"dataset_download\(\s*[\"'](?P<dataset>[^\"']+/[^\"']+)[\"']"
)

# ── Kaggle 下载 ────────────────────────────────────────────────────


def download_kagglehub(dataset: str, out_dir: Path) -> Path:
    import kagglehub

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[kagglehub] 下载 {dataset} -> {out_dir}")
    sig = inspect.signature(kagglehub.dataset_download)
    kwargs = {}
    if "output_dir" in sig.parameters:
        kwargs["output_dir"] = str(out_dir)
    download_path = Path(kagglehub.dataset_download(dataset, **kwargs))
    if download_path.resolve() != out_dir.resolve():
        _copy_downloaded_tree(download_path, out_dir)
    print(f"[kagglehub] 完成: {download_path}")
    return out_dir


def download_kaggle_cli(dataset: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "datasets", "download", dataset, "-p", str(out_dir), "--unzip"]
    print(f"[kaggle CLI] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[kaggle CLI] 失败: {result.stderr}")
        sys.exit(1)
    print(f"[kaggle CLI] 完成 -> {out_dir}")
    return out_dir


def list_files_kagglehub(dataset: str):
    try:
        import kagglehub
        files = kagglehub.dataset_list_files(dataset)
        print(f"\n[{dataset}] 文件列表:")
        for f in files:
            print(f"  {f['name']} ({f['size']} bytes)" if isinstance(f, dict) else f"  {f}")
    except Exception as e:
        print(f"无法列出文件: {e}")


def list_files_kaggle_cli(dataset: str):
    if not shutil.which("kaggle"):
        print("[错误] 未找到 kaggle CLI。请先 pip install kaggle，或安装 kagglehub。")
        return
    result = subprocess.run(["kaggle", "datasets", "files", dataset], capture_output=True, text=True)
    print(result.stdout or result.stderr)


def check_auth() -> bool:
    if shutil.which("kaggle"):
        if (Path.home() / ".kaggle" / "kaggle.json").exists():
            return True
        if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
            return True
    return False


def _has_kagglehub() -> bool:
    try:
        import kagglehub  # noqa: F401
        return True
    except ImportError:
        return False


def _copy_downloaded_tree(src: Path, dst: Path) -> None:
    """兼容旧版 kagglehub：只能下载到缓存时，把内容同步到目标目录。"""
    if not src.exists():
        return
    if src.is_file():
        shutil.copy2(src, dst / src.name)
        return
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)

# ── 格式检测 ──────────────────────────────────────────────────────


def detect_format(data_dir: Path) -> str:
    """扫描目录，返回 'yolo' | 'coco' | 'voc' | 'classification' | 'unknown'"""
    if not data_dir.exists():
        return "unknown"

    # YOLO: 存在 data.yaml 或 images/labels 目录结构
    if (data_dir / "data.yaml").exists():
        return "yolo"
    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"
    if images_dir.is_dir() and labels_dir.is_dir():
        return "yolo"

    # 递归扫描前 500 个文件判断格式
    all_files = list(data_dir.rglob("*"))
    if not all_files:
        return "unknown"

    sample = all_files[:500]
    has_json = any(f.suffix == ".json" for f in sample)
    has_xml = any(f.suffix == ".xml" for f in sample)

    # COCO: 检查 JSON 是否含 annotations/images 字段
    if has_json:
        for f in sample:
            if f.suffix == ".json" and _is_coco_json(f):
                return "coco"

    # VOC: 检查 XML 是否含 <annotation> 根节点
    if has_xml:
        for f in sample:
            if f.suffix == ".xml" and _is_voc_xml(f):
                return "voc"

    # Classification: 子目录都是类别文件夹 (每类里面有图片)
    subdirs = [d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    if subdirs and all(_has_images(d) for d in subdirs):
        return "classification"

    return "unknown"


def _is_coco_json(path: Path) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return "annotations" in data and "images" in data
    except (json.JSONDecodeError, OSError):
        return False


def _is_voc_xml(path: Path) -> bool:
    try:
        tree = ET.parse(path)
        return tree.getroot().tag == "annotation"
    except (ET.ParseError, OSError):
        return False


def _has_images(directory: Path) -> bool:
    for f in directory.iterdir():
        if f.suffix.lower() in IMAGE_EXTS:
            return True
    return False

# ── 格式转换 ──────────────────────────────────────────────────────


def convert_coco_to_yolo(data_dir: Path) -> dict:
    """COCO JSON → YOLO txt 标签。返回类别映射 {id: name}"""
    json_files = [f for f in data_dir.rglob("*.json") if _is_coco_json(f)]
    if not json_files:
        print("[COCO] 未找到 COCO JSON 标注文件")
        sys.exit(1)

    coco_json = json_files[0]
    print(f"[COCO] 转换 {coco_json}")

    with open(coco_json, "r", encoding="utf-8") as f:
        coco = json.load(f)

    sorted_categories = sorted(coco.get("categories", []), key=lambda cat: cat["id"])
    category_id_to_yolo = {cat["id"]: idx for idx, cat in enumerate(sorted_categories)}
    categories = {idx: cat["name"] for idx, cat in enumerate(sorted_categories)}
    images = {img["id"]: img for img in coco.get("images", [])}
    annotations = coco.get("annotations", [])

    # 按 image_id 分组标注
    ann_by_image: dict[int, list] = defaultdict(list)
    for ann in annotations:
        ann_by_image[ann["image_id"]].append(ann)

    # 确保目录存在
    (data_dir / "images").mkdir(parents=True, exist_ok=True)
    (data_dir / "labels").mkdir(parents=True, exist_ok=True)

    converted = 0
    for img_id, img_info in images.items():
        filename = img_info["file_name"]
        img_w = img_info["width"]
        img_h = img_info["height"]

        src_path = _find_image(data_dir, filename)
        if src_path is None:
            # 检查是否已经在 images/ 下
            candidate = data_dir / "images" / Path(filename).name
            if candidate.exists():
                src_path = candidate

        if src_path is None:
            continue

        dst_img = data_dir / "images" / Path(filename).name
        if src_path != dst_img:
            shutil.move(str(src_path), str(dst_img))

        label_name = Path(filename).stem + ".txt"
        label_path = data_dir / "labels" / label_name

        lines = []
        for ann in ann_by_image[img_id]:
            if ann["category_id"] not in category_id_to_yolo:
                continue
            x, y, w, h = ann["bbox"]  # COCO: 绝对像素 [x, y, w, h]
            cls = category_id_to_yolo[ann["category_id"]]
            cx = (x + w / 2) / img_w
            cy = (y + h / 2) / img_h
            nw = w / img_w
            nh = h / img_h
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        label_path.write_text("\n".join(lines), encoding="utf-8")
        converted += 1

    print(f"[COCO] 转换完成: {converted} 张图片, {len(categories)} 个类别")
    return categories


def convert_voc_to_yolo(data_dir: Path) -> dict:
    """VOC XML → YOLO txt 标签。返回类别映射 {id: name}"""
    xml_files = [f for f in data_dir.rglob("*.xml") if _is_voc_xml(f)]
    if not xml_files:
        print("[VOC] 未找到 VOC XML 标注文件")
        sys.exit(1)

    print(f"[VOC] 转换 {len(xml_files)} 个 XML 文件")

    class_names: dict[str, int] = {}
    next_cls_id = 0

    (data_dir / "images").mkdir(parents=True, exist_ok=True)
    (data_dir / "labels").mkdir(parents=True, exist_ok=True)

    converted = 0
    for xml_path in xml_files:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        filename_el = root.find("filename")
        if filename_el is None or filename_el.text is None:
            continue
        img_filename = filename_el.text.strip()

        size_el = root.find("size")
        if size_el is None:
            continue
        img_w = int(float(size_el.findtext("width", "0")))
        img_h = int(float(size_el.findtext("height", "0")))
        if img_w == 0 or img_h == 0:
            continue

        src_path = _find_image(data_dir, img_filename)
        if src_path is None:
            continue

        dst_img = data_dir / "images" / Path(img_filename).name
        if src_path != dst_img:
            shutil.move(str(src_path), str(dst_img))

        lines = []
        for obj in root.findall("object"):
            name = obj.findtext("name", "")
            if not name:
                continue
            if name not in class_names:
                class_names[name] = next_cls_id
                next_cls_id += 1
            cls = class_names[name]

            bbox = obj.find("bndbox")
            if bbox is None:
                continue
            xmin = float(bbox.findtext("xmin", "0"))
            ymin = float(bbox.findtext("ymin", "0"))
            xmax = float(bbox.findtext("xmax", "0"))
            ymax = float(bbox.findtext("ymax", "0"))

            cx = ((xmin + xmax) / 2) / img_w
            cy = ((ymin + ymax) / 2) / img_h
            nw = (xmax - xmin) / img_w
            nh = (ymax - ymin) / img_h
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        label_name = Path(img_filename).stem + ".txt"
        (data_dir / "labels" / label_name).write_text("\n".join(lines), encoding="utf-8")
        converted += 1

        xml_path.unlink()

    cats = {v: k for k, v in class_names.items()}
    print(f"[VOC] 转换完成: {converted} 张图片, {len(cats)} 个类别")
    return cats


def convert_classification_to_yolo(data_dir: Path) -> dict:
    """分类文件夹 → YOLO 检测格式 (全图作为检测框)"""
    subdirs = sorted(d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith("."))
    if not subdirs:
        print("[分类] 未找到类别文件夹")
        sys.exit(1)

    print(f"[分类] 转换 {len(subdirs)} 个类别文件夹")

    # 需要读取图片尺寸，尝试用 PIL
    try:
        from PIL import Image
        has_pil = True
    except ImportError:
        has_pil = False
        print("[分类] 警告: 未安装 Pillow，假设所有图片为 640×640")

    (data_dir / "images").mkdir(parents=True, exist_ok=True)
    (data_dir / "labels").mkdir(parents=True, exist_ok=True)

    class_names: dict[int, str] = {}
    converted = 0

    for cls_id, subdir in enumerate(subdirs):
        class_names[cls_id] = subdir.name
        for img_path in subdir.iterdir():
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue

            if has_pil:
                try:
                    with Image.open(img_path) as im:
                        w, h = im.size
                except Exception:
                    continue
            else:
                w, h = 640, 640

            dst_img = data_dir / "images" / img_path.name
            shutil.move(str(img_path), str(dst_img))

            label = f"{cls_id} 0.5 0.5 1.0 1.0" if w > 0 and h > 0 else f"{cls_id} 0.5 0.5 1.0 1.0"
            label_path = data_dir / "labels" / (img_path.stem + ".txt")
            label_path.write_text(label, encoding="utf-8")
            converted += 1

    # 清理空文件夹
    for subdir in subdirs:
        try:
            subdir.rmdir()
        except OSError:
            pass

    print(f"[分类] 转换完成: {converted} 张图片, {len(class_names)} 个类别")
    return class_names

# ── 辅助 ──────────────────────────────────────────────────────────


def _find_image(data_dir: Path, filename: str) -> Path | None:
    """在 data_dir 下搜索图片文件"""
    name = Path(filename).name
    direct = data_dir / name
    if direct.exists():
        return direct
    for f in data_dir.rglob(name):
        return f
    stem = Path(filename).stem
    for ext in IMAGE_EXTS:
        candidate = data_dir / (stem + ext)
        if candidate.exists():
            return candidate
    return None

# ── 数据集分割 ────────────────────────────────────────────────────


def split_dataset(data_dir: Path, seed: int = 42, use_test: bool = False):
    """将 images/ 和 labels/ 按 train/val(/test) 分割。

    默认 8:2 (train:val)，use_test=True 时 8:1:1 (train:val:test)
    """
    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"
    if not images_dir.is_dir():
        print("[分割] 未找到 images/ 目录，跳过")
        return

    # 检查是否已经分割
    if (images_dir / "train").exists() or (images_dir / "val").exists():
        print("[分割] 已存在 train/val 目录，跳过分割")
        return

    # 收集图片-标签对
    pairs = []
    for img in images_dir.iterdir():
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        label = labels_dir / (img.stem + ".txt")
        if label.exists():
            pairs.append((img, label))
        else:
            pairs.append((img, None))

    if len(pairs) < 2:
        print(f"[分割] 图片数不足 ({len(pairs)}), 跳过")
        return
    if len(pairs) < 50:
        print(f"[分割] 警告: 仅 {len(pairs)} 张图片，分割可能不理想")

    rng = random.Random(seed)
    rng.shuffle(pairs)
    n = len(pairs)

    if use_test:
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        splits = {
            "train": pairs[:n_train],
            "val": pairs[n_train:n_train + n_val],
            "test": pairs[n_train + n_val:],
        }
    else:
        n_train = int(n * 0.8)
        splits = {
            "train": pairs[:n_train],
            "val": pairs[n_train:],
        }

    for split_name, split_pairs in splits.items():
        split_img = images_dir / split_name
        split_lbl = labels_dir / split_name
        split_img.mkdir(parents=True, exist_ok=True)
        split_lbl.mkdir(parents=True, exist_ok=True)
        for img, lbl in split_pairs:
            shutil.move(str(img), str(split_img / img.name))
            if lbl:
                shutil.move(str(lbl), str(split_lbl / lbl.name))

    sizes = {k: len(v) for k, v in splits.items()}
    print(f"[分割] 完成: {sizes} (seed={seed})")


def generate_data_yaml(data_dir: Path, class_names: dict, use_test: bool = False) -> Path:
    """生成标准 Ultralytics data.yaml"""
    yaml_path = data_dir / "data.yaml"

    names_list = [class_names[k] for k in sorted(class_names)]
    nc = len(names_list)
    train_rel = _image_split_rel(data_dir, "train", fallback="images")
    val_rel = _image_split_rel(data_dir, "val", fallback=train_rel)

    lines = [
        f"path: {data_dir.resolve()}",
        f"train: {train_rel}",
        f"val: {val_rel}",
    ]
    if (data_dir / "images" / "test").exists():
        lines.append("test: images/test")
    lines.append(f"nc: {nc}")
    if nc <= 30:
        lines.append("names:")
        for name in names_list:
            lines.append(f"  - {name}")
    else:
        lines.append(f"names: [{', '.join(repr(n) for n in names_list)}]")

    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[data.yaml] 已生成: {yaml_path} (nc={nc})")
    return yaml_path

# ── 健康检查 ──────────────────────────────────────────────────────


def health_check(data_dir: Path):
    """复用 yolo_data_health.py 的核心逻辑做快速检查"""
    yaml_path = data_dir / "data.yaml"
    if not yaml_path.exists():
        print("[健康检查] 未找到 data.yaml，跳过")
        return

    try:
        import yaml
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except ImportError:
        print("[健康检查] 未安装 pyyaml，跳过")
        return

    print("\n=== 数据集健康检查 ===")
    print(f"路径: {data_dir.resolve()}")
    print(f"类别数: {cfg.get('nc', '?')}")
    names = cfg.get("names", [])
    if isinstance(names, dict):
        names = list(names.values())
    if names:
        print(f"类别: {names}")

    for split in ["train", "val", "test"]:
        if split not in cfg:
            continue
        rel_path = cfg[split]
        split_dir = data_dir / rel_path
        if not split_dir.is_dir():
            print(f"  {split}: 目录不存在 ({split_dir})")
            continue

        images = [f for f in split_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS]
        # 找到对应的 labels 目录
        label_dir = Path(str(split_dir).replace("images", "labels"))
        labels = list(label_dir.glob("*.txt")) if label_dir.is_dir() else []

        missing = sum(1 for img in images if not (label_dir / (img.stem + ".txt")).exists())
        print(f"  {split}: {len(images)} 图片, {len(labels)} 标签, {missing} 缺失标签")

    print("=" * 40)


def print_auth_guide():
    print("""
=== Kaggle API 认证配置指南 ===

方法一：kagglehub（推荐，公开数据集无需认证）
  pip install kagglehub

方法二：kaggle CLI（需要 Kaggle 账号）
  1. 前往 https://www.kaggle.com/settings/account
  2. 点击 "Create New API Token"，下载 kaggle.json
  3. 放到 ~/.kaggle/kaggle.json
  4. pip install kaggle

方法三：环境变量
  export KAGGLE_USERNAME="your_username"
  export KAGGLE_KEY="your_api_key"
""")


def prompt_dataset_input() -> str:
    """交互读取 Kaggle 数据集名、kagglehub 代码片段或文件路径。"""
    print("""
请输入 Kaggle 数据集信息，支持三种形式：
  1) aadityadeth/yolov5-weld
  2) kagglehub.dataset_download("aadityadeth/yolov5-weld")
  3) 保存了上述内容的 .py/.txt 文件路径
单行输入会直接开始；粘贴多行 Python 代码时，粘贴完再按一次空行结束。
""")
    lines = []
    try:
        first_line = input("dataset/code/file> ").strip()
    except EOFError:
        return ""

    if parse_dataset_input(first_line):
        return first_line

    lines.append(first_line)
    print("继续粘贴剩余内容，空行结束。")
    while True:
        try:
            line = input("... ")
        except EOFError:
            break
        if not line.strip():
            break
        lines.append(line)
    return "\n".join(lines).strip()


def parse_dataset_input(raw: str) -> str:
    """从数据集名、Kaggle URL、kagglehub 代码片段或文件中提取 owner/slug。"""
    text = raw.strip().strip("\"'")
    if not text:
        return ""

    path = Path(text).expanduser()
    if path.exists() and path.is_file():
        text = path.read_text(encoding="utf-8").strip()

    call_match = KAGGLEHUB_CALL_RE.search(text)
    if call_match:
        text = call_match.group("dataset")

    url_match = KAGGLE_URL_RE.search(text)
    if url_match:
        return f"{url_match.group('owner')}/{url_match.group('slug')}"

    ref_match = DATASET_REF_RE.search(text)
    if ref_match:
        return f"{ref_match.group('owner')}/{ref_match.group('slug')}"

    return ""


# ── 主入口 ────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Kaggle 下载 + YOLO 转换 + 分割")
    parser.add_argument("dataset", nargs="?", help="Kaggle 数据集标识符 (例如 ultralytics/coco8)")
    parser.add_argument("--path", "--out", dest="out_dir", type=Path, default=None,
                        help="下载目标路径（默认: data/<dataset-name>）")
    parser.add_argument("--list", action="store_true", help="仅列出数据集文件")
    parser.add_argument("--method", choices=["kagglehub", "kaggle", "auto"], default="auto",
                        help="下载方式")
    parser.add_argument("--setup-auth", action="store_true", help="输出认证指南")

    # 转换选项
    parser.add_argument("--convert", action="store_true", default=True,
                        help="自动转换格式 (默认)")
    parser.add_argument("--no-convert", action="store_false", dest="convert",
                        help="不转换格式")
    parser.add_argument("--force", action="store_true",
                        help="强制覆盖已有 data.yaml")

    # 分割选项
    parser.add_argument("--split", action="store_true", default=True,
                        help="自动分割数据集 (默认)")
    parser.add_argument("--no-split", action="store_false", dest="split",
                        help="不分割")
    parser.add_argument("--split-test", action="store_true",
                        help="使用 8:1:1 (train/val/test)，默认 8:2")
    parser.add_argument("--split-seed", type=int, default=42,
                        help="分割随机种子 (默认 42)")

    # 检查选项
    parser.add_argument("--health-check", action="store_true",
                        help="下载/转换后运行数据健康检查")

    args = parser.parse_args()

    if args.setup_auth:
        print_auth_guide()
        return

    if not args.dataset:
        args.dataset = prompt_dataset_input()

    args.dataset = parse_dataset_input(args.dataset)

    if not args.dataset:
        parser.error("无法识别 Kaggle 数据集名；可输入 aadityadeth/yolov5-weld 或 kagglehub.dataset_download(...)")

    if args.list:
        if args.method == "kagglehub" or (args.method == "auto" and _has_kagglehub()):
            list_files_kagglehub(args.dataset)
        else:
            list_files_kaggle_cli(args.dataset)
        return

    # 确定输出路径
    if args.out_dir is None:
        dataset_name = args.dataset.split("/")[-1]
        args.out_dir = Path("data") / dataset_name

    # ── Step 1: 下载 ──
    if (args.out_dir / "data.yaml").exists() and not args.force:
        print(f"[跳过下载] {args.out_dir} 已存在 data.yaml，使用 --force 强制重新下载")
    else:
        if args.method == "kaggle":
            if not check_auth():
                print("[错误] --method=kaggle 需配置 ~/.kaggle/kaggle.json")
                sys.exit(1)
            download_kaggle_cli(args.dataset, args.out_dir)
        elif args.method == "kagglehub" or _has_kagglehub():
            download_kagglehub(args.dataset, args.out_dir)
        elif check_auth():
            download_kaggle_cli(args.dataset, args.out_dir)
        else:
            print("[错误] 无可用下载方式。pip install kagglehub 或配置 kaggle API")
            sys.exit(1)

    # ── Step 2: 检测格式 ──
    fmt = detect_format(args.out_dir)
    print(f"\n[检测] 数据集格式: {fmt}")

    # ── Step 3: 转换 ──
    class_names: dict = {}

    if fmt == "yolo":
        # 尝试从已有 data.yaml 读取类别名
        yaml_path = args.out_dir / "data.yaml"
        if yaml_path.exists():
            try:
                import yaml
                cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
                names = cfg.get("names", [])
                if isinstance(names, dict):
                    class_names = {int(k): v for k, v in names.items()}
                elif isinstance(names, list):
                    class_names = dict(enumerate(names))
            except Exception:
                pass

    elif fmt == "coco" and args.convert:
        class_names = convert_coco_to_yolo(args.out_dir)

    elif fmt == "voc" and args.convert:
        class_names = convert_voc_to_yolo(args.out_dir)

    elif fmt == "classification" and args.convert:
        class_names = convert_classification_to_yolo(args.out_dir)

    elif fmt == "unknown":
        print("[警告] 无法识别数据集格式，尝试按图片目录处理")
        images = _collect_all_images(args.out_dir)
        if images:
            (args.out_dir / "images").mkdir(parents=True, exist_ok=True)
            for img in images:
                dst = args.out_dir / "images" / img.name
                if img.parent != args.out_dir / "images":
                    shutil.move(str(img), str(dst))
            (args.out_dir / "labels").mkdir(parents=True, exist_ok=True)
        else:
            print("[错误] 未找到任何图片文件")
            sys.exit(1)

    if not args.convert and fmt != "yolo":
        print(f"[跳过转换] 格式为 {fmt}，已禁用自动转换")

    # ── Step 4: 分割 ──
    if args.split:
        has_splits = ((args.out_dir / "images" / "train").exists() or
                       (args.out_dir / "images" / "val").exists())
        if has_splits:
            print("[分割] 已存在 train/val 子目录，跳过")
        else:
            split_dataset(args.out_dir, seed=args.split_seed, use_test=args.split_test)

    # ── Step 5: 生成 data.yaml ──
    yaml_path = args.out_dir / "data.yaml"
    if not yaml_path.exists() or args.force:
        if not class_names:
            class_names = _infer_names(args.out_dir)
        if class_names:
            generate_data_yaml(args.out_dir, class_names, use_test=args.split_test)
        else:
            print("[警告] 无法推断类别名，请手动编辑 data.yaml")
            # 至少写一个最小可用的 data.yaml
            _write_minimal_yaml(args.out_dir)

    # ── Step 6: 健康检查 ──
    if args.health_check:
        health_check(args.out_dir)

    print(f"\n✓ 完成: {args.out_dir.resolve()}")
    if (args.out_dir / "data.yaml").exists():
        print(f"  data.yaml: {args.out_dir / 'data.yaml'}")
        print(f"  训练命令: python train_yolo.py train --data-yaml {args.out_dir / 'data.yaml'} --task-name <name>")


def _collect_all_images(data_dir: Path) -> list[Path]:
    return [f for f in data_dir.rglob("*") if f.suffix.lower() in IMAGE_EXTS]


def _infer_names(data_dir: Path) -> dict[int, str]:
    """从 labels/ 目录推断类别名"""
    labels_dir = data_dir / "labels"
    if not labels_dir.is_dir():
        return {}
    class_ids: set[int] = set()
    for label in labels_dir.rglob("*.txt"):
        try:
            for line in label.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                cls = int(line.strip().split()[0])
                class_ids.add(cls)
        except Exception:
            continue
    return {cid: f"class_{cid}" for cid in sorted(class_ids)}


def _write_minimal_yaml(data_dir: Path):
    images = _collect_all_images(data_dir / "images")
    nc = 1 if not images else 1
    train_rel = _image_split_rel(data_dir, "train", fallback="images")
    val_rel = _image_split_rel(data_dir, "val", fallback=train_rel)
    has_test = (data_dir / "images" / "test").exists()
    lines = [
        f"path: {data_dir.resolve()}",
        f"train: {train_rel}",
        f"val: {val_rel}",
    ]
    if has_test:
        lines.append("test: images/test")
    lines.append(f"nc: {nc}")
    lines.append("names: [class_0]")
    (data_dir / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _image_split_rel(data_dir: Path, split: str, fallback: str) -> str:
    rel = Path("images") / split
    if (data_dir / rel).is_dir():
        return rel.as_posix()
    return fallback


if __name__ == "__main__":
    main()
