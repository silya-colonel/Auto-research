from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from clearml import Dataset, Task


def local_copy(dataset_id: str) -> Path:
    dataset = Dataset.get(dataset_id=dataset_id)
    return Path(dataset.get_local_copy()).resolve()


def find_file(root: Path, name: str) -> Path:
    direct = root / name
    if direct.exists():
        return direct
    matches = sorted(root.glob(f"**/{name}"))
    if not matches:
        raise SystemExit(f"{name} not found under {root}")
    return matches[0]


def prepare_remote_data_yaml(data_root: Path) -> Path:
    try:
        data_yaml = find_file(data_root, "data.yaml")
    except SystemExit:
        # data.yaml not in dataset — generate one from known templates
        data_yaml = _generate_data_yaml(data_root)
    lines = data_yaml.read_text(encoding="utf-8").splitlines()
    rewritten: list[str] = []
    has_path = False
    for line in lines:
        if line.strip().startswith("path:"):
            rewritten.append(f"path: {data_root}")
            has_path = True
        else:
            rewritten.append(line)
    if not has_path:
        rewritten.insert(0, f"path: {data_root}")
    out = data_root / "data.remote.yaml"
    out.write_text("\n".join(rewritten) + "\n", encoding="utf-8")
    return out


_STEEL_DEFECT_MIXED_YAML = """\
path: .
train: images/train
val: images/val
nc: 10
names:
  - inclusions
  - dents
  - oil_spots
  - pits
  - punching
  - linear
  - macular
  - welding_seams
  - water_spots
  - scratches
"""


def _generate_data_yaml(data_root: Path) -> Path:
    """Generate a data.yaml when the dataset doesn't include one."""
    out = data_root / "data.yaml"

    # Check for Steel-Defect-Mixed dataset by looking for characteristic content
    steel_mixed_marker = data_root / "steel-defect-mixed" / "README.md"
    if (data_root / "steel-defect-mixed").is_dir() or steel_mixed_marker.exists():
        out.write_text(_STEEL_DEFECT_MIXED_YAML, encoding="utf-8")
        return out

    # Generic fallback: try to infer from directory structure
    train_dir = None
    val_dir = None
    for candidate in ["images/train", "train/images", "train"]:
        if (data_root / candidate).is_dir():
            train_dir = candidate
            break
    for candidate in ["images/val", "valid/images", "val", "valid"]:
        if (data_root / candidate).is_dir():
            val_dir = candidate
            break

    if train_dir and val_dir:
        # Count classes from label files (max class index in any label file)
        label_dir = data_root / "labels" / "train"
        if not label_dir.is_dir():
            label_dirs = sorted(data_root.glob("**/labels/train"))
            label_dir = label_dirs[0] if label_dirs else None
        nc = 1
        if label_dir and label_dir.is_dir():
            max_cls = 0
            for lbl in label_dir.glob("*.txt"):
                try:
                    for line in lbl.read_text().strip().splitlines():
                        cls_id = int(line.split()[0])
                        max_cls = max(max_cls, cls_id)
                except (ValueError, IndexError):
                    pass
            nc = max_cls + 1
        yaml_content = f"path: .\ntrain: {train_dir}\nval: {val_dir}\nnc: {nc}\nnames:\n"
        for i in range(nc):
            yaml_content += f"  - class_{i}\n"
        out.write_text(yaml_content, encoding="utf-8")
        return out

    raise SystemExit(f"Cannot generate data.yaml: no recognizable dataset structure under {data_root}")


def resolve_packaged_path(code_root: Path, value: str | None) -> str | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute() and path.exists():
        return str(path)
    packaged = code_root / value
    if packaged.exists():
        return str(packaged.resolve())
    matches = sorted(code_root.glob(f"**/{value}"))
    if matches:
        return str(matches[0].resolve())
    return value


def upload_artifact_if_exists(task: Task | None, name: str, path: Path) -> None:
    if task is None or not path.exists():
        return
    print(f"uploading artifact {name}: {path}")
    task.upload_artifact(name=name, artifact_object=str(path))


def upload_run_artifacts(task: Task | None, code_root: Path, task_name: str, runs_dir: str) -> None:
    if task is None:
        return
    run_dir = (code_root / runs_dir / task_name).resolve()
    val_dir = (code_root / "runs" / "val" / task_name).resolve()

    upload_artifact_if_exists(task, "metrics.json", run_dir / "metrics.json")
    upload_artifact_if_exists(task, "results.csv", run_dir / "results.csv")
    upload_artifact_if_exists(task, "best.pt", run_dir / "weights" / "best.pt")
    upload_artifact_if_exists(task, "last.pt", run_dir / "weights" / "last.pt")

    if run_dir.exists():
        archive_base = run_dir.parent / f"{task_name}_run"
        archive_path = Path(shutil.make_archive(str(archive_base), "zip", root_dir=str(run_dir)))
        upload_artifact_if_exists(task, "run_dir.zip", archive_path)
    if val_dir.exists():
        archive_base = val_dir.parent / f"{task_name}_val"
        archive_path = Path(shutil.make_archive(str(archive_base), "zip", root_dir=str(val_dir)))
        upload_artifact_if_exists(task, "val_dir.zip", archive_path)


def assert_cuda_available(require_cuda: bool) -> None:
    if not require_cuda:
        return
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("CUDA preflight failed: torch is not installed") from exc
    if not torch.cuda.is_available():
        raise SystemExit("CUDA preflight failed: torch.cuda.is_available() is false; refusing to run YOLO training on CPU")
    print(f"CUDA preflight ok: {torch.cuda.get_device_name(0)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-dataset-id", required=True)
    parser.add_argument("--data-dataset-id", required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--imgsz", default="640")
    parser.add_argument("--epochs", default="3")
    parser.add_argument("--batch", default="-1")
    parser.add_argument("--workers", default="4")
    parser.add_argument("--seed", default="42")
    parser.add_argument("--device", default="0")
    parser.add_argument("--runs-dir", default="runs/yolo")
    parser.add_argument("--pretrained-weights", default=None)
    parser.add_argument("--extra", nargs="*", default=[])
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--clearml-remote", action="store_true")
    parser.add_argument("--clearml-project", default="yolo-welding-defect")
    parser.add_argument("--clearml-queue", default="gpu-any")
    parser.add_argument("--remote-python-binary", default="python3.12")
    args = parser.parse_args()

    if args.clearml_remote:
        Task.force_store_standalone_script(True)
        task = Task.init(project_name=args.clearml_project, task_name=f"{args.task_name}_bundle")
        task.set_script(binary=args.remote_python_binary)
        task.set_packages(
            [
                "clearml",
                "numpy<2",
                "torch==2.4.1",
                "torchvision==0.19.1",
                "ultralytics==8.4.52",
                "pyyaml",
                "kagglehub",
            ]
        )
        task.execute_remotely(queue_name=args.clearml_queue, clone=False, exit_process=True)

    task = Task.current_task()
    if task:
        task.connect(vars(args), name="launcher")

    code_root = local_copy(args.code_dataset_id)
    data_root = local_copy(args.data_dataset_id)
    assert_cuda_available(not args.allow_cpu)
    train_script = find_file(code_root, "train_yolo.py")
    data_yaml = prepare_remote_data_yaml(data_root)
    model_path = resolve_packaged_path(code_root, args.model)
    pretrained_weights = resolve_packaged_path(code_root, args.pretrained_weights)

    command = [
        sys.executable,
        str(train_script),
        "train",
        "--task-name",
        args.task_name,
        "--data-yaml",
        str(data_yaml),
        "--model",
        model_path or args.model,
        "--imgsz",
        str(args.imgsz),
        "--epochs",
        str(args.epochs),
        "--batch",
        str(args.batch),
        "--device",
        str(args.device),
        "--workers",
        str(args.workers),
        "--seed",
        str(args.seed),
        "--runs-dir",
        args.runs_dir,
        "--disable-clearml",
    ]
    if pretrained_weights:
        command.extend(["--pretrained-weights", pretrained_weights])
    if args.extra:
        command.append("--extra")
        command.extend(args.extra)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(code_root) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(command, cwd=str(code_root), env=env, check=True, timeout=86400)
    upload_run_artifacts(task, code_root, args.task_name, args.runs_dir)


if __name__ == "__main__":
    main()
