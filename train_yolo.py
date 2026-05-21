
from __future__ import annotations

import argparse
import os
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from tools.clearml_remote import maybe_execute_remotely


PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_CACHE_ROOT = PROJECT_ROOT / ".cache"
os.environ.setdefault("MPLCONFIGDIR", str(LOCAL_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(LOCAL_CACHE_ROOT / "ultralytics"))
os.environ.setdefault("XDG_CACHE_HOME", str(LOCAL_CACHE_ROOT / "xdg"))
for cache_dir in (Path(os.environ["MPLCONFIGDIR"]), Path(os.environ["YOLO_CONFIG_DIR"]), Path(os.environ["XDG_CACHE_HOME"])):
    cache_dir.mkdir(parents=True, exist_ok=True)


def load_yolo():
    from ultralytics import YOLO

    return YOLO


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YOLO train / val / export / predict")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train a YOLO model")
    train.add_argument("--task-name", required=True)
    train.add_argument("--data-yaml", required=True)
    train.add_argument("--clearml-dataset-id", default=None, help="Optional ClearML Dataset ID to materialize before training")
    train.add_argument("--clearml-dataset-project", default=None, help="Optional ClearML Dataset project for name/version lookup")
    train.add_argument("--clearml-dataset-name", default=None, help="Optional ClearML Dataset name for remote materialization")
    train.add_argument("--clearml-dataset-version", default=None, help="Optional ClearML Dataset version for remote materialization")
    train.add_argument("--model", default="yolo11n.pt")
    train.add_argument("--pretrained-weights", default=None, help="Optional weights to load before training a YAML model")
    train.add_argument("--imgsz", type=int, default=640)
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--batch", default="-1")
    train.add_argument("--device", default=None)
    train.add_argument("--workers", type=int, default=8)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--runs-dir", default="runs/yolo")
    train.add_argument("--resume", action="store_true")
    train.add_argument("--max-retries", type=int, default=2)
    train.add_argument("--enable-clearml", action="store_true", help="Enable Ultralytics ClearML integration")
    train.add_argument("--disable-clearml", action="store_true", help="Disable Ultralytics ClearML integration")
    train.add_argument("--clearml-remote", action="store_true", help="Submit this training task to a ClearML queue and exit locally")
    train.add_argument("--clearml-queue", default="gpu-any", help="ClearML queue name for remote execution")
    train.add_argument("--clearml-project", default="yolo-welding-defect", help="ClearML project name for remote execution")
    train.add_argument("--enable-mlflow", action="store_true", help="Enable Ultralytics MLflow integration")
    train.add_argument("--extra", nargs="*", default=[], help="Extra YOLO args in key=value form")

    val = sub.add_parser("val", help="Validate a YOLO model")
    val.add_argument("--weights", required=True)
    val.add_argument("--data-yaml", required=True)
    val.add_argument("--clearml-dataset-id", default=None, help="Optional ClearML Dataset ID to materialize before validation")
    val.add_argument("--clearml-dataset-project", default=None, help="Optional ClearML Dataset project for name/version lookup")
    val.add_argument("--clearml-dataset-name", default=None, help="Optional ClearML Dataset name for remote materialization")
    val.add_argument("--clearml-dataset-version", default=None, help="Optional ClearML Dataset version for remote materialization")
    val.add_argument("--imgsz", type=int, default=640)
    val.add_argument("--batch", type=int, default=16)
    val.add_argument("--device", default=None)
    val.add_argument("--save-json", action="store_true", help="Save COCO-format results")
    val.add_argument("--output-dir", default="runs/val")

    export = sub.add_parser("export", help="Export a YOLO model")
    export.add_argument("--weights", required=True)
    export.add_argument("--format", default="onnx", choices=["onnx", "engine", "tflite", "coreml"])
    export.add_argument("--imgsz", type=int, default=640)
    export.add_argument("--half", action="store_true", help="FP16 export")
    export.add_argument("--dynamic", action="store_true", help="Dynamic batch export")

    predict = sub.add_parser("predict", help="Run inference")
    predict.add_argument("--weights", required=True)
    predict.add_argument("--source", required=True)
    predict.add_argument("--imgsz", type=int, default=640)
    predict.add_argument("--conf", type=float, default=0.25)
    predict.add_argument("--device", default=None)
    predict.add_argument("--save", action="store_true", default=True)
    predict.add_argument("--output-dir", default="runs/predict")

    return parser


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def prepare_data_yaml_for_ultralytics(data_yaml: Path) -> Path:
    """Ultralytics may resolve `path: .` against cwd; rewrite it to the dataset dir."""
    text = data_yaml.read_text(encoding="utf-8")
    lines = text.splitlines()
    changed = False
    rewritten: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("path:"):
            value = stripped.split(":", 1)[1].strip().strip("'\"")
            if value in {"", "."}:
                rewritten.append(f"path: {data_yaml.parent}")
                changed = True
                continue
        rewritten.append(line)
    if not changed:
        return data_yaml
    out_dir = PROJECT_ROOT / "runs" / "prepared_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{data_yaml.parent.name}.yaml"
    out_path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")
    return out_path


def maybe_materialize_clearml_dataset(args: argparse.Namespace) -> Path | None:
    dataset_id = getattr(args, "clearml_dataset_id", None)
    dataset_name = getattr(args, "clearml_dataset_name", None)
    if not dataset_id and not dataset_name:
        return None

    from clearml import Dataset

    if dataset_id:
        dataset = Dataset.get(dataset_id=dataset_id)
    else:
        dataset = Dataset.get(
            dataset_project=getattr(args, "clearml_dataset_project", None),
            dataset_name=dataset_name,
            dataset_version=getattr(args, "clearml_dataset_version", None),
        )
    local_copy = Path(dataset.get_local_copy()).resolve()
    print(f"ClearML dataset materialized: {local_copy}")
    return local_copy


def resolve_data_yaml(args: argparse.Namespace) -> Path:
    dataset_root = maybe_materialize_clearml_dataset(args)
    requested = Path(args.data_yaml)
    if dataset_root is not None:
        candidates = [
            dataset_root / requested.name,
            dataset_root / requested,
            dataset_root / "data.yaml",
        ]
        for candidate in candidates:
            if candidate.exists():
                return prepare_data_yaml_for_ultralytics(candidate.resolve())
        raise SystemExit(f"data.yaml not found in ClearML dataset copy: {dataset_root}")

    data_yaml = resolve_project_path(requested)
    if not data_yaml.exists():
        sys.exit(f"data.yaml not found: {data_yaml}")
    return prepare_data_yaml_for_ultralytics(data_yaml)


def coerce_batch(value: str) -> int | float:
    try:
        return int(value)
    except ValueError:
        return float(value)


def parse_extra_value(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def parse_extra(items: list[str]) -> dict[str, Any]:
    extra: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--extra expects key=value, got: {item}")
        key, value = item.split("=", 1)
        extra[key] = parse_extra_value(value)
    return extra


def configure_ultralytics_integrations(enable_clearml: bool = True, enable_mlflow: bool = False) -> None:
    from ultralytics import settings

    settings.update({
        "clearml": bool(enable_clearml),
        "mlflow": bool(enable_mlflow),
    })


def notify(title: str, message: str) -> None:
    print(f"[{title}] {message}")


def save_metrics(save_dir: Path, task_name: str, metrics: dict[str, Any]) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "task_name": task_name,
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics,
    }
    metrics_file = save_dir / "metrics.json"
    metrics_file.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"metrics saved: {metrics_file}")


def collect_per_class_metrics(val_results: Any) -> dict[str, dict[str, float]]:
    per_class: dict[str, dict[str, float]] = {}
    names = getattr(val_results, "names", {}) or {}
    box = getattr(val_results, "box", None)
    if box is None or not hasattr(box, "class_result"):
        return per_class
    for class_id, class_name in names.items():
        try:
            precision, recall, map50, map5095 = box.class_result(int(class_id))
        except Exception:
            continue
        per_class[str(class_name)] = {
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "mAP50": round(float(map50), 4),
            "mAP50-95": round(float(map5095), 4),
        }
    return per_class


def cmd_train(args: argparse.Namespace) -> None:
    YOLO = load_yolo()
    from tools.yolo_custom_modules import (
        patch_detection_cls_loss,
        patch_detection_iou_loss,
        register_yolo_modules,
    )

    register_yolo_modules()
    clearml_enabled = (args.enable_clearml or args.clearml_remote) and not args.disable_clearml
    configure_ultralytics_integrations(clearml_enabled, args.enable_mlflow)

    maybe_execute_remotely(
        enabled=args.clearml_remote,
        project_name=args.clearml_project,
        task_name=args.task_name,
        queue_name=args.clearml_queue,
        clone=False,
        exit_process=True,
    )

    data_yaml = resolve_data_yaml(args)

    runs_dir = resolve_project_path(args.runs_dir)
    extra = parse_extra(args.extra)
    custom_iou_loss = extra.pop("custom_iou_loss", None)
    custom_cls_loss = extra.pop("custom_cls_loss", None)
    focal_gamma = float(extra.pop("focal_gamma", 1.5))
    focal_alpha = float(extra.pop("focal_alpha", 0.25))
    sahb_scale_weight = float(extra.pop("sahb_scale_weight", 1.0))
    sahb_hard_bg_weight = float(extra.pop("sahb_hard_bg_weight", 0.0))
    skip_final_val = bool(extra.pop("skip_final_val", False))
    if (
        custom_iou_loss
        and str(custom_iou_loss).lower() == "sahb"
        and sahb_hard_bg_weight > 0
        and custom_cls_loss is None
    ):
        custom_cls_loss = "focal"
        focal_gamma = max(focal_gamma, 1.5 + sahb_hard_bg_weight)
    patch_detection_iou_loss(
        str(custom_iou_loss) if custom_iou_loss else None,
        scale_weight=sahb_scale_weight,
        hard_bg_weight=sahb_hard_bg_weight,
    )
    patch_detection_cls_loss(str(custom_cls_loss) if custom_cls_loss else None, gamma=focal_gamma, alpha=focal_alpha)

    model = YOLO(args.model)

    if args.resume:
        last_pt = runs_dir / args.task_name / "weights" / "last.pt"
        if last_pt.exists():
            print(f"resuming from: {last_pt}")
            model = YOLO(str(last_pt))
        else:
            print(f"resume checkpoint not found, starting fresh: {last_pt}")
    elif args.pretrained_weights:
        pretrained_weights = resolve_project_path(args.pretrained_weights)
        if not pretrained_weights.exists():
            sys.exit(f"pretrained weights not found: {pretrained_weights}")
        print(f"loading pretrained weights: {pretrained_weights}")
        model.load(str(pretrained_weights))

    kwargs: dict[str, Any] = {
        "data": str(data_yaml),
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "batch": coerce_batch(args.batch),
        "workers": args.workers,
        "seed": args.seed,
        "project": str(runs_dir),
        "name": args.task_name,
    }
    if args.device:
        kwargs["device"] = args.device
    kwargs.update(extra)

    results = None
    for attempt in range(1, args.max_retries + 2):
        try:
            print(f"\n-- {args.task_name} (attempt {attempt}/{args.max_retries + 1}) --")
            results = model.train(**kwargs)
            break
        except Exception as exc:
            print(f"training failed (attempt {attempt}): {exc}")
            if attempt > args.max_retries:
                notify("training failed", f"{args.task_name}: {exc}")
                sys.exit(1)
            time.sleep(30)

    save_dir = Path(getattr(results, "save_dir", runs_dir / args.task_name)).resolve()
    metrics: dict[str, Any] = {}

    train_map50 = getattr(getattr(results, "box", None), "map50", None)
    if train_map50 is not None:
        metrics["mAP50"] = round(float(train_map50), 4)

    if skip_final_val:
        print("final validation skipped by --extra skip_final_val=True")
    else:
        try:
            val_results = model.val(
                data=str(data_yaml),
                imgsz=args.imgsz,
                batch=16,
                device=args.device,
                project=str(resolve_project_path("runs/val")),
                name=args.task_name,
            )
            metrics.update({
                "mAP50": round(float(val_results.box.map50), 4),
                "mAP50-95": round(float(val_results.box.map), 4),
                "precision": round(float(val_results.box.mp), 4),
                "recall": round(float(val_results.box.mr), 4),
                "per_class": collect_per_class_metrics(val_results),
            })
            print(f"mAP50: {val_results.box.map50:.4f}  mAP50-95: {val_results.box.map:.4f}")
        except Exception as exc:
            print(f"validation failed: {exc}")

    save_metrics(save_dir, args.task_name, metrics)

    message = f"{args.task_name}: done (mAP50={metrics.get('mAP50', 'N/A')})"
    print(message)
    notify("training complete", message)


def cmd_val(args: argparse.Namespace) -> None:
    YOLO = load_yolo()
    data_yaml = resolve_data_yaml(args)
    output_dir = resolve_project_path(args.output_dir)
    model = YOLO(str(resolve_project_path(args.weights)))
    results = model.val(
        data=str(data_yaml),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        save_json=args.save_json,
        project=str(output_dir),
    )
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    print(f"Precision: {results.box.mp:.4f}")
    print(f"Recall: {results.box.mr:.4f}")


def cmd_export(args: argparse.Namespace) -> None:
    YOLO = load_yolo()
    model = YOLO(str(resolve_project_path(args.weights)))
    model.export(format=args.format, imgsz=args.imgsz, half=args.half, dynamic=args.dynamic)
    print(f"export complete: {args.weights}")


def cmd_predict(args: argparse.Namespace) -> None:
    YOLO = load_yolo()
    model = YOLO(str(resolve_project_path(args.weights)))
    results = model.predict(
        source=str(resolve_project_path(args.source)),
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        save=args.save,
        project=str(resolve_project_path(args.output_dir)),
    )
    for result in results:
        print(f"{result.path} -> {len(result.boxes)} detections")


COMMANDS = {
    "train": cmd_train,
    "val": cmd_val,
    "export": cmd_export,
    "predict": cmd_predict,
}


if __name__ == "__main__":
    parsed_args = build_parser().parse_args()
    COMMANDS[parsed_args.command](parsed_args)
