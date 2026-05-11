"""YOLO11 训练/评估/导出 — ClearML 集成

用法:
  # 训练
  python train_yolo_clearml.py train --task-name baseline_yolo11n_640 \
    --data-yaml ~/datasets/defect/data.yaml --model yolo11n.pt --epochs 100

  # 断点续训
  python train_yolo_clearml.py train --task-name baseline_yolo11n_640 --resume \
    --data-yaml ~/datasets/defect/data.yaml

  # 评估已有模型
  python train_yolo_clearml.py val --weights runs/yolo/baseline/weights/best.pt \
    --data-yaml ~/datasets/defect/data.yaml

  # 导出 ONNX/TensorRT
  python train_yolo_clearml.py export --weights runs/yolo/baseline/weights/best.pt \
    --format onnx

  # 推理单张图片
  python train_yolo_clearml.py predict --weights runs/yolo/baseline/weights/best.pt \
    --source ~/test.jpg --imgsz 640

  # 训练完成后自动关机(云GPU)
  python train_yolo_clearml.py train --task-name exp_v2 \
    --data-yaml ~/datasets/defect/data.yaml --model yolo11n.pt --shutdown-when-done
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from clearml import Task, Logger
from ultralytics import YOLO


# ── CLI ──────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="YOLO11 + ClearML 训练/评估/导出/推理")
    sub = p.add_subparsers(dest="command", required=True)

    # train
    t = sub.add_parser("train", help="训练模型")
    t.add_argument("--project-name", default="yolo")
    t.add_argument("--task-name", required=True)
    t.add_argument("--data-yaml", required=True)
    t.add_argument("--model", default="yolo11n.pt")
    t.add_argument("--imgsz", type=int, default=640)
    t.add_argument("--epochs", type=int, default=100)
    t.add_argument("--batch", default="-1")
    t.add_argument("--device", default=None)
    t.add_argument("--workers", type=int, default=8)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--runs-dir", default="runs/yolo")
    t.add_argument("--resume", action="store_true")
    t.add_argument("--shutdown-when-done", action="store_true")
    t.add_argument("--max-retries", type=int, default=2)
    t.add_argument("--extra", nargs="*", default=[])

    # val
    v = sub.add_parser("val", help="评估模型")
    v.add_argument("--weights", required=True)
    v.add_argument("--data-yaml", required=True)
    v.add_argument("--imgsz", type=int, default=640)
    v.add_argument("--batch", type=int, default=16)
    v.add_argument("--device", default=None)
    v.add_argument("--save-json", action="store_true", help="保存COCO格式结果")
    v.add_argument("--task-name", default=None, help="ClearML task name (可选)")

    # export
    e = sub.add_parser("export", help="导出模型")
    e.add_argument("--weights", required=True)
    e.add_argument("--format", default="onnx", choices=["onnx", "engine", "tflite", "coreml"])
    e.add_argument("--imgsz", type=int, default=640)
    e.add_argument("--half", action="store_true", help="FP16")
    e.add_argument("--dynamic", action="store_true", help="动态batch")

    # predict
    pr = sub.add_parser("predict", help="推理")
    pr.add_argument("--weights", required=True)
    pr.add_argument("--source", required=True)
    pr.add_argument("--imgsz", type=int, default=640)
    pr.add_argument("--conf", type=float, default=0.25)
    pr.add_argument("--device", default=None)
    pr.add_argument("--save", action="store_true", default=True)
    pr.add_argument("--output-dir", default="runs/predict")

    return p


# ── Helpers ──────────────────────────────────────────────────────

def coerce_batch(v: str):
    try:
        return int(v)
    except ValueError:
        return float(v)


def parse_extra(items: list[str]) -> dict:
    d = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"extra 参数需要 key=value 格式: {item}")
        k, val = item.split("=", 1)
        d[k] = val
    return d


def notify(title: str, message: str):
    if sys.platform == "darwin":
        subprocess.run([
            "osascript", "-e",
            f'display notification "{message}" with title "{title}"'
        ], capture_output=True)
    elif sys.platform == "linux":
        subprocess.run(["notify-send", title, message], capture_output=True)


def do_shutdown():
    subprocess.run(["wall", "训练任务完成，系统将在1分钟后关机"], capture_output=True)
    subprocess.run(["sync"], capture_output=True)
    time.sleep(60)
    subprocess.run(["sudo", "shutdown", "-h", "now"])


# ── Commands ─────────────────────────────────────────────────────

def cmd_train(args):
    data_yaml = Path(args.data_yaml)
    if not data_yaml.exists():
        sys.exit(f"data.yaml 不存在: {data_yaml}")

    task = Task.init(project_name=args.project_name, task_name=args.task_name)
    task.connect(vars(args))

    model = YOLO(args.model)

    if args.resume:
        last_pt = Path(args.runs_dir) / args.task_name / "weights" / "last.pt"
        if last_pt.exists():
            print(f"→ 续训: {last_pt}")
            model = YOLO(str(last_pt))
        else:
            print(f"→ 未找到 {last_pt}，从头训练")

    kwargs = {
        "data": str(data_yaml),
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "batch": coerce_batch(args.batch),
        "workers": args.workers,
        "seed": args.seed,
        "project": args.runs_dir,
        "name": args.task_name,
    }
    if args.device:
        kwargs["device"] = args.device
    kwargs.update(parse_extra(args.extra))

    for attempt in range(1, args.max_retries + 2):
        try:
            print(f"\n── {args.task_name} (attempt {attempt}/{args.max_retries + 1}) ──")
            results = model.train(**kwargs)
            break
        except Exception as e:
            print(f"训练失败 (attempt {attempt}): {e}")
            if attempt > args.max_retries:
                notify("训练失败", f"{args.task_name}: {e}")
                task.close()
                sys.exit(1)
            time.sleep(30)

    save_dir = getattr(results, "save_dir", None) or str(
        Path(args.runs_dir) / args.task_name
    )
    task.upload_artifact("ultralytics_results", artifact_object=save_dir)

    try:
        val_results = model.val()
        metrics = {
            "mAP50": float(val_results.box.map50),
            "mAP50-95": float(val_results.box.map),
            "precision": float(val_results.box.mp),
            "recall": float(val_results.box.mr),
        }
        for k, v in metrics.items():
            Logger.current_logger().report_scalar("val", k, iteration=0, value=v)
        task.upload_artifact("best_weights", artifact_object=f"{save_dir}/weights/best.pt")
    except Exception:
        pass

    mAP = getattr(getattr(results, "box", None), "map50", None)
    msg = f"{args.task_name}: 完成 (mAP50={mAP:.3f})" if mAP else f"{args.task_name}: 完成"
    print(msg)
    notify("训练完成", msg)
    task.close()

    if args.shutdown_when_done:
        do_shutdown() if os.geteuid() == 0 else subprocess.run(
            ["sudo", "shutdown", "-h", "+2"], capture_output=True)


def cmd_val(args):
    model = YOLO(args.weights)
    results = model.val(
        data=args.data_yaml,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        save_json=args.save_json,
    )
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    print(f"Precision: {results.box.mp:.4f}")
    print(f"Recall: {results.box.mr:.4f}")

    if args.task_name:
        task = Task.init(project_name="yolo", task_name=args.task_name)
        for k, v in {
            "mAP50": float(results.box.map50),
            "mAP50-95": float(results.box.map),
            "precision": float(results.box.mp),
            "recall": float(results.box.mr),
        }.items():
            Logger.current_logger().report_scalar("val", k, iteration=0, value=v)
        task.close()


def cmd_export(args):
    model = YOLO(args.weights)
    model.export(format=args.format, imgsz=args.imgsz, half=args.half, dynamic=args.dynamic)
    print(f"→ 导出完成: {args.weights.replace('.pt', f'.{args.format}')}")


def cmd_predict(args):
    model = YOLO(args.weights)
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        save=args.save,
        project=args.output_dir,
    )
    for r in results:
        print(f"  {r.path} → {len(r.boxes)} 个检测结果")


# ── Main ─────────────────────────────────────────────────────────

COMMANDS = {
    "train": cmd_train,
    "val": cmd_val,
    "export": cmd_export,
    "predict": cmd_predict,
}

if __name__ == "__main__":
    args = build_parser().parse_args()
    COMMANDS[args.command](args)
