"""YOLO11 训练/评估/导出/推理

用法:
  # 训练
  python train_yolo.py train --task-name baseline_yolo11n_640 \
    --data-yaml ~/datasets/defect/data.yaml --model yolo11n.pt --epochs 100

  # 断点续训
  python train_yolo.py train --task-name baseline_yolo11n_640 --resume \
    --data-yaml ~/datasets/defect/data.yaml

  # 评估已有模型
  python train_yolo.py val --weights runs/yolo/baseline/weights/best.pt \
    --data-yaml ~/datasets/defect/data.yaml

  # 导出 ONNX/TensorRT
  python train_yolo.py export --weights runs/yolo/baseline/weights/best.pt \
    --format onnx

  # 推理单张图片
  python train_yolo.py predict --weights runs/yolo/baseline/weights/best.pt \
    --source ~/test.jpg --imgsz 640

  # 训练完成后自动关机
  python train_yolo.py train --task-name exp_v2 \
    --data-yaml ~/datasets/defect/data.yaml --model yolo11n.pt --shutdown
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO


# ── CLI ──────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="YOLO11 训练/评估/导出/推理")
    sub = p.add_subparsers(dest="command", required=True)

    # train
    t = sub.add_parser("train", help="训练模型")
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
    t.add_argument("--shutdown", action="store_true", help="训练完成后自动关机")
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


def save_metrics(save_dir: str, task_name: str, metrics: dict):
    """保存训练指标到本地 JSON 文件"""
    p = Path(save_dir)
    p.mkdir(parents=True, exist_ok=True)
    record = {
        "task_name": task_name,
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics,
    }
    metrics_file = p / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    print(f"→ 指标已保存: {metrics_file}")


# ── Commands ─────────────────────────────────────────────────────

def cmd_train(args):
    data_yaml = Path(args.data_yaml)
    if not data_yaml.exists():
        sys.exit(f"data.yaml 不存在: {data_yaml}")

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
                sys.exit(1)
            time.sleep(30)

    save_dir = getattr(results, "save_dir", None) or str(
        Path(args.runs_dir) / args.task_name
    )

    # 保存训练指标到本地 JSON
    mAP = getattr(getattr(results, "box", None), "map50", None)
    metrics = {
        "mAP50": round(mAP, 4) if mAP else None,
    }

    # 尝试运行验证并记录更多指标
    try:
        val_results = model.val()
        metrics.update({
            "mAP50": round(float(val_results.box.map50), 4),
            "mAP50-95": round(float(val_results.box.map), 4),
            "precision": round(float(val_results.box.mp), 4),
            "recall": round(float(val_results.box.mr), 4),
        })
        print(f"mAP50: {val_results.box.map50:.4f}  mAP50-95: {val_results.box.map:.4f}")
    except Exception as e:
        print(f"验证失败: {e}")

    save_metrics(save_dir, args.task_name, metrics)

    msg = f"{args.task_name}: 完成 (mAP50={metrics.get('mAP50', 'N/A')})"
    print(msg)
    notify("训练完成", msg)

    if args.shutdown:
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
