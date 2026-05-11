# YOLO Experiment Manifest

> 配合 `/yolo-pipeline` 使用。文件放在 `experiments/` 目录，通过 Git 同步。**不要将数据集和权重放入 Git。**

## 项目

- **项目名**: yolo
- **任务**: YOLO 缺陷检测
- **框架**: Ultralytics YOLO
- **训练目标**: Linux 服务器 (自建 或 云GPU)
- **实验追踪**: ClearML
- **ClearML 队列**: yolo-linux
- **数据 YAML**: ~/datasets/defect/data.yaml
- **主目标**: recall-first / precision-first / balanced

## 安全规则

- 原始图片和标签不入 Git
- 大文件 (.pt/.pth/.onnx/.engine) 和 ClearML 缓存不入 Git
- `requirements.txt` 入 Git 方便 Agent 安装依赖
- 修改结构/loss 前必须确认 baseline 证据充分
- 云 GPU 运行前必须确认费用预算

## Stage 0: 环境验证

| 检查项 | 命令 | 通过标准 |
|--------|------|---------|
| CUDA可用 | `python -c "import torch; print(torch.cuda.is_available())"` | True |
| ClearML连通 | `python -c "from clearml import Task; t=Task.init(project_name='yolo', task_name='conn_check'); t.close()"` | Web UI 出现任务 |
| 数据可读 | `python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"` | 无报错 |
| smoke test | `python train_yolo_clearml.py train --task-name smoke_1epoch --data-yaml ~/datasets/defect/data.yaml --model yolo11n.pt --epochs 1` | 1 epoch 完成 |

## Stage 1: Baselines

| Run ID | Model | Imgsz | Epochs | 目的 |
|--------|-------|-------|--------|------|
| baseline_yolo11n_640 | yolo11n.pt | 640 | 100 | 快速基线 |
| baseline_yolo11s_640 | yolo11s.pt | 640 | 100 | 容量对比 |
| baseline_yolo11n_960 | yolo11n.pt | 960 | 100 | 小目标敏感度 |

## Stage 2: 调参

Stage 1 完成后根据结果填写:

| Run ID | 改动 | 假设 | 关注指标 |
|--------|------|------|---------|
| tune_aug_conservative | 保守增强 | 减少正常纹理误检 | precision, mAP50-95 |
| tune_imgsz_960_s | yolo11s + 960 | 小缺陷召回提升 | recall, per-class AP |
| tune_threshold | 降低置信度阈值 | 检出更多缺陷 | recall, 误检/图 |

## Stage 3: 结构/Loss 改进

仅 baseline 稳定后进入:

| 候选 | 具体改动 | 理论依据 | 风险 | 最低证据 |
|------|---------|---------|------|---------|
| attention_fusion | TBD | 纹理缺陷需特征重加权 | 推理变慢 | 提升少数类recall |
| small_object_head | TBD | 微小缺陷欠表达 | 代码风险高 | 提升小目标AP |
| iou_loss_variant | TBD | 定位精度是瓶颈 | 训练可能不稳 | 提升mAP50-95 |

## 结果汇总

| Run ID | 状态 | mAP50 | mAP50-95 | Precision | Recall | 备注 |
|--------|------|-------|----------|-----------|--------|------|
| smoke_1epoch | pending | - | - | - | - | |

## 下一步决策

- **继续调参**: 
- **进入消融**: 
- **修改数据/标注**: 
- **准备论文/专利/基金材料**: 
