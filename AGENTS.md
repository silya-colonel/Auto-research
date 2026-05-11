# YOLO11 缺陷检测项目

> 此文件告诉 Codex 和 Claude Code 如何操作这个项目。

## 项目信息

| 配置项 | 值 |
|--------|-----|
| 任务 | YOLO 缺陷检测 |
| 框架 | Ultralytics YOLO11 |
| 训练目标 | Linux 服务器 (自建 / 云GPU) |
| 实验追踪 | ClearML |
| GitHub | 代码+配置+报告，不放数据和权重 |

## 服务器配置

```yaml
server:
  type: linux-lab                          # linux-lab 或 cloud-gpu
  host: ""                                 # 服务器 IP 或 SSH 别名
  user: ""                                 #
  port: 22
  project_dir: ~/Auto-research             # 本项目在服务器上的路径
  data_yaml: ~/datasets/defect/data.yaml   # 数据集配置文件路径 (和图片/标签放一起)
  conda_env: yolo
  gpu_ids: "0"

clearml:
  project: yolo
  queue: yolo-linux
```

## 训练命令

```bash
# 基础训练
python train_yolo_clearml.py train \
  --task-name baseline_yolo11n_640 \
  --data-yaml ~/datasets/defect/data.yaml \
  --model yolo11n.pt --epochs 100

# 续训
python train_yolo_clearml.py train --task-name baseline_yolo11n_640 --resume \
  --data-yaml ~/datasets/defect/data.yaml

# 评估
python train_yolo_clearml.py val \
  --weights runs/yolo/baseline/weights/best.pt \
  --data-yaml ~/datasets/defect/data.yaml

# 导出
python train_yolo_clearml.py export \
  --weights runs/yolo/baseline/weights/best.pt --format onnx
```

## 自动化边界

**AI 可以自动做:**
- 检查 data.yaml、标签格式、数据健康度
- 生成 baseline 和调参实验矩阵
- 通过 SSH 或 ClearML Agent 提交训练
- 从 ClearML 拉取指标做复盘
- 生成 NARRATIVE_REPORT.md → 论文/专利/基金

**必须人工确认:**
- 修改类别定义或标注策略
- 花费云 GPU 费用
- 修改模型结构或 loss
- 将结果写入论文/专利 claim

## 日常操作

```text
1. 改代码/配置 → git push
2. SSH 服务器 git pull
3. python train_yolo_clearml.py train ...
4. ClearML Web UI 看训练
5. Codex 复盘 → experiments/YOLO_EXPERIMENT_MANIFEST.md → 下一轮
```
