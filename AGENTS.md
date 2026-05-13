# YOLO11 缺陷检测项目

> 此文件告诉 Codex 和 Claude Code 如何操作这个项目。

## 项目信息

| 配置项 | 值 |
|--------|-----|
| 任务 | YOLO 缺陷检测 |
| 框架 | Ultralytics YOLO11 |
| 训练目标 | 自有 Linux / Windows GPU 服务器 |
| 指标存储 | runs/<task-name>/metrics.json |
| GitHub | 代码+配置+报告，不放数据和权重 |

## 服务器配置

```yaml
active_server: linux                       # 手动选择: linux 或 windows

servers:
  linux:
    host: ""                               # 手动填写：Linux 服务器 IP 或 SSH 别名
    user: ""                               # 手动填写：SSH 用户名
    port: 22
    project_dir: ~/Auto-research           # 如不同请手动修改
    data_yaml: ~/datasets/defect/data.yaml # 手动确认
    conda_env: silya
    gpu_ids: "0"
  windows:
    host: ""                               # 手动填写：Windows 服务器 IP 或 SSH 别名
    user: ""                               # 手动填写：Windows SSH 用户名
    port: 22
    project_dir: C:\Users\<你的Windows用户名>\Auto-research
    data_yaml: C:\datasets\defect\data.yaml
    conda_env: silya
    gpu_ids: "0"
```

## 训练命令

```bash
# 基础训练
python train_yolo.py train \
  --task-name baseline_yolo11n_640 \
  --data-yaml ~/datasets/defect/data.yaml \
  --model yolo11n.pt --epochs 100

# 续训
python train_yolo.py train --task-name baseline_yolo11n_640 --resume \
  --data-yaml ~/datasets/defect/data.yaml

# 评估
python train_yolo.py val \
  --weights runs/yolo/baseline/weights/best.pt \
  --data-yaml ~/datasets/defect/data.yaml

# 导出
python train_yolo.py export \
  --weights runs/yolo/baseline/weights/best.pt --format onnx
```

## 自动化边界

**AI 可以自动做:**
- 检查 data.yaml、标签格式、数据健康度
- 生成 baseline 和调参实验矩阵
- 通过 SSH 提交 Linux 或 Windows 训练
- 从 runs/<task>/metrics.json 读取指标做复盘
- 生成 NARRATIVE_REPORT.md → 论文/专利/基金

**必须人工确认:**
- 修改类别定义或标注策略
- 修改模型结构或 loss
- 将结果写入论文/专利 claim

## 日常操作

```text
1. 改代码/配置 → git push
2. SSH 到选定服务器 git pull
3. 在选定服务器执行 python train_yolo.py train ...
4. 查看 runs/<task-name>/metrics.json 看训练指标
5. Codex 复盘 → experiments/YOLO_EXPERIMENT_MANIFEST.md → 下一轮
```
