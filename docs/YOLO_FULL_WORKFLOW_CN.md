# YOLO 缺陷检测全流程使用方法 — Linux 版

适用场景：

```text
Mac 写代码和调度
→ GitHub 私有仓库同步代码
→ Linux 服务器训练 YOLO
→ ClearML 看数据、日志、指标和模型
→ Claude Code 执行训练 / Codex 复盘结果、生成论文/专利/基金材料
```

支持两种部署：**自有实验室 Linux 服务器** 和 **租用云 GPU**。

## 0. 最终产物

跑通后，项目形成：

```text
project/
  AGENTS.md                          # 项目配置
  requirements.txt                   # Python依赖
  train_yolo_clearml.py              # 训练包装脚本
  .gitignore
  experiments/
    YOLO_EXPERIMENT_MANIFEST.md      # 实验清单
  results/
    DATA_HEALTH_REPORT.md            # 数据体检
    RUN_TABLE.csv                    # 结果表
    RESULTS_SUMMARY.md               # 复盘
    NEXT_EXPERIMENTS.md              # 下一轮建议
  NARRATIVE_REPORT.md                # 论文/专利/基金底稿
  tools/
    setup_linux_server.sh            # 自有服务器初始化
    cloud_init.sh                    # 云GPU初始化
    watchdog.py                      # 训练监控
```

## 1. 总体分工

### Mac
负责：写代码、改配置、Codex 编排、文献、GitHub 同步、ClearML 查看、论文/专利/基金材料
不负责：长时间训练、存储原始数据、存储大模型权重

### Linux 服务器
负责：训练 YOLO、运行 ClearML Agent、存储数据和权重

### GitHub 私有仓库
负责：同步代码、配置、实验清单、小型结果表
不负责：不存原始图片、标签、大权重、ClearML 缓存

### Claude Code + Codex
Claude Code 负责：项目初始化、服务器环境部署、训练任务调度、异常恢复
Codex 负责：数据体检、实验规划、结果复盘、论文/专利/基金材料生成

## 2. Step 1 — 克隆项目

```bash
cd ~/vibe_coding/codex-work
git clone git@github.com:<user>/Auto-research.git
cd Auto-research
```

克隆下来就是完整可用项目：`train_yolo_clearml.py`、`AGENTS.md`、`requirements.txt`、`experiments/`、`tools/` 一应俱全。

## 3. Step 2 — 编辑配置

编辑 `AGENTS.md`，填写服务器信息：

```yaml
server:
  type: linux-lab
  host: "lab-gpu-01"          # SSH 别名或 IP
  user: "silya"
  port: 22
  project_dir: ~/Auto-research
  data_dir: ~/datasets/defect
  conda_env: yolo
  gpu_ids: "0"
```

在服务器数据集目录准备 `data.yaml`（和图片/标签放一起）：

```yaml
path: /home/silya/datasets/defect
train: images/train
val: images/val
test: images/test
nc: 3
names:
  - scratch
  - crack
  - stain
```

推送到 GitHub：

```bash
git add -A && git commit -m "init yolo project" && git push -u origin main
```

## 4. Step 3 — 服务器初始化

### 自有服务器

```bash
# Mac 远程执行
ssh lab-gpu-01 "bash -s" < tools/setup_linux_server.sh
```

或服务器上直接：

```bash
git clone git@github.com:<user>/Auto-research.git ~/work/yolo
cd ~/Auto-research
bash tools/setup_linux_server.sh
```

验证：

```bash
ssh lab-gpu-01 "conda activate yolo && python -c 'import torch; print(torch.cuda.is_available())'"
```

### 云GPU (AutoDL)

```bash
# 1. AutoDL 控制台：选 GPU + PyTorch 镜像，启动
# 2. SSH 连接
git clone git@github.com:<user>/Auto-research.git /root/autodl-tmp/Auto-research
cd /root/autodl-tmp/yolo

# 3. 设置 ClearML 凭证
export CLEARML_API_ACCESS_KEY="your-key"
export CLEARML_API_SECRET_KEY="your-secret"

# 4. 初始化 (含数据拉取、定时关机)
bash tools/cloud_init.sh \
  --data-source oss://my-bucket/datasets/defect \
  --max-hours 8
```

## 5. Step 4 — 数据体检

```bash
# Mac 端（或服务器上）
python Auto-research/tools/yolo_data_health.py \
  ~/datasets/defect/data.yaml \
  --markdown-out results/DATA_HEALTH_REPORT.md \
  --json-out results/DATA_HEALTH_REPORT.json
```

重点看：缺失标签、坏行、未知类别、极小框、类别不平衡。

## 6. Step 5 — Smoke Test

```bash
ssh lab-gpu-01 "cd ~/Auto-research && conda activate yolo && \
  python train_yolo_clearml.py train --task-name smoke_1epoch \
  --data-yaml ~/datasets/defect/data.yaml --model yolo11n.pt --epochs 1"
```

成功标准：CUDA 不报错、数据可读、输出目录有内容、ClearML 里出现任务。

失败时不要进入批量实验。

## 7. Step 6 — Baseline 实验

### 方式A：SSH 直接执行

```bash
for exp in baseline_yolo11n_640 baseline_yolo11s_640 baseline_yolo11n_960; do
  ssh lab-gpu-01 "cd ~/Auto-research && conda activate yolo && \
    python train_yolo_clearml.py train --task-name $exp \
    --data-yaml ~/datasets/defect/data.yaml --model yolo11n.pt \
    --imgsz 640 --epochs 100 --batch -1" &
done
```

### 方式B：ClearML Agent 队列

```bash
# 确保服务器 Agent 在运行
ssh lab-gpu-01 "systemctl --user status clearml-agent-yolo"

# Mac 端提交
clearml-task --project yolo --name baseline_yolo11n_640 \
  --repo git@github.com:<user>/Auto-research.git --branch main \
  --queue yolo-linux --script train_yolo_clearml.py \
  --requirements requirements.txt \
  --args "--task-name baseline_yolo11n_640 --data-yaml ~/datasets/defect/data.yaml --model yolo11n.pt --epochs 100"
```

### 云GPU 直接执行

```bash
ssh -p <port> root@<instance> \
  "cd /root/autodl-tmp/yolo && \
   python train_yolo_clearml.py train --task-name baseline_yolo11n_640 \
   --data-yaml /root/autodl-tmp/datasets/defect/data.yaml \
   --model yolo11n.pt --epochs 100 --shutdown-when-done"
```

## 8. Step 7 — Codex 复盘

Mac 项目目录启动 Codex：

```text
/yolo-pipeline "复盘 ClearML 中 yolo 项目的最新实验，生成 RUN_TABLE.csv、RESULTS_SUMMARY.md 和 NEXT_EXPERIMENTS.md"
```

## 9. Step 8 — 调参 → 结构 → 产出

第二轮根据复盘跑 3-8 个调参实验。第三轮（可选）考虑结构/loss 修改。

结果稳定后：

```text
/yolo-pipeline "生成 NARRATIVE_REPORT.md"
/paper-writing "NARRATIVE_REPORT.md"
/patent-pipeline "NARRATIVE_REPORT.md -- CN"
```

## 10. 自动实验节奏

```
第一轮: smoke → baseline_yolo11n_640 → baseline_yolo11s_640 → baseline_yolo11n_960
第二轮: 调参 (尺寸/增强/类别平衡/阈值)，选 3-8 个
第三轮: 结构/loss (仅baseline稳定后)，每个必须有假设+风险+回滚
```

## 11. 日常循环

```
1. Mac 上改代码/配置
2. git commit + git push
3. 服务器 git pull 拉取代码
4. Claude Code 提交训练任务 (SSH或ClearML)
5. ClearML Web UI 看训练状态
6. Codex 读取结果，生成复盘
7. Codex 输出下一轮实验建议
8. 确认高风险实验，继续迭代
```

## 12. ClearML 里看什么

- Scalars: loss, mAP, precision, recall
- Plots: PR curve, confusion matrix, F1 curve
- Artifacts: 权重、结果目录
- Configuration: 每次训练的完整参数

缺陷检测重点：漏检成本高看 recall；误报多看 precision；小缺陷看高分辨率提升；类别不平衡看少数类。

## 13. 常见问题

### 何时用云GPU？
实验室 GPU 不够、需要并行多个 ablation、显存不足、想快速验证大模型或高分辨率。

### 何时改模型结构？
baseline 稳定 + 数据体检没问题 + 已知瓶颈类型 + 普通调参已不够。

### 训练中断怎么恢复？
```bash
ssh lab-gpu-01 "cd ~/Auto-research && conda activate yolo && \
  python train_yolo_clearml.py train --task-name baseline_yolo11n_640 --resume \
  --data-yaml ~/datasets/defect/data.yaml --model yolo11n.pt"
```

### 云GPU如何控制费用？
初始化时 `--max-hours 8` 设置定时关机。训练脚本 `--shutdown-when-done` 完成后关机。`sudo shutdown -c` 取消自动关机。

## 14. 最小可行路径

```text
1. git clone Auto-research，编辑 AGENTS.md
2. 服务器上准备数据集 + data.yaml
3. 服务器运行 tools/setup_linux_server.sh (自有) 或 tools/cloud_init.sh (云)
4. 跑 1 epoch smoke test，确认 ClearML 有日志
5. SSH 提交 baseline 实验
6. Codex 复盘，生成下一轮建议
7. 迭代实验
```
