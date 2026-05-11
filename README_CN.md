# Auto-research for Codex

基于 Markdown Skills 的科研自动化工作流仓库。**当前主线：YOLO 缺陷检测**，支持两种 Linux 服务器部署方案。

## 分工模型

```
Claude Code (执行/训练调度)           Codex (结果复盘/论文产出)
    │                                        │
    ├── 创建项目、配置环境                   ├── 读 ClearML 指标
    ├── SSH 远程提交训练任务                 ├── 生成 RUN_TABLE.csv
    ├── 生成 Baseline 实验矩阵               ├── 生成 RESULTS_SUMMARY.md
    ├── 批量运行调参实验                     ├── 输出 NEXT_EXPERIMENTS.md
    ├── 监控训练状态                         ├── 生成 NARRATIVE_REPORT.md
    └── 处理异常恢复                         └── 论文/专利/基金/海报/演讲稿
```

---

## 1. 两种部署方案

### 方案一：自有实验室 Linux 服务器

适合：有实验室 GPU 服务器，局域网或可 SSH 访问，数据不出实验室。

```text
Mac (Claude Code/Codex编排)
  │ git push/pull
  ▼
GitHub 私有仓库 (代码/配置/报告)
  │ git pull
  ▼
Linux 服务器 (训练)
  ├── conda 环境
  ├── ClearML Agent (systemd 常驻)
  └── 数据集本地存储
  │
  ▼
ClearML (实验追踪/指标对比)
```

### 方案二：租用云 GPU

适合：实验室无空闲 GPU，需要更高规格显卡 (A100/H100)，短期冲刺。

```text
Mac (Claude Code/Codex编排)
  │ git push
  ▼
GitHub 私有仓库
  │ git clone (每次租用新实例)
  ▼
云GPU实例 (AutoDL/Vast.ai)
  ├── 临时环境
  ├── ClearML Agent
  ├── 数据从对象存储拉取
  └── 自动关机保护
  │
  ▼
ClearML (实验追踪/指标对比)
```

### 对比总结

| 维度 | 自有服务器 | 云GPU |
|------|-----------|-------|
| **费用** | 电费+折旧 | 按小时付费 (~2元/h起) |
| **数据安全** | 数据不出实验室 | 需上传对象存储 |
| **初始化** | 一次配置，永久使用 | 每次租用需重新初始化 |
| **弹性** | 固定算力 | 随时选更高规格 |
| **适合场景** | 大量长时间训练 | 短期大模型冲刺 |
| **联网** | 局域网低延迟 | 取决于机房 |

---

## 2. 完整实验流程（8步）

### Step 1 — 克隆项目 (Mac)

```bash
cd ~/vibe_coding/codex-work
git clone git@github.com:<user>/Auto-research.git
cd Auto-research
```

克隆下来就是完整可用项目，内含：
- `train_yolo_clearml.py` — 训练/评估/导出/推理
- `AGENTS.md` — 服务器配置（编辑即可）
- `requirements.txt` — 依赖清单
- `experiments/YOLO_EXPERIMENT_MANIFEST.md` — 实验清单
- `tools/` — 服务器初始化、数据体检、看门狗

**编辑 `AGENTS.md`**，填写你的服务器信息（host、user、data_yaml 路径）。

### Step 2 — 准备数据集

整理为 Ultralytics YOLO 格式，**data.yaml 和图片/标签放在同一个数据集目录**：

```text
~/datasets/defect/       # 数据集根目录 (服务器上，不入Git)
  data.yaml              # 唯一的权威配置
  images/ train/ val/ test/
  labels/ train/ val/ test/
```

`data.yaml` 示例：

```yaml
path: /home/user/datasets/defect
train: images/train
val: images/val
test: images/test
nc: 3
names:
  - scratch
  - crack
  - stain
```

**自有服务器**：`scp -r datasets/defect lab-server:~/datasets/`

**云GPU**：上传到对象存储，例如 `rclone copy datasets/defect oss:my-bucket/datasets/defect`

### Step 3 — 服务器环境初始化

**自有服务器**：

```bash
# Mac 上远程执行
ssh lab-server "bash -s" < tools/setup_linux_server.sh

# 或在服务器上直接执行
git clone git@github.com:<user>/Auto-research.git ~/Auto-research
cd ~/Auto-research
bash tools/setup_linux_server.sh
```

脚本自动完成：
1. 检查 NVIDIA 驱动、CUDA 版本、RAM、磁盘
2. 创建 conda 环境 (Python 3.10)
3. 安装 PyTorch + CUDA + Ultralytics + ClearML
4. 配置 ClearML 凭证
5. 安装 systemd 后台服务 (ClearML Agent + Watchdog)

**云GPU** (AutoDL 示例)：

```bash
# 1. 在 AutoDL 控制台选 GPU + PyTorch 镜像，启动实例
# 2. SSH 连接后，把代码 clone 到数据盘（/root/autodl-tmp 是持久化盘，实例释放后文件保留）
git clone git@github.com:<user>/Auto-research.git /root/autodl-tmp/Auto-research
cd /root/autodl-tmp/Auto-research

# 3. 设置 ClearML 凭证
export CLEARML_API_ACCESS_KEY="你的key"
export CLEARML_API_SECRET_KEY="你的secret"

# 4. 运行初始化 (拉数据、装依赖、启动Agent、定时关机)
bash tools/cloud_init.sh \
  --data-source oss://my-bucket/datasets/defect \
  --max-hours 8
```

> **注意**：AutoDL 的 `/root/` 是系统盘（实例释放丢失），`/root/autodl-tmp/` 是数据盘（持久保留）。代码和数据都要放到 `autodl-tmp` 下。下次租用实例时，如果数据盘还在，直接 `cd /root/autodl-tmp/Auto-research && git pull` 即可，不用重新 clone。

### Step 4 — 数据体检

```bash
# 在服务器上执行（数据在哪里就在哪里做体检）
ssh lab-server "cd ~/datasets/defect && \
  python ~/work/yolo/tools/yolo_data_health.py data.yaml \
  --markdown-out ~/work/yolo/results/DATA_HEALTH_REPORT.md"
```

检查项：
- train/val/test 路径是否可解析
- 类别数是否与 `names` 一致
- 标签格式是否正确 (5列 YOLO 格式，归一化坐标)
- 空标签、缺失图片、重复文件名、类别不平衡、极小框

### Step 5 — Baseline 实验

```bash
# 提交到服务器（通过 SSH 远程执行）
ssh lab-server "cd ~/Auto-research && conda activate yolo && \
  python train_yolo_clearml.py train --task-name smoke_1epoch \
  --data-yaml ~/datasets/defect/data.yaml --model yolo11n.pt --epochs 1"

# smoke test 通过后，批量跑 baseline
for exp in baseline_yolo11n_640 baseline_yolo11s_640 baseline_yolo11n_960; do
  ssh lab-server "cd ~/Auto-research && conda activate yolo && \
    python train_yolo_clearml.py train --task-name $exp \
    --data-yaml ~/datasets/defect/data.yaml --model yolo11n.pt \
    --imgsz 640 --epochs 100 --batch -1" &
done
```

**或者使用 ClearML Agent (批量自动化)**：

```bash
# 确保服务器上 ClearML Agent 在运行
ssh lab-server "systemctl --user start clearml-agent-yolo"

# Mac 端提交任务
clearml-task --project yolo --name baseline_yolo11n_640 \
  --repo git@github.com:<user>/Auto-research.git --branch main \
  --queue yolo-linux --script train_yolo_clearml.py \
  --args "--task-name baseline_yolo11n_640 --data-yaml ~/datasets/defect/data.yaml --model yolo11n.pt --epochs 100"
```

### Step 6 — 结果复盘 (Codex)

在 Mac 项目目录启动 Codex，然后执行：

```text
/yolo-pipeline "复盘 ClearML 中 yolo 项目的最新实验，生成 RUN_TABLE.csv、RESULTS_SUMMARY.md 和 NEXT_EXPERIMENTS.md"
```

Codex 会：
- 从 ClearML 拉取所有实验的 mAP、precision、recall、loss 曲线
- 对比各实验，标记最优模型
- 分析每个类别的表现
- 识别瓶颈（数据量/分辨率/模型容量/类别不平衡/阈值）
- 输出下一轮 3-8 个高价值实验

### Step 7 — 调参实验（第二轮）

根据复盘结果，Codex 会建议具体实验。Claude Code 执行：

```bash
# 批量提交调参实验
ssh lab-server "cd ~/Auto-research && conda activate yolo && \
  python train_yolo_clearml.py train --task-name tune_aug_strong \
  --data-yaml ~/datasets/defect/data.yaml --model yolo11n.pt \
  --imgsz 640 --epochs 100 --extra hsv_h=0.015 hsv_s=0.7 hsv_v=0.4"
```

第二轮实验类型：
- 图像尺寸 (640, 960, 1280)
- 数据增强 (保守 vs 激进)
- 模型容量 (n, s, m)
- 类别平衡 (采样策略 or loss 加权)
- 置信度/NMS 阈值调优

### Step 8 — 结构/Loss 实验 + 产出

第三轮才考虑结构修改（仅在 baseline 稳定后进行）：

```text
候选: attention_fusion, small_object_head, iou_loss_variant, lightweight_neck
每个候选必须写明: 假设 / 代码改动 / 预期指标变化 / 风险 / 回滚方案
```

产出生成：

```text
# 结果稳定后，Codex 生成综合底稿
/yolo-pipeline "生成 NARRATIVE_REPORT.md，区分已证实结论、合理解释和未证实claim"

# 按需求分流
/paper-writing "NARRATIVE_REPORT.md" — style: Nature
/patent-pipeline "NARRATIVE_REPORT.md -- CN"
/grant-proposal "NARRATIVE_REPORT.md -- NSFC"
/paper-poster "paper/"
/paper-slides "paper/"
```

---

## 3. 项目结构

```text
Auto-research/
  AGENTS.md                          # 项目配置（编辑填写服务器信息）
  requirements.txt                   # Python依赖
  train_yolo_clearml.py              # YOLO11 训练/评估/导出/推理
  .gitignore                         # 排除数据和权重
  experiments/
    YOLO_EXPERIMENT_MANIFEST.md      # 实验清单
  results/                           # 结果输出目录
  NARRATIVE_REPORT.md                # 论文/专利/基金底稿（训练后自动生成）
  skills/                            # 科研自动化 Skills
  templates/                         # 论文/专利模板
  tools/
    setup_linux_server.sh            # 自有服务器初始化
    cloud_init.sh                    # 云GPU初始化
    watchdog.py                      # 训练监控守护进程
    yolo_data_health.py              # 数据体检工具
  docs/                              # 详细文档
```

`data.yaml` 放在数据集目录里（如 `~/datasets/defect/data.yaml`），**不入项目仓库**。

---

## 4. 常用操作速查

### 查看训练状态

```bash
# 服务器上
python tools/watchdog.py --status

# 远程
ssh lab-server "python ~/work/yolo/tools/watchdog.py --status"
```

### 恢复中断的训练

```bash
ssh lab-server "cd ~/Auto-research && conda activate yolo && \
  python train_yolo_clearml.py train --task-name baseline_yolo11n_640 --resume \
  --data-yaml ~/datasets/defect/data.yaml --model yolo11n.pt"
```

### 同步代码并重启 Agent

```bash
ssh lab-server "cd ~/Auto-research && git pull && systemctl --user restart clearml-agent-yolo"
```

### 云GPU：取消自动关机

```bash
ssh -p <port> root@<instance> "sudo shutdown -c"
```

---

## 5. ClearML 任务提交速查

### 自有服务器 (直接 SSH 执行)

```bash
ssh lab-server "cd ~/Auto-research && conda activate yolo && \
  python train_yolo_clearml.py \
  --task-name <任务名> \
  --data-yaml ~/datasets/defect/data.yaml \
  --model yolo11n.pt \
  --imgsz 640 --epochs 100 --batch -1"
```

### 自有服务器 (ClearML Agent 队列)

```bash
clearml-task --project yolo --name <任务名> \
  --repo git@github.com:<user>/Auto-research.git \
  --branch main --queue yolo-linux \
  --script train_yolo_clearml.py \
  --requirements requirements.txt \
  --args "--task-name <任务名> --data-yaml ~/datasets/defect/data.yaml --model yolo11n.pt --epochs 100"
```

### 云GPU (直接执行)

```bash
ssh -p <port> root@<instance> \
  "cd /root/autodl-tmp/yolo && \
   python train_yolo_clearml.py train --task-name <任务名> \
   --data-yaml /root/autodl-tmp/datasets/defect/data.yaml \
   --model yolo11n.pt --epochs 100 --shutdown-when-done"
```

---

## 6. 自动实验节奏

```
第一轮: baseline (证明环境可用)
  smoke_1epoch → baseline_yolo11n_640 → baseline_yolo11s_640 → baseline_yolo11n_960

第二轮: 调参 (在baseline上优化)
  tune_imgsz → tune_aug → tune_threshold → tune_balance
  (根据复盘选 3-8 个高价值实验)

第三轮: 结构/loss (仅在baseline充分后)
  attention_fusion / small_object_head / iou_loss / lightweight_neck
  (每个必须有假设+风险+回滚方案)
```

---

## 7. 扩展能力索引

需要时使用，不放在主线前面。

### 通用科研流水线

| 入口 | 用途 |
|------|------|
| `/idea-discovery "方向"` | 从方向找 idea |
| `/experiment-bridge` | 实验计划 → 代码和运行 |
| `/auto-review-loop "范围"` | 自动审稿/修复/再审 |
| `/paper-writing "NARRATIVE_REPORT.md"` | 叙事报告 → 论文 |
| `/research-pipeline "方向"` | 全流程：idea→实验→论文 |
| `/rebuttal "paper/ + reviews"` | 审稿回复 |

### 论文/专利/展示

| 入口 | 用途 |
|------|------|
| `/paper-plan` / `/paper-write` / `/paper-compile` | 论文大纲/正文/编译 |
| `/patent-pipeline` | 专利交底和申请 |
| `/grant-proposal` | 基金申请 |
| `/paper-poster` / `/paper-slides` | 海报/PPT演讲 |
| `/citation-audit` / `/paper-claim-audit` | 引用/数字审计 |

### 文献/知识库

| 入口 | 用途 |
|------|------|
| `/research-lit` / `/novelty-check` | 文献检索/查新 |
| `/arxiv` / `/deepxiv` | arXiv检索/分层阅读 |
| `/research-wiki init` | 项目知识库 |
| Zotero / Obsidian 集成 | 论文管理/Notes |

### 算力扩展

| 入口 | 用途 |
|------|------|
| `/vast-gpu` | Vast.ai 云GPU |
| `/monitor-experiment` | 监控训练 |
| `/training-check` | 检查NaN/发散/GPU空转 |

### 其他领域

| 入口 | 用途 |
|------|------|
| `/idea-discovery-robot` | 机器人/具身智能选题 |
| `/comm-lit-review` | 通信方向文献综述 |
| `/formula-derivation` / `/proof-writer` | 公式推导/数学证明 |

---

## 8. 安装和维护

### Codex 全局安装 Skills

```bash
mkdir -p ~/.codex/skills
cp -a Auto-research/skills/skills-codex/* ~/.codex/skills/
```

### 项目级安装

```bash
cd /path/to/your-project
bash Auto-research/tools/install_aris.sh --platform codex
```

### 更新

```bash
cd Auto-research && git pull
```

---

## 9. 核心规则

- GitHub 只放代码/配置/报告，**不放数据和权重**
- 先跑 smoke test (1 epoch) 验证环境，再批量实验
- Baseline 不稳定时不要改结构/loss
- 云GPU 必须设置定时关机
- 写作前区分：**已证实的结论** vs **合理解释** vs **未证实claim**

---

## License

见 [LICENSE](LICENSE)。
