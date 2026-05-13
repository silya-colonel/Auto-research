# Auto-research (ARIS) — 自有 Linux / Windows GPU 服务器部署

基于 Markdown Skills 的科研自动化工作流仓库。**当前主线：YOLO 缺陷检测**。

## 架构

```
Mac (Claude Code/Codex 编排)
  │ git push/pull
  ▼
GitHub 私有仓库 (代码/配置/报告)
  │ git pull
  ▼
自有 Linux 服务器 / Windows 服务器 (训练)
  ├── conda 环境
  ├── 数据集本地存储
  └── runs/<task-name>/metrics.json
```

部署方案：你的 Linux 或 Windows GPU 服务器负责训练，Mac 负责代码和编排，GitHub 做中转。两台服务器可以同时保留：Linux 适合长期后台任务，Windows 适合你已有显卡机器或 VS Code Remote SSH 直接操作。

## 分工模型

```
Claude Code (执行/训练调度)           Codex (结果复盘/论文产出)
    │                                        │
    ├── 创建项目、配置环境                   ├── 读本地 metrics.json 指标
    ├── SSH 远程提交训练任务                 ├── 生成 RUN_TABLE.csv
    ├── 生成 Baseline 实验矩阵               ├── 生成 RESULTS_SUMMARY.md
    ├── 批量运行调参实验                     ├── 输出 NEXT_EXPERIMENTS.md
    ├── 监控训练状态                         ├── 生成 NARRATIVE_REPORT.md
    └── 处理异常恢复                         └── 论文/专利/基金/海报/演讲稿
```

---

## 快速开始（4 步）

### Step 1 — 克隆项目 (Mac)

```bash
cd ~/vibe_coding/codex-work
git clone git@github.com:<你的GitHub用户名或组织名>/Auto-research.git
cd Auto-research
```

> 需要手动替换：`<你的GitHub用户名或组织名>` 是 GitHub 仓库地址里的 owner，例如仓库是 `https://github.com/abc/Auto-research`，这里就写 `abc`。如果仓库在组织下面，就写组织名。

克隆下来就是完整可用项目，内含：
- `train_yolo.py` — 训练/评估/导出/推理脚本
- `AGENTS.md` — 服务器配置（编辑即可）
- `requirements.txt` — 依赖清单
- `experiments/YOLO_EXPERIMENT_MANIFEST.md` — 实验清单
- `tools/` — 服务器初始化、数据体检、看门狗

**编辑 `AGENTS.md`**，填写你的服务器信息（host、user、data_yaml 路径）。如果两台服务器都要用，Linux 和 Windows 都填，运行实验时明确指定用哪台。

### Step 2 — 准备数据集

整理为 Ultralytics YOLO 格式，**data.yaml 和图片/标签放在同一个目录**：

```text
~/datasets/defect/       # 数据集根目录 (服务器上，不入Git)
  data.yaml              # 唯一的权威配置
  images/ train/ val/ test/
  labels/ train/ val/ test/
```

Linux `data.yaml` 示例：

```yaml
path: /home/<你的Linux用户名>/datasets/defect
train: images/train
val: images/val
test: images/test
nc: 3
names:
  - scratch
  - crack
  - stain
```

Windows `data.yaml` 示例：

```yaml
path: C:/datasets/defect
train: images/train
val: images/val
test: images/test
nc: 3
names:
  - scratch
  - crack
  - stain
```

上传到 Linux 服务器：

```bash
scp -r datasets/defect <你的服务器SSH别名或IP>:~/datasets/
```

上传到 Windows 服务器时，推荐先在 Windows 上创建 `C:\datasets`，再用 VS Code Remote SSH、资源管理器、Kaggle 下载脚本，或下面这种 `scp` 方式：

```bash
scp -r datasets/defect <你的Windows服务器SSH别名或IP>:/C:/datasets/
```

如果 Windows 的 `scp` 路径不兼容，直接在 Windows PowerShell 里进入项目后运行：

```powershell
python tools\kaggle_download.py
```

然后粘贴 KaggleHub 的 `dataset_download(...)` 代码片段，让脚本自动下载和转换。

### Step 3 — 服务器环境初始化（Linux）

需要你手动替换的字段：

| 占位符 | 含义 | 示例 |
|--------|------|------|
| `<你的GitHub用户名或组织名>` | GitHub 仓库 owner，不是 Linux 用户名 | `silyalovelance-creator` |
| `<你的服务器SSH别名或IP>` | `~/.ssh/config` 里的 Host，或服务器 IP | `linux-yan` / `172.28.x.x` |
| `<你的Linux用户名>` | 服务器登录用户名，只在直接写完整 SSH 命令时需要 | `useryan` |
| `~/datasets/defect/data.yaml` | 服务器上的数据集配置路径 | `/home/useryan/datasets/defect/data.yaml` |

```bash
# 方式 A：在服务器上直接执行（最不容易出错）
git clone git@github.com:<你的GitHub用户名或组织名>/Auto-research.git ~/Auto-research
cd ~/Auto-research
bash tools/setup_linux_server.sh

# 方式 B：Mac 上远程执行（前提：服务器上已经有 ~/Auto-research）
ssh <你的服务器SSH别名或IP> "cd ~/Auto-research && bash tools/setup_linux_server.sh"
```

脚本自动完成：
1. 检查 NVIDIA 驱动、CUDA 版本、RAM、磁盘空间
2. 创建 conda 环境 `silya` (Python 3.11)
3. 安装 PyTorch + CUDA + Ultralytics
4. 安装 systemd 后台服务 (Watchdog 监控守护进程)

验证环境：

```bash
ssh <你的服务器SSH别名或IP> "conda run -n silya python -c 'import torch; print(torch.cuda.is_available())'"
# 应输出 True
```

### Step 3b — 服务器环境初始化（Windows）

Windows 服务器建议用 PowerShell。若要从 Mac 远程执行，Windows 需要先启用 OpenSSH Server；如果你通过 cpolar 连接 Windows，免费 TCP 地址/端口重启后可能变化，运行前先确认当前 endpoint。需要你手动替换：

| 占位符 | 含义 | 示例 |
|--------|------|------|
| `<你的GitHub用户名或组织名>` | GitHub 仓库 owner | `silyalovelance-creator` |
| `<你的Windows服务器SSH别名或IP>` | Windows OpenSSH 的 Host 或 IP | `win-gpu` / `172.28.123.19` |
| `<你的Windows用户名>` | Windows 登录用户名 | `34064` / `useryan` |
| `C:\datasets\defect\data.yaml` | Windows 上的数据集配置路径 | `C:\datasets\defect\data.yaml` |

在 Windows PowerShell 里直接执行：

```powershell
git clone git@github.com:<你的GitHub用户名或组织名>/Auto-research.git "$env:USERPROFILE\Auto-research"
cd "$env:USERPROFILE\Auto-research"
powershell -ExecutionPolicy Bypass -File .\tools\setup_windows_server.ps1
```

或者从 Mac 远程执行（前提：Windows 已启用 OpenSSH Server，且已经 clone 到用户目录）：

```bash
ssh <你的Windows服务器SSH别名或IP> 'powershell -NoProfile -ExecutionPolicy Bypass -Command "cd $env:USERPROFILE\Auto-research; .\tools\setup_windows_server.ps1"'
```

脚本自动完成：
1. 检查 NVIDIA 驱动和 `nvidia-smi`
2. 创建 conda 环境 `silya` (Python 3.11)
3. 安装 PyTorch CUDA + Ultralytics + KaggleHub
4. 验证 `torch.cuda.is_available()`

验证 Windows 环境：

```bash
ssh <你的Windows服务器SSH别名或IP> 'powershell -NoProfile -Command "conda run -n silya python -c `"import torch; print(torch.cuda.is_available())`""'
```

### Step 4 — 跑第一个实验

Linux smoke test：

```bash
ssh <你的服务器SSH别名或IP> "cd ~/Auto-research && \
  conda run -n silya python train_yolo.py train --task-name smoke_1epoch \
  --data-yaml ~/datasets/defect/data.yaml --model yolo11n.pt --epochs 1"
```

Windows smoke test：

```bash
ssh <你的Windows服务器SSH别名或IP> 'powershell -NoProfile -Command "cd $env:USERPROFILE\Auto-research; conda run -n silya python train_yolo.py train --task-name smoke_1epoch --data-yaml C:\datasets\defect\data.yaml --model yolo11n.pt --epochs 1"'
```

Linux 批量 baseline：

```bash
for exp in baseline_yolo11n_640 baseline_yolo11s_640 baseline_yolo11n_960; do
  ssh <你的服务器SSH别名或IP> "cd ~/Auto-research && \
    conda run -n silya python train_yolo.py train --task-name $exp \
    --data-yaml ~/datasets/defect/data.yaml --model yolo11n.pt \
    --imgsz 640 --epochs 100" &
done
```

Windows 批量 baseline：

```bash
for exp in baseline_yolo11n_640 baseline_yolo11s_640 baseline_yolo11n_960; do
  ssh <你的Windows服务器SSH别名或IP> 'powershell -NoProfile -Command "cd $env:USERPROFILE\Auto-research; conda run -n silya python train_yolo.py train --task-name '"$exp"' --data-yaml C:\datasets\defect\data.yaml --model yolo11n.pt --imgsz 640 --epochs 100"' &
done
```

训练完成后，指标保存在 `runs/<task-name>/metrics.json`：

```json
{
  "task_name": "baseline_yolo11n_640",
  "timestamp": "2026-05-12T15:30:00",
  "metrics": {
    "mAP50": 0.7234,
    "mAP50-95": 0.4512,
    "precision": 0.7891,
    "recall": 0.6523
  }
}
```

---

## 完整实验流程（8 步）

### Step 1 — 克隆和配置

见上面「快速开始」Step 1。

### Step 2 — 数据集准备

见上面「快速开始」Step 2。

### Step 3 — 服务器初始化

Linux 见「快速开始」Step 3；Windows 见「快速开始」Step 3b。

### Step 4 — 数据体检

```bash
# Linux
ssh <你的服务器SSH别名或IP> "cd ~/datasets/defect && \
  conda run -n silya python ~/Auto-research/tools/yolo_data_health.py data.yaml \
  --markdown-out ~/Auto-research/results/DATA_HEALTH_REPORT.md"

# Windows
ssh <你的Windows服务器SSH别名或IP> 'powershell -NoProfile -Command "cd C:\datasets\defect; conda run -n silya python $env:USERPROFILE\Auto-research\tools\yolo_data_health.py data.yaml --markdown-out $env:USERPROFILE\Auto-research\results\DATA_HEALTH_REPORT.md"'
```

检查项：train/val/test 路径、类别数一致性、标签格式、空标签、缺失图片、类别不平衡、极小框。

### Step 5 — Baseline 实验

```bash
# Linux 批量提交 baseline
for exp in baseline_yolo11n_640 baseline_yolo11s_640 baseline_yolo11n_960; do
  ssh <你的服务器SSH别名或IP> "cd ~/Auto-research && \
    conda run -n silya python train_yolo.py train --task-name $exp \
    --data-yaml ~/datasets/defect/data.yaml --model yolo11n.pt \
    --imgsz 640 --epochs 100" &
done
```

Windows 使用上面 Step 4 的 Windows baseline 命令，保持相同的 task name，方便后续结果汇总。

### Step 6 — 结果复盘 (Codex)

在 Mac 项目目录启动 Codex：

```text
/yolo-pipeline "复盘 runs/ 目录中最新实验结果，生成 RUN_TABLE.csv、RESULTS_SUMMARY.md 和 NEXT_EXPERIMENTS.md"
```

Codex 会：
- 读取各实验的 `metrics.json`，汇总 mAP、precision、recall
- 对比各实验，标记最优模型
- 分析每个类别的表现
- 识别瓶颈（数据量/分辨率/模型容量/类别不平衡）
- 输出下一轮 3-8 个高价值实验

### Step 7 — 调参实验（第二轮）

根据复盘结果运行调参实验：

```bash
# Linux
ssh <你的服务器SSH别名或IP> "cd ~/Auto-research && \
  conda run -n silya python train_yolo.py train --task-name tune_aug_strong \
  --data-yaml ~/datasets/defect/data.yaml --model yolo11n.pt \
  --imgsz 640 --epochs 100 --extra hsv_h=0.015 hsv_s=0.7 hsv_v=0.4"

# Windows
ssh <你的Windows服务器SSH别名或IP> 'powershell -NoProfile -Command "cd $env:USERPROFILE\Auto-research; conda run -n silya python train_yolo.py train --task-name tune_aug_strong --data-yaml C:\datasets\defect\data.yaml --model yolo11n.pt --imgsz 640 --epochs 100 --extra hsv_h=0.015 hsv_s=0.7 hsv_v=0.4"'
```

第二轮实验类型：图像尺寸、数据增强、模型容量、类别平衡、置信度/NMS 阈值。

### Step 8 — 结构/Loss 实验 + 产出

第三轮才考虑结构修改（仅在 baseline 稳定后进行）。每个候选必须写明：假设 / 代码改动 / 预期指标变化 / 风险 / 回滚方案。

结果稳定后生成产出：

```text
/yolo-pipeline "生成 NARRATIVE_REPORT.md，区分已证实结论、合理解释和未证实claim"

# 按需求分流
/paper-writing "NARRATIVE_REPORT.md" — style: Nature
/patent-pipeline "NARRATIVE_REPORT.md -- CN"
/grant-proposal "NARRATIVE_REPORT.md -- NSFC"
/paper-poster "paper/"
/paper-slides "paper/"
```

---

## 项目结构

```text
Auto-research/
  AGENTS.md                          # 项目配置（编辑填写服务器信息）
  requirements.txt                   # Python依赖
  train_yolo.py                      # YOLO11 训练/评估/导出/推理
  .gitignore                         # 排除数据和权重
  experiments/
    YOLO_EXPERIMENT_MANIFEST.md      # 实验清单
  results/                           # 结果输出目录
  NARRATIVE_REPORT.md                # 论文/专利/基金底稿
  skills/                            # 科研自动化 Skills
  templates/                         # 论文/专利模板
  tools/
    setup_linux_server.sh            # 服务器一键初始化
    setup_windows_server.ps1         # Windows 服务器一键初始化
    watchdog.py                      # 训练监控守护进程
    yolo_data_health.py              # 数据体检工具
  docs/                              # 详细文档
```

`data.yaml` 放在数据集目录里（Linux: `~/datasets/defect/data.yaml`；Windows: `C:\datasets\defect\data.yaml`），**不入项目仓库**。

---

## 常用操作速查

### 查看训练状态

```bash
# 服务器上
python tools/watchdog.py --status

# 远程
ssh <你的服务器SSH别名或IP> "conda run -n silya python ~/Auto-research/tools/watchdog.py --status"

# Windows 上一般直接看 runs/<task-name>/metrics.json；watchdog 主要给 Linux 后台任务使用。
```

### 恢复中断的训练

```bash
# Linux
ssh <你的服务器SSH别名或IP> "cd ~/Auto-research && \
  conda run -n silya python train_yolo.py train --task-name baseline_yolo11n_640 --resume \
  --data-yaml ~/datasets/defect/data.yaml"

# Windows
ssh <你的Windows服务器SSH别名或IP> 'powershell -NoProfile -Command "cd $env:USERPROFILE\Auto-research; conda run -n silya python train_yolo.py train --task-name baseline_yolo11n_640 --resume --data-yaml C:\datasets\defect\data.yaml"'
```

### 查看训练指标

```bash
# Linux
ssh <你的服务器SSH别名或IP> "cat ~/Auto-research/runs/yolo/<task-name>/metrics.json"

# Windows
ssh <你的Windows服务器SSH别名或IP> 'powershell -NoProfile -Command "Get-Content $env:USERPROFILE\Auto-research\runs\yolo\<task-name>\metrics.json"'
```

### 同步代码

```bash
# Linux
ssh <你的服务器SSH别名或IP> "cd ~/Auto-research && git pull"

# Windows
ssh <你的Windows服务器SSH别名或IP> 'powershell -NoProfile -Command "cd $env:USERPROFILE\Auto-research; git pull"'
```

---

## 自动实验节奏

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

## 扩展能力索引

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

### 训练监控

| 入口 | 用途 |
|------|------|
| `/monitor-experiment` | 监控训练 |
| `/training-check` | 检查NaN/发散/GPU空转 |

---

## 核心规则

- GitHub 只放代码/配置/报告，**不放数据和权重**
- 先跑 smoke test (1 epoch) 验证环境，再批量实验
- Baseline 不稳定时不要改结构/loss
- 写作前区分：**已证实的结论** vs **合理解释** vs **未证实claim**
- 训练指标保存在 `runs/<task-name>/metrics.json`，通过 Codex 复盘时读取

---

## 安装和维护

### 项目级安装 Skills

```bash
cd /path/to/your-project
bash Auto-research/tools/install_aris.sh --platform codex
```

### 更新

```bash
cd Auto-research && git pull
```

---

## License

见 [LICENSE](LICENSE)。
