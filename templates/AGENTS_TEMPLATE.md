# YOLO 缺陷检测项目

> 此文件告诉 Codex 和 Claude Code 如何操作你的项目。
> 根据你的部署模式填写对应部分。

---

## 项目信息

```yaml
project:
  name: yolo-defect-detection
  task: YOLO defect detection
  framework: ultralytics

server:
  type: linux-lab          # linux-lab 或 cloud-gpu
  host: ""                 # 服务器 IP 或 SSH 别名
  user: ""                 # SSH 用户名
  port: 22                 # SSH 端口
  project_dir: ~/work/yolo       # 服务器上的项目路径
  data_dir: ~/datasets/defect    # 服务器上的数据集路径
  conda_env: yolo          # conda 环境名
  gpu_ids: "0"             # 使用的GPU, 例如 "0" 或 "0,1"
```

## ClearML 配置

```yaml
clearml:
  project: yolo
  queue: yolo-linux        # ClearML Agent 监听队列名
  web_ui: ""               # ClearML Web UI 地址 (留空则用 SaaS)
```

## 自动化边界

**Claude Code / ARIS 可以自动做:**
- 检查 data.yaml、图片、标签、类别和数据划分
- 生成 YOLO baseline 实验矩阵
- 生成训练命令并远程提交到服务器
- 通过 SSH 或 ClearML Agent 执行训练
- 根据 ClearML 指标复盘结果
- 生成下一轮实验建议
- 生成 NARRATIVE_REPORT.md

**必须人工确认:**
- 修改类别定义
- 修改模型结构或 loss
- 花费云 GPU 费用
- 将结果写入论文/专利 claim

## 数据安全

- 原始图片和标签**不入 Git**
- 模型权重(.pt/.pth)**不入 Git**
- ClearML 缓存**不入 Git**
- GitHub 只放代码、配置、实验清单和报告
- 数据集放在服务器本地或对象存储

## 日常操作循环

```
1. Mac 上用 Claude Code 或 Codex 改代码/配置
2. git commit + git push
3. 服务器 git pull 拉取最新代码
4. Claude Code 提交训练任务 (SSH 远程执行 或 ClearML 队列)
5. ClearML Web UI 查看训练状态
6. Codex 读取 ClearML 结果并复盘
7. Codex 输出下一轮实验建议 (RESULTS_SUMMARY.md / NEXT_EXPERIMENTS.md)
8. 你确认高风险实验，继续迭代
```

## 部署模式说明

### linux-lab (自有实验室服务器)

优点: 数据不出实验室、无按量收费、局域网低延迟

服务器配置需求:
- Ubuntu 20.04/22.04
- NVIDIA GPU (RTX 3090/4090/A100)
- CUDA 12.1+
- conda 或 venv

### cloud-gpu (租用云GPU)

优点: 弹性算力、可随时升级更高规格

推荐平台: AutoDL (国内)、Vast.ai (国际)

注意事项:
- 数据需上传对象存储 (OSS/COS/S3)
- 设置自动关机时间限制费用
- 实例释放后数据丢失，注意持久化
