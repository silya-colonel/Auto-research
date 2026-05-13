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

active_server: linux       # 手动选择: linux 或 windows

servers:
  linux:
    host: ""               # 手动填写：Linux 服务器 IP 或 SSH 别名
    user: ""               # 手动填写：SSH 用户名
    port: 22
    project_dir: ~/Auto-research
    data_yaml: ~/datasets/defect/data.yaml
    conda_env: silya
    gpu_ids: "0"
  windows:
    host: ""               # 手动填写：Windows 服务器 IP 或 SSH 别名
    user: ""               # 手动填写：Windows SSH 用户名
    port: 22
    project_dir: C:\Users\<你的Windows用户名>\Auto-research
    data_yaml: C:\datasets\defect\data.yaml
    conda_env: silya
    gpu_ids: "0"
```

## 自动化边界

**Claude Code / ARIS 可以自动做:**
- 检查 data.yaml、图片、标签、类别和数据划分
- 生成 YOLO baseline 实验矩阵
- 生成训练命令并通过 SSH 提交到 Linux 或 Windows 服务器
- 根据本地 metrics.json 指标复盘结果
- 生成下一轮实验建议
- 生成 NARRATIVE_REPORT.md

**必须人工确认:**
- 修改类别定义
- 修改模型结构或 loss
- 将结果写入论文/专利 claim

## 数据安全

- 原始图片和标签**不入 Git**
- 模型权重(.pt/.pth)**不入 Git**
- GitHub 只放代码、配置、实验清单和报告
- 数据集放在服务器本地

## 日常操作循环

```
1. Mac 上用 Claude Code 或 Codex 改代码/配置
2. git commit + git push
3. 选定服务器 git pull 拉取最新代码
4. SSH 到选定服务器执行 python train_yolo.py train ...
5. 查看 runs/<task-name>/metrics.json 了解训练指标
6. Codex 读取结果并复盘
7. Codex 输出下一轮实验建议 (RESULTS_SUMMARY.md / NEXT_EXPERIMENTS.md)
8. 你确认高风险实验，继续迭代
```
