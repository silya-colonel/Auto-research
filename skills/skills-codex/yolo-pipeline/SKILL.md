---
name: yolo-pipeline
description: Linux-native YOLO defect detection pipeline for Ultralytics YOLO with ClearML, GitHub sync, automatic experiment planning, result analysis, and paper/patent/fund handoff. Use when the user says "YOLO defect detection", "defect detection", "ClearML YOLO", "缺陷检测", or wants ARIS to automate a YOLO experiment lifecycle.
argument-hint: [project-brief-or-data-yaml]
---

# YOLO Pipeline

Run a YOLO defect detection project with **Mac orchestration + GitHub private repo + Linux training + ClearML + ARIS/Codex automation**.

This Codex-native skill is the primary entrypoint for YOLO defect projects. Claude Code handles training execution; Codex handles result review and paper production.

Human-facing full setup guide: `docs/YOLO_FULL_WORKFLOW_CN.md`.

## Default Stack

- **Developer machine**: Mac for editing, AI orchestration, literature work, GitHub sync.
- **Code sync**: GitHub private repository (code, configs, manifests, small reports only).
- **Training target**: Linux server — lab-owned or cloud GPU (AutoDL, Vast.ai).
- **Experiment tracker**: ClearML SDK + ClearML Agent.
- **Detector framework**: Ultralytics YOLO.

## Automation Boundary

ARIS can automate:

- inspect `data.yaml`, class names, image/label folders, label health
- propose baseline and tuning grids
- generate training commands (SSH remote or ClearML Agent submission)
- collect metrics and artifacts from ClearML
- compare runs and recommend next experiment wave
- turn validated evidence into paper, patent, grant, poster, slides

ARIS must pause before:

- changing dataset taxonomy or relabeling policy
- making expensive cloud GPU purchases
- claiming novelty from weak results
- replacing the detector framework
- invasive architecture/loss changes without baseline evidence

## Required Project Inputs

```text
AGENTS.md
experiments/
results/
```

Recommended `AGENTS.md` project block:

```markdown
## YOLO Project
- task: YOLO defect detection
- framework: ultralytics
- server_type: linux-lab | cloud-gpu
- server_host: <ssh-host-or-ip>
- server_user: <user>
- data_yaml: ~/datasets/defect/data.yaml
- clearml_project: yolo
- clearml_queue: yolo-linux
- code_sync: git
- train_target: linux-native
- gpu_policy: single-gpu-first
- human_checkpoint: architecture-change, cloud-gpu, paper-claim
- large_files_policy: do-not-git-data-or-weights
```

## Workflow

### Phase 0: Project Scaffold

Use `tools/init_yolo_project.sh` to create the project. Verify:
- `.gitignore` covers datasets, weights, ClearML cache
- `experiments/`, `results/` directories exist
- `AGENTS.md` has server config

### Phase 1: Data Health Check

Inspect `data.yaml` (located with the dataset, e.g. `~/datasets/defect/data.yaml`):
- train/val/test paths resolve on Linux
- class count matches `names`
- labels have 5 YOLO columns, normalized coordinates
- report empty labels, missing images, duplicates, extreme class imbalance, tiny boxes

```bash
python tools/yolo_data_health.py ~/datasets/defect/data.yaml \
  --markdown-out results/DATA_HEALTH_REPORT.md
```

### Phase 2: Baseline Matrix

```yaml
project: yolo
stage: baseline
framework: ultralytics
runs:
  - id: smoke_1epoch
    command: python train_yolo_clearml.py train --task-name smoke_1epoch --data-yaml ~/datasets/defect/data.yaml --model yolo11n.pt --epochs 1
  - id: baseline_yolo11n_640
    command: python train_yolo_clearml.py train --task-name baseline_yolo11n_640 --data-yaml ~/datasets/defect/data.yaml --model yolo11n.pt --imgsz 640 --epochs 100
  - id: baseline_yolo11s_640
    command: python train_yolo_clearml.py train --task-name baseline_yolo11s_640 --data-yaml ~/datasets/defect/data.yaml --model yolo11s.pt --imgsz 640 --epochs 100
  - id: baseline_yolo11n_960
    command: python train_yolo_clearml.py train --task-name baseline_yolo11n_960 --data-yaml ~/datasets/defect/data.yaml --model yolo11n.pt --imgsz 960 --epochs 100
```

### Phase 3: Linux Execution

**lab server (direct SSH)**:
```bash
ssh lab-server "cd ~/work/yolo && conda activate yolo && \
  python train_yolo_clearml.py train --task-name baseline_yolo11n_640 \
  --data-yaml ~/datasets/defect/data.yaml --model yolo11n.pt --epochs 100"
```

**lab server (ClearML Agent)**:
```bash
clearml-task --project yolo --name baseline_yolo11n_640 \
  --repo git@github.com:<user>/yolo-defect.git --branch main \
  --queue yolo-linux --script train_yolo_clearml.py \
  --args "--task-name baseline_yolo11n_640 --data-yaml ~/datasets/defect/data.yaml --model yolo11n.pt --epochs 100"
```

**cloud GPU (direct)**:
```bash
ssh -p <port> root@<instance> \
  "cd /root/autodl-tmp/yolo && \
   python train_yolo_clearml.py train --task-name baseline_yolo11n_640 \
   --data-yaml /root/autodl-tmp/datasets/defect/data.yaml \
   --model yolo11n.pt --epochs 100 --shutdown-when-done"
```

### Phase 4: YOLO Tuning Wave

After baseline results, run 3-8 targeted experiments:
- image size: 640, 960, 1280
- augmentation: conservative vs stronger
- model size: n, s, m
- class imbalance: sampling or loss weighting
- decision thresholds: confidence/NMS for recall-first vs precision-first

### Phase 5: Architecture and Loss Candidates

Only after baselines are stable. For each candidate:
```markdown
- Hypothesis:
- Exact code/config change:
- Expected metric movement:
- Risk:
- Revert plan:
- Minimum run needed:
```

### Phase 6: Result Analysis and Claim Routing

Collect into:
```text
results/RESULTS_SUMMARY.md
results/RUN_TABLE.csv
results/NEXT_EXPERIMENTS.md
```

Route: continue tuning / ablation / data fix / writing prep.

### Phase 7: Writing and Extension Handoff

Generate `NARRATIVE_REPORT.md`, then:
- `/paper-writing` for manuscript drafting
- `/patent-pipeline` for invention disclosure
- `/grant-proposal` for fund applications
- `/paper-poster` and `/paper-slides` for presentation

Must separate: confirmed evidence / plausible interpretation / unsupported claims.

## Key Rules

- GitHub: code/config/report sync, NOT datasets or large weights.
- Linux training via SSH remote or ClearML Agent queues.
- Start with baselines before custom modules.
- Prefer few, well-motivated experiments over large blind sweeps.
- Cloud GPU: always set auto-shutdown.
- Claude Code executes training. Codex reviews results. Maintain separation.
