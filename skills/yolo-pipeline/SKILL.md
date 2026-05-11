---
name: yolo-pipeline
description: Linux-native YOLO defect detection pipeline for Ultralytics YOLO with ClearML, GitHub sync, Claude Code execution, Codex review, and automatic experiment planning. Use when the user says "YOLO defect detection", "defect detection", "ClearML YOLO", "缺陷检测", or wants ARIS to automate a YOLO experiment lifecycle.
argument-hint: [project-brief-or-data-yaml]
allowed-tools: Bash(*), Read, Grep, Glob, Edit, Write, Agent, Skill(research-lit), Skill(experiment-plan), Skill(run-experiment), Skill(analyze-results), Skill(result-to-claim), Skill(ablation-planner), Skill(paper-writing), Skill(patent-pipeline), Skill(grant-proposal), Skill(paper-poster), Skill(paper-slides), Skill(vast-gpu)
---

# YOLO Pipeline

Run a YOLO defect detection project with **Mac orchestration + GitHub private repo + Linux training + ClearML tracking + ARIS automation**.

Two deployment modes:
- **linux-lab**: Own lab Linux server with conda, systemd, ClearML Agent
- **cloud-gpu**: Rented cloud GPU (AutoDL/Vast.ai) with auto-shutdown

Claude Code handles execution/training dispatch. Codex handles result review/paper production.

## Default Stack

- **Developer machine**: Mac for editing, Claude Code/Codex orchestration, GitHub sync.
- **Code sync**: GitHub private repository (code, configs, manifests, small reports only).
- **Training target**: Linux server (lab-owned or cloud GPU).
- **Experiment tracker**: ClearML (SaaS or self-hosted).
- **Detector framework**: Ultralytics YOLO.
- **Agent**: ClearML Agent (systemd on lab server, manual on cloud GPU).

## Automation Boundary

ARIS can automate:
- inspect `data.yaml`, class names, image/label folders, and label health
- propose baseline and tuning grids
- generate training commands for SSH remote or ClearML Agent submission
- collect metrics and artifacts from ClearML
- compare runs and recommend next experiment wave
- turn validated evidence into paper, patent, grant, poster, and slides material

ARIS must pause before:
- changing dataset taxonomy or relabeling policy
- making expensive cloud GPU purchases
- claiming a method is novel or paper-worthy from weak results
- replacing the detector framework
- landing invasive architecture/loss changes without baseline evidence

## Required Project Inputs

```text
AGENTS.md
experiments/
results/
```

## Workflow — 8 Phases

### Phase 0: Project Scaffold

Use `tools/init_yolo_project.sh` to create the project skeleton. Verify:
- `.gitignore` covers datasets, weights, ClearML cache
- `experiments/` for run manifests
- `results/` for summarized outputs
- `AGENTS.md` with server config

### Phase 1: Data Health Check

Inspect `data.yaml` (located with the dataset, e.g. `~/datasets/defect/data.yaml`):
- train/val/test paths resolve on Linux
- class count matches `names`
- labels have 5 YOLO columns, normalized coordinates
- report empty labels, missing images, duplicates, extreme imbalance, tiny boxes

Use `tools/yolo_data_health.py`:
```bash
python tools/yolo_data_health.py ~/datasets/defect/data.yaml \
  --markdown-out results/DATA_HEALTH_REPORT.md
```

### Phase 2: Server Environment

**linux-lab**: Run `tools/setup_linux_server.sh` on the server (or via SSH from Mac).
**cloud-gpu**: Run `tools/cloud_init.sh` after provisioning the instance.

Verify: CUDA visible, ClearML connected, watchdog running.

### Phase 3: Baseline Matrix

Minimum baseline wave:
- `smoke_1epoch` (verify end-to-end)
- `baseline_yolo11n_640` (fast baseline)
- `baseline_yolo11s_640` (capacity check)
- `baseline_yolo11n_960` (small-defect sensitivity)

Submit via SSH or ClearML Agent. Track all in ClearML.

### Phase 4: Result Review (Codex)

Codex reads ClearML metrics and produces:
- `results/RUN_TABLE.csv`
- `results/RESULTS_SUMMARY.md`
- `results/NEXT_EXPERIMENTS.md`

Identifies: best model, weakest class, false positives/negatives, resolution gaps, class imbalance.

### Phase 5: Tuning Wave

Based on review, run 3-8 targeted experiments:
- image size, augmentation, model capacity
- class balancing, confidence/NMS thresholds

### Phase 6: Architecture & Loss (optional)

Only after baselines are stable. Each candidate needs:
- Hypothesis, exact code change, expected metric movement, risk, revert plan.

### Phase 7: Writing Handoff

Codex produces `NARRATIVE_REPORT.md`, then routes to:
- `/paper-writing`, `/patent-pipeline`, `/grant-proposal`, `/paper-poster`, `/paper-slides`

Must separate: confirmed evidence / plausible interpretation / unsupported claims.

## Key Rules

- GitHub: code/config/reports only, no datasets or weights
- Always smoke test (1 epoch) before batch experiments
- No architecture/loss changes without baseline evidence
- Cloud GPU: always set auto-shutdown timer
- Claude Code executes, Codex reviews — maintain separation of concerns
