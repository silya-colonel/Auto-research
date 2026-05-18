# Ablation Plan

## Purpose

The ablation study must prove which parts of WD-YOLO11 actually matter.

## Core Ablations

| ID | Variant | Purpose |
|---|---|---|
| A0 | YOLO11 baseline | Reference point |
| A1 | Loss only | Test morphology-aware localization contribution |
| A2 | Feature module only | Test small-defect feature contribution |
| A3 | Loss + feature module | Test combined contribution |
| A4 | Loss + feature module + safe augmentation | Test auxiliary data strategy |

## Loss Ablations

| ID | Variant | Purpose |
|---|---|---|
| L0 | Baseline box loss | Control |
| L1 | DIoU or DIoU-like loss | Test distance-aware localization |
| L2 | Small-object bounded weight | Test small target emphasis |
| L3 | Elongated aspect term | Test crack-like and narrow defect localization |
| L4 | L2 + L3 | Test small and elongated morphology together |
| L5 | Full WDLoss | Test final loss design |

## Feature Ablations

| ID | Variant | Purpose |
|---|---|---|
| F0 | Baseline neck/head | Control |
| F1 | High-resolution small-defect branch | Preserve shallow fine detail |
| F2 | Edge or texture guided attention | Emphasize low-contrast defect cues |
| F3 | Lightweight neck attention | Test efficient feature recalibration |
| F4 | Selected feature module | Final module |

## Required Reporting

For every accepted ablation:

```text
run_id
dataset
model
imgsz
epochs
seed
mAP50
mAP50-95
precision
recall
per-class AP/recall
params
FLOPs
FPS or latency
notes
```

## Reviewer Risk Checks

1. If only the combined model improves, isolate whether gain comes from loss, module, or augmentation.
2. If mAP50 improves but mAP50-95 drops, do not overclaim localization quality.
3. If recall improves but precision collapses, report false-positive tradeoff.
4. If a component hurts transfer datasets, narrow the claim to the main dataset or remove the component.
