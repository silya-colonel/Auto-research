# Method Paper Outline

## Working Title

WD-YOLO11: Morphology-Aware YOLO11 for Small and Elongated Welding Defect Detection

## Target Story

Welding defect detection is difficult because defects are often small, elongated, low contrast, and imbalanced across classes. WD-YOLO11 improves localization and recall by combining a morphology-aware localization loss with a lightweight small-defect feature enhancement module.

## Draft Structure

1. Abstract
   - Problem: small and elongated welding defects.
   - Method: morphology-aware loss plus lightweight feature module.
   - Evidence: main welding dataset and transfer datasets.

2. Introduction
   - Industrial importance of welding defect detection.
   - Limits of generic YOLO11 for small and elongated defects.
   - Contributions.

3. Related Work
   - YOLO-based welding defect detection.
   - Industrial surface defect detection.
   - Small-object detection.
   - Elongated/crack-like defect detection.
   - Localization losses for object detection.
   - Lightweight attention, neck, and feature fusion.

4. Method
   - Baseline YOLO11.
   - Welding defect morphology-aware localization loss.
   - Lightweight small-defect feature enhancement module.
   - Training and inference pipeline.

5. Experiments
   - Datasets and preprocessing.
   - Metrics.
   - Baselines.
   - Main results on welding defect dataset.
   - Ablation studies.
   - Transferability validation.
   - Efficiency analysis.

6. Discussion
   - Which defect classes benefit most.
   - Failure cases.
   - Generality and limits.

7. Conclusion
   - Supported claims only.

## Contributions Draft

Do not finalize until experiments support them.

1. A morphology-aware localization loss for small and elongated welding defects.
2. A lightweight feature enhancement module for preserving fine defect cues in YOLO11.
3. A systematic validation across welding and industrial defect datasets, including ablation and transferability analysis.

## Linked Survey Paper

The survey paper should provide:

```text
YOLO welding defect taxonomy
dataset table
method family comparison
small-object and elongated-defect challenges
open problems that motivate WD-YOLO11
```

The method paper should reuse only audited citations from the survey literature source.
