# AI-Researcher Socratic Mode

Use this mode when giving AI-Researcher a high-level problem description and asking it to propose directions.

## Problem Prompt

```text
We are improving YOLO11 for welding defect detection. The main challenge is robust detection of small, elongated, low-contrast, and class-imbalanced defects such as cracks, air holes, slag inclusion, unfused regions, overlap, hollow bead, bite edge, and broken arc.

The method paper should focus on two core contributions:
1. Welding defect morphology-aware localization loss.
2. Lightweight small-defect feature enhancement module.

The primary metric is mAP50-95. The second primary target is recall or AP improvement for small and elongated defect classes. Efficiency must be reported with parameters, FLOPs, and FPS or latency.

Please propose concrete improvement directions that are likely to increase mAP50-95 and small/elongated defect recall without making the model unnecessarily heavy.
```

## Required Output

Ask AI-Researcher to return:

```text
idea_name
motivation
target_failure_mode
method_components
expected_metric_gain
implementation_complexity
ablation_needed
risk
papers_needed
```

## Filtering Rules

Reject ideas that:

1. Require rewriting the entire detector.
2. Are only name-level attention insertion without a defect-specific reason.
3. Add heavy modules without efficiency control.
4. Cannot be ablated into baseline, loss-only, module-only, and combined variants.
5. Depend on citations not present in Zotero or the candidate pool.
