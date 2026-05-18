# Selected Ideas

This file records ideas that pass Gate 1 and are allowed into the experiment matrix.

## Current Baseline Decision

Accepted method direction:

```text
WD-YOLO11 = morphology-aware localization loss + lightweight small-defect feature enhancement module
```

## Selection Table

| Idea ID | Name | Source | Status | Reason | Experiment Link |
|---|---|---|---|---|---|
| I-001 | Morphology-aware localization loss | discussion | selected | Directly targets small and elongated defect localization | experiments/experiment_matrix.yaml |
| I-002 | Lightweight small-defect feature enhancement | discussion | selected | Preserves fine defect cues and supports recall gains | experiments/experiment_matrix.yaml |
| I-003 | Safe augmentation and class balancing | discussion | auxiliary | Supports minority classes but should not be the main claim | experiments/experiment_matrix.yaml |
