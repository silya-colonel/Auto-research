# Claims Ledger

This file prevents the paper from making claims before the experiments support them.

## Claim Rules

1. Every numerical claim must map to a raw result file.
2. Every comparative claim must identify the baseline.
3. Every scope claim must identify the dataset or datasets where it holds.
4. Every literature claim must cite a Zotero-backed reference.
5. Failed or negative transfer results must not be hidden.

## Pending Claims

| Claim ID | Claim | Status | Evidence Needed |
|---|---|---|---|
| C-001 | WDLoss improves localization quality on the main welding dataset. | pending | baseline vs loss-only mAP50-95 and per-class AP |
| C-002 | The feature module improves recall for small or elongated defect classes. | pending | baseline vs module-only per-class recall/AP |
| C-003 | Combining loss and feature module gives the best overall performance. | pending | full ablation matrix |
| C-004 | The method transfers across industrial defect datasets. | pending | at least two transfer datasets with positive gains |
| C-005 | The method remains lightweight enough for practical use. | pending | params, FLOPs, FPS/latency |

## Claim Status Values

```text
pending
supported
partially_supported
unsupported
rejected
```

## Evidence Links

Add links after experiments complete:

```text
runs/yolo/<run_id>/metrics.json
research/yolo11_welding_defect/experiments/results_table.csv
research/yolo11_welding_defect/experiments/figures/
```
