# Transfer Validation Plan

## Decision

Transferability validation will use per-dataset training comparisons.

Do not use one model trained on the main welding dataset for direct zero-shot evaluation on datasets with incompatible label spaces.

## Transfer Claim

Allowed claim shape:

```text
The proposed components show transferable effectiveness across multiple industrial defect detection datasets when trained under the same protocol.
```

Avoid claim shape:

```text
The model generalizes zero-shot from welding defects to all industrial defects.
```

## Required Datasets

| Dataset | Role | Required Status |
|---|---|---|
| Main welding COCO | Main dataset | Convert to YOLO and run full matrix |
| NEU-DET | Transfer dataset | Verify YOLO data.yaml |
| Magnetic tile | Transfer dataset | Ready, has baseline history |
| X-ray COCO | Transfer dataset | Convert to YOLO and inspect split |

Optional:

| Dataset | Role | Note |
|---|---|---|
| radiographs-welding | Optional large transfer dataset | Use if class labels and compute budget are manageable |

## Per-Dataset Comparison

For each transfer dataset:

```text
YOLO11 baseline
YOLO11 + selected loss
YOLO11 + selected feature module
YOLO11 + selected loss + selected feature module
```

## Result Interpretation

Positive transfer:

The improved model beats baseline on mAP50-95 and does not severely harm precision or efficiency.

Partial transfer:

The improved model boosts recall or weak-class AP but has tradeoffs. Claim only the supported behavior.

Negative transfer:

Report honestly. Analyze whether the failure comes from dataset domain shift, label noise, object scale, class imbalance, or module overfitting.
