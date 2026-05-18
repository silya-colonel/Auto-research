# Related Work Plan

## Source Rule

Formal related work must cite papers from the Zotero export.

AI-discovered papers must first enter:

```text
../literature/candidate_papers.md
```

## Method Paper Buckets

### YOLO-Based Welding Defect Detection

Purpose:

Show that YOLO is a practical and widely used detector family for welding defect detection, but existing work often focuses on generic accuracy improvements or dataset-specific tuning.

Needed evidence:

```text
welding defect datasets
YOLO versions used
defect categories
reported metrics
limitations
```

### Industrial Surface Defect Detection

Purpose:

Place welding defects in the broader industrial defect detection landscape, including steel surfaces, magnetic tiles, and X-ray defects.

Needed evidence:

```text
domain shift
texture defects
small anomalies
class imbalance
transfer validation practices
```

### Small-Object Detection

Purpose:

Motivate high-resolution feature preservation and small-target-aware optimization.

Needed evidence:

```text
feature pyramid limitations
shallow feature detail
scale imbalance
small object metrics
```

### Elongated and Crack-Like Defect Detection

Purpose:

Motivate aspect and morphology-aware localization.

Needed evidence:

```text
crack detection
narrow objects
elongated bounding boxes
center distance and aspect sensitivity
```

### Localization Losses

Purpose:

Ground the WDLoss design in existing IoU-family and shape-aware losses.

Needed evidence:

```text
DIoU
CIoU
Wise-IoU
Shape-IoU or shape-aware variants
small-object weighting
quality focal or localization-quality alignment
```

### Attention, Neck, and Feature Fusion

Purpose:

Ground the lightweight small-defect feature module.

Needed evidence:

```text
neck refinement
feature fusion
attention modules
edge or texture guidance
lightweight deployment constraints
```

## Survey Paper Buckets

The survey paper can use a broader taxonomy:

```text
YOLO version evolution
welding defect modalities
datasets
augmentation
attention and feature fusion
loss functions
lightweight deployment
small-object and crack-like defects
generalization and transfer
open challenges
```
