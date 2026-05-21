# IDEA_BRIEF: TPS-YOLO11 for Welding Tiny Defect Detection

Generated: 2026-05-21

## 1. Frozen Idea v1

This project targets a publishable object detection paper on welding tiny defect detection under industrially complex backgrounds.

Working method name:

- Overall framework: **TPS-YOLO11**  
  Tiny Defect and Pseudo-defect Suppression YOLO11
- Neck module: **FGDC-FPN**  
  Fine-Grained Detail Compensation Feature Pyramid Network
- Loss module: **SAHB-Loss**  
  Scale-Aware Hard Background Suppression Loss

## 2. Research Problem

Welding defect images contain many tiny or weak defects whose visual cues are easily submerged by weld texture, imaging noise, scratches, spatter-like patterns, and other pseudo-defect background regions. A plain YOLO11 detector may miss tiny defects because fine spatial details are weakened during feature fusion, and may also produce high-confidence false positives on background regions that resemble real defects.

The core problem is therefore:

> How can YOLO11 simultaneously improve tiny defect localization and suppress pseudo-defect background false positives in welding defect detection?

## 3. Core Hypothesis

For welding tiny defect detection, simply strengthening deep semantic features or stacking attention blocks is not sufficient to stably reduce missed detections and false positives. A more effective route is to preserve and compensate fine-grained shallow defect cues in the Neck while using a scale-aware loss to emphasize tiny-object localization errors and hard background confusions.

Expected effect:

- Higher recall and AP for tiny/small defects.
- Better localization quality, especially under high IoU thresholds.
- Lower high-confidence false positives on hard background regions.
- Better robustness on an auxiliary X-ray welding defect pressure test.

## 4. Method Direction

### 4.1 FGDC-FPN: Fine-Grained Detail Compensation Neck

Primary role:

- Improve multi-scale fine-grained fusion in YOLO11 Neck.
- Preserve tiny defect details from shallow/high-resolution features.
- Compensate weak defect cues that may be diluted during top-down and bottom-up fusion.
- Reduce the chance that pseudo-defect textures are incorrectly amplified as foreground.

Design constraints:

- The module should be located mainly in the Neck.
- Backbone and Head changes should be avoided unless necessary.
- The design should support clean ablations: baseline, +FGDC-FPN, +SAHB-Loss, +both.
- Complexity increase must be controlled and reported with parameters, FLOPs, and FPS.

### 4.2 SAHB-Loss: Scale-Aware Hard Background Suppression Loss

Primary role:

- Improve tiny-object localization by making the loss more sensitive to small-box errors.
- Add a hard-background suppression term or weight modulation for high-confusion background regions.
- Connect optimization directly to the failure mode: tiny defect localization and pseudo-defect false positives.

Design constraints:

- The loss should not depend on manually adding a new training dataset.
- Hard background samples should be mined from baseline YOLO11 false positives.
- The loss should be separable for ablation:
  - without scale-aware term
  - without hard-background term
  - full SAHB-Loss

## 5. Data Design

### 5.1 Main Dataset

Path:

`/Users/silya/vibe_coding/codex/Auto-research/data/welding-defect-detection-yolo`

Role:

- Main training, validation, comparison, and ablation dataset.

Classes:

- air-hole
- bite-edge
- broken-arc
- crack
- hollow-bead
- overlap
- slag-inclusion
- unfused

Current verified statistics:

- Train images: 2341
- Val images: 1003
- Train boxes: 4758
- Val boxes: 1959
- Train tiny boxes: 3325
- Val tiny boxes: 1322
- Dominant class: air-hole
- Minority classes: bite-edge, crack, slag-inclusion, hollow-bead, overlap, unfused

Why this is suitable:

- The dataset has a strong tiny-object distribution, which fits the paper's central claim.
- It contains multiple welding defect categories, allowing per-class analysis.
- Class imbalance should be explicitly reported and considered during training and evaluation.

### 5.2 Auxiliary Dataset

Path:

`/Users/silya/vibe_coding/codex/Auto-research/data/xray-welding-defect-yolo`

Role:

- Auxiliary cross-modality complex-background pressure test.
- Not a primary contribution.
- Not a direct pseudo-defect subset.

Classes:

- LOP
- creck
- porosity
- slag

Current verified statistics:

- Train images: 2422
- Val images: 24
- Train boxes: 22710
- Val boxes: 224
- Train tiny boxes: 5567
- Val tiny boxes: 63

Important caveat:

- The validation split is too small for a serious standalone conclusion. Before use, this dataset should be re-split into a more reliable train/val/test or train/test setting.
- Because the imaging domain is X-ray, it should be framed as robustness or cross-modality pressure testing, not as the pseudo-defect background subset.

### 5.3 Hard Background Subset

Role:

- Dedicated evaluation subset for pseudo-defect false positive suppression.
- Also used as the basis for SAHB-Loss dynamic hard-background weighting.

Construction plan:

1. Train a plain YOLO11 baseline on the main dataset.
2. Run inference on the main validation set, and optionally on held-out or additional welding images if available.
3. Collect high-confidence predictions with low IoU against ground truth.
4. Treat these regions as hard background or pseudo-defect candidates.
5. Review a representative subset to confirm that they correspond to weld texture, noise, scratches, spatter-like regions, or other defect-like background.
6. Evaluate false positives per image and high-confidence false positive count on this subset.

Recommended thresholds to tune during pilot:

- Prediction confidence: start with `conf >= 0.25` and inspect; use `conf >= 0.50` for high-confidence FP analysis.
- False positive condition: max IoU with ground truth `< 0.10` or `< 0.20`.
- Report both total FP/Image and high-confidence FP/Image if useful.

## 6. Evaluation Metrics

Standard detection metrics:

- Precision
- Recall
- mAP50
- mAP50-95

Tiny and small defect metrics:

- AP_tiny
- AP_small
- Recall_tiny
- Recall_small

Localization quality metrics:

- AP75
- AP75_tiny, if the implementation is stable

Pseudo-defect suppression metrics:

- FP/Image on hard background subset
- High-confidence FP count on hard background subset
- Optional: false positive reduction rate compared with YOLO11 baseline

Efficiency metrics:

- Parameters
- FLOPs
- FPS or latency
- Training/inference image size

## 7. Comparison Plan

Recommended comparison scope: medium-width, not an oversized model zoo.

Required baselines:

- YOLO11 baseline
- YOLOv8 or YOLOv10
- RT-DETR
- Faster R-CNN or Deformable DETR

Optional baselines if reproducible:

- 2-3 welding-defect or tiny-object detection methods with available code.
- Recent YOLO-based industrial defect detection variants, if training cost is acceptable.

Reporting rule:

- If a method cannot be reproduced fairly, do not force it into the main table. Mention it in related work instead.

## 8. Ablation Plan

Main ablation table:

| Setting | Purpose |
|---|---|
| YOLO11 baseline | reference point |
| + FGDC-FPN | test Neck contribution |
| + SAHB-Loss | test Loss contribution |
| + FGDC-FPN + SAHB-Loss | test full TPS-YOLO11 |
| + FGDC-FPN variant A/B | test detail compensation or fusion design choices |
| + SAHB-Loss without scale-aware term | isolate tiny-object localization effect |
| + SAHB-Loss without hard-background term | isolate pseudo-defect suppression effect |

Required analysis:

- Tiny/small AP changes.
- AP75 changes.
- Hard background FP/Image changes.
- Per-class gains, especially minority classes.
- Qualitative examples of missed detections corrected and false positives suppressed.

## 9. Expected Contributions

Contribution 1: Problem-level framing

- A YOLO11-based welding tiny defect detection framework for industrial complex backgrounds, explicitly targeting both tiny defect miss detection and pseudo-defect false positives.

Contribution 2: Method-level design

- FGDC-FPN, a fine-grained detail compensation feature fusion module in the YOLO11 Neck for tiny defect feature preservation and multi-scale interaction.

Contribution 3: Optimization and evaluation

- SAHB-Loss, combining scale-aware tiny-object localization modulation with hard-background suppression, plus a baseline-false-positive-driven hard background evaluation subset.

## 10. Main Risks and Backup Plans

Risk 1: FGDC-FPN improves mAP but not tiny AP.

- Backup: adjust the module to increase shallow feature participation or add explicit small-object branch weighting.
- Evidence to inspect: AP_tiny, AP_small, feature visualization, missed detection cases.

Risk 2: SAHB-Loss improves precision but hurts recall.

- Backup: reduce hard-background weight or make the term warm-up after early epochs.
- Evidence to inspect: precision-recall curve and per-class recall.

Risk 3: Hard background subset is too small or visually ambiguous.

- Backup: mine with multiple confidence and IoU thresholds, then report it as a hard-case evaluation protocol rather than a new dataset.

Risk 4: X-ray pressure test causes domain-drift confusion.

- Backup: clearly frame X-ray results as supplementary robustness evidence only, not as the main proof.

Risk 5: Runtime cost increases too much.

- Backup: use a lightweight variant of FGDC-FPN or restrict compensation to selected pyramid levels.

## 11. Next Step

The next document should be an experiment plan that converts this brief into executable runs:

1. Establish YOLO11 baseline on the main dataset.
2. Implement AP_tiny/AP_small and hard-background FP metrics.
3. Mine the first hard background subset from baseline predictions.
4. Implement FGDC-FPN.
5. Implement SAHB-Loss.
6. Run ablations.
7. Re-split and run the X-ray pressure test.

