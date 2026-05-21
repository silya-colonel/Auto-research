# EXPERIMENT_PLAN: TPS-YOLO11

Generated: 2026-05-21

Source brief: `idea-stage/IDEA_BRIEF.md`

## 1. Objective

Turn the frozen idea **TPS-YOLO11** into an executable experiment sequence for a publishable welding tiny defect detection paper.

The plan tests three claims:

1. **Tiny defect claim**: TPS-YOLO11 improves tiny/small defect detection over YOLO11.
2. **Localization claim**: SAHB-Loss improves high-IoU localization quality, especially for tiny boxes.
3. **Pseudo-defect suppression claim**: TPS-YOLO11 reduces high-confidence false positives on hard background regions mined from baseline YOLO11 failures.

## 2. Fixed Paths

Repository root:

`/Users/silya/vibe_coding/codex/Auto-research`

Main dataset:

`data/welding-defect-detection-yolo/data.yaml`

Auxiliary X-ray dataset:

`data/xray-welding-defect-yolo/data.yaml`

Primary training entry:

`train_yolo.py`

Default output roots:

- Training runs: `runs/yolo`
- Validation runs: `runs/val`
- Prediction runs: `runs/predict`
- Experiment artifacts: `idea-stage/artifacts`

## 3. Global Settings

Primary model:

- `yolo11n.pt` for fast screening and ablation.
- `yolo11s.pt` for final confirmation if the method works on `n`.

Image sizes:

- `640` for baseline and first screening.
- `960` for tiny-object confirmation, because the main dataset contains many tiny boxes.

Seeds:

- Screening: `42`
- Final runs: `42`, `3407`, `2026` if compute allows.

Epochs:

- Smoke test: `3`
- Screening: `20`
- Main ablation: `100`
- Final confirmation: `150` only if 100-epoch results are positive but not saturated.

Core metrics:

- Precision
- Recall
- mAP50
- mAP50-95
- AP75
- AP_tiny
- AP_small
- Recall_tiny
- Recall_small
- FP/Image on hard background subset
- High-confidence FP/Image on hard background subset
- Params, FLOPs, FPS or latency

## 4. Phase 0: Environment and Data Sanity

Goal:

Confirm that the current repository can train YOLO11 on the main dataset without path or dependency problems.

Run from:

```bash
cd /Users/silya/vibe_coding/codex/Auto-research
```

Smoke command:

```bash
python train_yolo.py train \
  --task-name smoke_yolo11n_welding_640_e3 \
  --data-yaml data/welding-defect-detection-yolo/data.yaml \
  --model yolo11n.pt \
  --imgsz 640 \
  --epochs 3 \
  --batch -1 \
  --workers 4 \
  --seed 42 \
  --runs-dir runs/yolo
```

Pass criteria:

- Training starts and completes.
- Validation runs after training.
- A `metrics.json` file is written under the run directory.

If it fails:

- Fix data path or dependency issues before touching model design.

## 5. Phase 1: Baseline Establishment

Goal:

Build strong YOLO11 baselines before implementing any new method.

### 1.1 Main baseline at 640

```bash
python train_yolo.py train \
  --task-name baseline_yolo11n_welding_640_e100_s42 \
  --data-yaml data/welding-defect-detection-yolo/data.yaml \
  --model yolo11n.pt \
  --imgsz 640 \
  --epochs 100 \
  --batch -1 \
  --workers 8 \
  --seed 42 \
  --runs-dir runs/yolo
```

### 1.2 Tiny-object image-size baseline at 960

```bash
python train_yolo.py train \
  --task-name baseline_yolo11n_welding_960_e100_s42 \
  --data-yaml data/welding-defect-detection-yolo/data.yaml \
  --model yolo11n.pt \
  --imgsz 960 \
  --epochs 100 \
  --batch -1 \
  --workers 8 \
  --seed 42 \
  --runs-dir runs/yolo
```

### 1.3 Capacity check with YOLO11s

```bash
python train_yolo.py train \
  --task-name baseline_yolo11s_welding_640_e100_s42 \
  --data-yaml data/welding-defect-detection-yolo/data.yaml \
  --model yolo11s.pt \
  --imgsz 640 \
  --epochs 100 \
  --batch -1 \
  --workers 8 \
  --seed 42 \
  --runs-dir runs/yolo
```

Decision gate:

- If `960` gives a large tiny-AP gain with acceptable speed, keep both `640` and `960` in the paper.
- If `yolo11s` only improves by capacity, use `yolo11n` for main ablation and `yolo11s` as final scalability evidence.

## 6. Phase 2: Evaluation Tooling

Goal:

Add metrics that directly match the paper claims before testing the method.

Required tools:

1. **Tiny/small AP evaluator**
   - Area bins:
     - tiny: normalized box area `< 0.0005`
     - small: `0.0005 <= area < 0.0025`
     - medium: `0.0025 <= area < 0.01`
     - large: `>= 0.01`
   - Outputs:
     - AP_tiny
     - AP_small
     - Recall_tiny
     - Recall_small
     - optional AP75_tiny
   - Implemented entry:
     - `python tools/yolo_area_metrics.py --labels <labels_dir> --predictions <predictions.json> --out <area_metrics.json>`

2. **Hard background miner**
   - Inputs:
     - trained baseline weights
     - main dataset validation images and labels
   - Candidate condition:
     - prediction confidence `>= 0.25`
     - max IoU with any ground-truth box `< 0.10` or `< 0.20`
   - Outputs:
     - image path
     - predicted class
     - confidence
     - predicted box
     - max IoU
     - crop path if crop export is implemented
   - Implemented entry:
     - `python tools/yolo_hard_background.py mine --labels <labels_dir> --predictions <predictions.json> --conf 0.25 --max-iou 0.10 --out <candidates.csv>`

3. **Hard background FP evaluator**
   - Inputs:
     - hard background candidate list
     - model weights
   - Outputs:
     - FP/Image
     - high-confidence FP/Image at `conf >= 0.50`
     - false positive reduction rate versus baseline
   - Implemented entry:
     - `python tools/yolo_hard_background.py evaluate --candidates <candidates.csv> --predictions <model_predictions.json> --baseline-predictions <baseline_predictions.json> --out <hard_bg_metrics.json>`

Suggested artifact paths:

- `idea-stage/artifacts/hard_background/baseline_candidates.csv`
- `idea-stage/artifacts/metrics/baseline_yolo11n_welding_640_area_metrics.json`
- `idea-stage/artifacts/metrics/baseline_yolo11n_welding_640_hard_bg.json`

Decision gate:

- Do not begin full method ablation until AP_tiny/AP_small and hard-background FP metrics work on the baseline.

## 7. Phase 3: Hard Background Subset Mining

Goal:

Create the pseudo-defect evaluation subset from real YOLO11 failure cases.

Baseline weights:

`runs/yolo/baseline_yolo11n_welding_640_e100_s42/weights/best.pt`

Mining steps:

1. Run baseline inference on the main validation set.
2. Match each prediction to ground truth by IoU.
3. Keep high-confidence predictions with low IoU.
4. Save metadata and optional crops.
5. Manually inspect a small representative sample to confirm that common sources include weld texture, noise, scratches, spatter-like regions, or other defect-like background.

Minimum useful subset:

- At least 100 hard background predictions, or
- At least 50 images with high-confidence false positives.

If too few candidates appear:

- Lower mining confidence from `0.50` to `0.25`.
- Use `max IoU < 0.20` instead of `< 0.10`.
- Include baseline predictions from the training split only for mining analysis, not for final evaluation.

## 8. Phase 4: FGDC-FPN Implementation and Screening

Goal:

Test whether the Neck module alone improves tiny defect metrics.

Implementation target:

- Add FGDC-FPN as a YOLO11 Neck replacement or Neck-inserted module.
- Prefer a YAML-selectable design so the training entry can still use `train_yolo.py`.
- Keep Backbone and Detection Head unchanged for the first version.

Implemented local entry:

- Module: `tools/yolo_custom_modules.py::FGDC`
- Model config: `configs/tps_yolo11n_fgdc.yaml`
- Training model argument: `--model configs/tps_yolo11n_fgdc.yaml`
- Verification: `ultralytics==8.4.52` and `torch==2.12.0` can construct the YAML as a `DetectionModel` with 27 layers after `register_yolo_modules()`.

Screening run:

```bash
python train_yolo.py train \
  --task-name fgdc_fpn_yolo11n_welding_640_e20_s42 \
  --data-yaml data/welding-defect-detection-yolo/data.yaml \
  --model configs/tps_yolo11n_fgdc.yaml \
  --pretrained-weights yolo11n.pt \
  --imgsz 640 \
  --epochs 20 \
  --batch -1 \
  --workers 8 \
  --seed 42 \
  --runs-dir runs/yolo
```

Pass criteria:

- AP_tiny or Recall_tiny improves over the matched 20-epoch YOLO11 baseline.
- mAP50-95 does not drop sharply.
- Params/FLOPs increase is explainable.

If it fails:

- Try a lighter or more direct detail compensation branch.
- Increase shallow feature participation.
- Avoid adding generic attention unless it is tied to tiny-detail compensation.

## 9. Phase 5: SAHB-Loss Implementation and Screening

Goal:

Test whether the loss alone improves tiny localization and hard background false positives.

Implementation target:

- Extend the existing loss patch path in `tools/yolo_custom_modules.py`.
- Keep the API compatible with `train_yolo.py --extra`.
- First implement separable options:
  - scale-aware localization term
  - hard-background suppression term
  - full SAHB-Loss

Implemented local behavior:

- `custom_iou_loss=sahb` enables scale-aware box/DFL weighting in `BboxLoss`.
- `sahb_scale_weight=<float>` controls how strongly tiny boxes are up-weighted.
- `sahb_hard_bg_weight=<float>` enables focal BCE by default when positive, acting as the first hard-background suppression proxy for difficult negative/classification examples.
- Explicit `custom_cls_loss=<value>` still overrides the default SAHB focal behavior.

Suggested command shape:

```bash
python train_yolo.py train \
  --task-name sahb_loss_yolo11n_welding_640_e20_s42 \
  --data-yaml data/welding-defect-detection-yolo/data.yaml \
  --model yolo11n.pt \
  --imgsz 640 \
  --epochs 20 \
  --batch -1 \
  --workers 8 \
  --seed 42 \
  --runs-dir runs/yolo \
  --extra custom_iou_loss=sahb sahb_scale_weight=1.0 sahb_hard_bg_weight=0.5
```

The exact `--extra` keys can change during implementation, but the final keys must be documented.

Pass criteria:

- AP75 or AP75_tiny improves.
- FP/Image or high-confidence FP/Image decreases on the hard background subset.
- Recall does not collapse.

If it fails:

- Warm up hard-background weighting after early epochs.
- Reduce hard-background weight.
- Keep the scale-aware part and demote the hard-background part if it is unstable.

## 10. Phase 6: Full TPS-YOLO11 Ablation

Goal:

Prove that FGDC-FPN and SAHB-Loss each contribute and work together.

Implemented matrix entry:

```bash
python tools/tps_experiment_matrix.py \
  --matrix idea-stage/tps_experiment_matrix.yaml \
  --print-commands
```

Generated command lists:

- `idea-stage/tps_screening_commands.sh`
- `idea-stage/tps_ablation_commands.sh`

Main 100-epoch ablation table:

| ID | Variant | Model | Loss | Epochs |
|---|---|---|---|---:|
| A0 | YOLO11 baseline | `yolo11n.pt` | default | 100 |
| A1 | + FGDC-FPN | `tps_yolo11_fgdc.yaml` | default | 100 |
| A2 | + SAHB-Loss | `yolo11n.pt` | SAHB-Loss | 100 |
| A3 | Full TPS-YOLO11 | `tps_yolo11_fgdc.yaml` | SAHB-Loss | 100 |
| A4 | FGDC-FPN variant | variant YAML | default or SAHB | 100 |
| A5 | SAHB without scale-aware term | best model | partial SAHB | 100 |
| A6 | SAHB without hard-bg term | best model | partial SAHB | 100 |

Primary decision criteria:

- Full TPS-YOLO11 should improve mAP50-95 and AP_tiny/AP_small over A0.
- A1 should mainly support tiny/small AP and recall.
- A2 should mainly support AP75 and hard-background FP reduction.
- A5/A6 should show why both SAHB subterms matter.

Claim discipline:

- If only A3 works, claim component interaction carefully.
- If A1 does not work independently, demote FGDC-FPN or redesign it.
- If A2 lowers recall too much, narrow the loss claim to precision/false-positive suppression or tune weights.

## 11. Phase 7: Final Confirmation Runs

Goal:

Verify that gains are not one-seed artifacts.

Minimum confirmation:

- Re-run A0 and A3 with seeds `3407` and `2026`.

Recommended confirmation if compute allows:

- Re-run A0 and A3 at `imgsz=960`.
- Re-run A0 and A3 with `yolo11s.pt` scale.

Report:

- Mean and standard deviation for key metrics.
- At minimum, report whether all seeds preserve the sign of improvement.

## 12. Phase 8: Auxiliary X-ray Pressure Test

Goal:

Evaluate whether the method has robustness value beyond the main surface-welding-style dataset.

Important:

- The current X-ray validation split has only 24 images, so it should be re-split before serious reporting.
- This experiment is supplementary, not a main contribution.

Steps:

1. Create a stable split from `xray-welding-defect-yolo`.
2. Train matched baseline and TPS-YOLO11 under the same settings.
3. Report relative gains, not overstate cross-modality generalization.

Implemented split entry:

```bash
python tools/yolo_split_dataset.py \
  data/xray-welding-defect-yolo \
  --out data/xray-welding-defect-yolo-split \
  --train 0.70 \
  --val 0.15 \
  --test 0.15 \
  --seed 42
```

Recommended split:

- Train: 70%
- Val: 15%
- Test: 15%

Minimum runs:

| Variant | Dataset | Model | Epochs |
|---|---|---|---:|
| X0 | X-ray baseline | YOLO11n | 100 |
| X1 | X-ray TPS-YOLO11 | FGDC-FPN + SAHB-Loss | 100 |

## 13. Phase 9: Qualitative and Failure Analysis

Implemented qualitative case export:

```bash
python tools/tps_qualitative_cases.py \
  --labels data/welding-defect-detection-yolo/labels/val \
  --baseline runs/predict/baseline/predictions.json \
  --tps runs/predict/tps/predictions.json \
  --images data/welding-defect-detection-yolo/images/val \
  --out idea-stage/artifacts/qualitative/cases.json
```

Required visual evidence:

- Tiny defects missed by YOLO11 but detected by TPS-YOLO11.
- Pseudo-defect background regions falsely detected by YOLO11 but suppressed by TPS-YOLO11.
- Cases where TPS-YOLO11 still fails.
- X-ray supplementary examples, clearly marked as auxiliary.

Required analysis:

- Per-class AP/recall table.
- Tiny/small/medium/large area-bin table.
- Hard background FP table.
- Complexity table.
- PR curves or confidence-threshold sensitivity if time allows.

## 14. Paper-Ready Result Tables

Implemented aggregation entry:

```bash
python tools/tps_result_tables.py \
  --runs runs/yolo \
  --extra-metrics idea-stage/artifacts/metrics \
  --out idea-stage/artifacts/tables
```

Implemented outputs:

- `main_comparison.md` and `main_comparison.csv`
- `ablation.md` and `ablation.csv`
- `tiny_object.md` and `tiny_object.csv`
- `hard_background.md` and `hard_background.csv`
- `efficiency.md` and `efficiency.csv`

Table 1: Main comparison

- YOLO11 baseline
- YOLOv8 or YOLOv10
- RT-DETR
- Faster R-CNN or Deformable DETR
- TPS-YOLO11

Table 2: Ablation

- A0 to A6 from Phase 6

Table 3: Tiny-object and localization analysis

- AP_tiny
- AP_small
- AP75
- AP75_tiny if available

Table 4: Hard background suppression

- FP/Image
- high-confidence FP/Image
- false positive reduction rate

Table 5: Efficiency

- params
- FLOPs
- FPS or latency

Supplementary table:

- X-ray pressure test baseline vs TPS-YOLO11

## 15. Stop/Go Gates

Gate 1: Baseline valid

- YOLO11 baseline trains cleanly.
- Baseline metrics and per-class metrics are saved.

Gate 2: Metrics valid

- AP_tiny/AP_small evaluator works.
- Hard background miner and FP evaluator work.

Gate 3: Method signal

- At least one of FGDC-FPN or SAHB-Loss shows positive 20-epoch signal.

Gate 4: Full ablation valid

- Full TPS-YOLO11 improves matched baseline on main metrics without unacceptable efficiency cost.

Gate 5: Paper claim valid

- Gains are supported by at least one final 100-epoch run and preferably multiple seeds.
- Ablation supports the claimed modules.
- Hard background results support the pseudo-defect suppression claim.

## 16. Immediate Next Actions

1. Run the 3-epoch smoke test.
2. Run the 100-epoch YOLO11n 640 baseline.
3. Implement AP_tiny/AP_small evaluator.
4. Mine baseline hard background false positives.
5. Implement FGDC-FPN.
6. Implement SAHB-Loss.
