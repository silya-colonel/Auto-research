# Issue Breakdown and TDD Implementation Plan

生成日期：2026-05-21

来源：

- `idea-stage/PRD_TPS_YOLO11.md`
- `idea-stage/IDEA_BRIEF.md`
- `idea-stage/EXPERIMENT_PLAN.md`

说明：

- 下面的 issue body 可用于发布到 GitHub issue tracker。
- 每个 issue 后面的 “TDD 实现计划” 是本地执行说明，比正式 issue body 更具体，包含建议文件和接口草案。
- 当前未自动发布到 GitHub，因为没有确认 `ready-for-agent` 标签和 issue tracker 工作流。

## Issue 1: Establish YOLO11 Baseline Smoke Path

Type: AFK

Blocked by: None

User stories covered: 1, 2, 17, 23, 24, 25

### Tracker Body

#### What to build

Create a verified baseline smoke path for training YOLO11 on the welding defect dataset. The completed slice should prove that the current training entry can run, validate, and save metrics for a short run before any method code is changed.

#### Acceptance criteria

- [ ] A smoke baseline command is documented and runnable.
- [ ] The run writes a stable metrics artifact after validation.
- [ ] The smoke path uses the main welding dataset and YOLO11.
- [ ] The result can be checked without reading raw training logs.
- [ ] The issue does not modify architecture or loss behavior.

#### Blocked by

None - can start immediately.

### TDD 实现计划

Public interface:

```bash
python train_yolo.py train --task-name smoke_yolo11n_welding_640_e3 ...
```

RED:

- Add a test that invokes the training command in a dry or mocked mode and expects a metrics JSON artifact path to be produced.

GREEN:

- If needed, add a lightweight run-summary helper that writes the expected metrics schema from validation results.

Refactor:

- Keep metrics writing isolated behind one public helper so later result aggregation can reuse it.

Suggested files:

- `tests/test_train_yolo_metrics.py`
- `train_yolo.py`

Code interface sketch:

```python
def save_metrics(save_dir: Path, task_name: str, metrics: dict[str, Any]) -> Path:
    ...
```

Verification:

```bash
python -m unittest tests.test_train_yolo_metrics
```

## Issue 2: Add Area-Bin Detection Metrics

Type: AFK

Blocked by: Issue 1

Status: implemented locally

User stories covered: 3, 4, 5, 20, 23, 24

### Tracker Body

#### What to build

Add a metrics tool that evaluates detection performance by object size bins. It should report tiny and small defect performance in a stable artifact format suitable for experiment comparison and paper tables.

#### Acceptance criteria

- [ ] The tool accepts YOLO labels and model predictions or exported detections.
- [ ] It reports AP_tiny, AP_small, Recall_tiny, Recall_small, and AP75 where supported.
- [ ] Area bins match the project data-health convention.
- [ ] The output is deterministic for a small fixture dataset.
- [ ] Tests verify behavior through the public tool interface.

#### Blocked by

Issue 1.

### TDD 实现计划

Public interface:

```bash
python tools/yolo_area_metrics.py \
  --labels data/welding-defect-detection-yolo/labels/val \
  --predictions runs/predict/baseline/predictions.json \
  --out idea-stage/artifacts/metrics/area_metrics.json
```

RED:

- Create a tiny fixture with two images, YOLO labels, and prediction JSON.
- Assert that a perfect tiny prediction gives `AP_tiny = 1.0` or the chosen simplified fixture equivalent.
- Assert that a missing tiny prediction reduces tiny recall.

GREEN:

- Implement label loading, prediction loading, IoU matching, area-bin assignment, and per-bin summaries.

Refactor:

- Extract reusable matching logic for hard-background mining.

Suggested files:

- `tools/yolo_area_metrics.py`
- `tests/test_yolo_area_metrics.py`

Implemented files:

- `tools/yolo_area_metrics.py`
- `tests/test_yolo_area_metrics.py`

Code interface sketch:

```python
@dataclass(frozen=True)
class Detection:
    image_id: str
    cls: int
    conf: float
    xywh: tuple[float, float, float, float]

def evaluate_area_bins(labels: list[Detection], predictions: list[Detection]) -> dict[str, Any]:
    ...
```

Verification:

```bash
python -m unittest tests.test_yolo_area_metrics
```

## Issue 3: Mine Hard Background False Positives

Type: AFK

Blocked by: Issue 2

Status: implemented locally

User stories covered: 6, 7, 8, 9, 21, 22, 23, 24

### Tracker Body

#### What to build

Create a hard-background miner that identifies high-confidence baseline predictions with low overlap against ground truth. The output should become the pseudo-defect evaluation subset for false-positive suppression analysis.

#### Acceptance criteria

- [ ] The miner accepts YOLO labels and exported predictions.
- [ ] It writes a candidate list with image, class, confidence, box, and max IoU.
- [ ] Confidence and IoU thresholds are configurable.
- [ ] The same fixture produces deterministic hard-background candidates.
- [ ] Tests verify that true positives are excluded and low-IoU high-confidence predictions are included.

#### Blocked by

Issue 2.

### TDD 实现计划

Public interface:

```bash
python tools/yolo_hard_background.py mine \
  --labels data/welding-defect-detection-yolo/labels/val \
  --predictions runs/predict/baseline/predictions.json \
  --conf 0.25 \
  --max-iou 0.10 \
  --out idea-stage/artifacts/hard_background/baseline_candidates.csv
```

RED:

- Fixture: one ground-truth box, one overlapping prediction, one distant high-confidence prediction.
- Assert only the distant prediction is mined.

GREEN:

- Reuse IoU/matching code from area metrics.
- Write CSV output with stable columns.

Refactor:

- Move shared YOLO box parsing into a small module if duplication appears.

Suggested files:

- `tools/yolo_hard_background.py`
- `tests/test_yolo_hard_background.py`
- optional `tools/yolo_eval_core.py`

Implemented files:

- `tools/yolo_hard_background.py`
- `tests/test_yolo_hard_background.py`

Code interface sketch:

```python
def mine_hard_background(
    labels: list[Detection],
    predictions: list[Detection],
    conf_threshold: float,
    max_iou_threshold: float,
) -> list[HardBackgroundCandidate]:
    ...
```

Verification:

```bash
python -m unittest tests.test_yolo_hard_background
```

## Issue 4: Evaluate Hard Background False Positives

Type: AFK

Blocked by: Issue 3

Status: implemented locally

User stories covered: 8, 9, 20, 21, 23, 24

### Tracker Body

#### What to build

Add an evaluator that computes false-positive rates on the mined hard-background subset. The evaluator should compare a candidate model against baseline behavior and produce paper-ready suppression metrics.

#### Acceptance criteria

- [ ] The evaluator reads a hard-background candidate list and model predictions.
- [ ] It reports FP/Image and high-confidence FP/Image.
- [ ] It reports false-positive reduction rate when a baseline artifact is provided.
- [ ] It handles empty or missing predictions gracefully.
- [ ] Tests cover improved, unchanged, and worse false-positive behavior.

#### Blocked by

Issue 3.

### TDD 实现计划

Public interface:

```bash
python tools/yolo_hard_background.py evaluate \
  --candidates idea-stage/artifacts/hard_background/baseline_candidates.csv \
  --predictions runs/predict/tps/predictions.json \
  --baseline-predictions runs/predict/baseline/predictions.json \
  --out idea-stage/artifacts/metrics/tps_hard_bg.json
```

RED:

- Fixture with two hard-background images.
- Candidate model removes one high-confidence FP.
- Assert FP/Image and reduction rate match expected values.

GREEN:

- Implement candidate grouping and prediction overlap/threshold checks.

Refactor:

- Keep miner and evaluator in one CLI with subcommands but separate functions.

Suggested files:

- `tools/yolo_hard_background.py`
- `tests/test_yolo_hard_background.py`

Implemented files:

- `tools/yolo_hard_background.py`
- `tests/test_yolo_hard_background.py`

Code interface sketch:

```python
def evaluate_hard_background(
    candidates: list[HardBackgroundCandidate],
    predictions: list[Detection],
    high_conf: float = 0.50,
) -> dict[str, float]:
    ...
```

Verification:

```bash
python -m unittest tests.test_yolo_hard_background
```

## Issue 5: Re-split X-ray Welding Dataset for Pressure Test

Type: AFK

Blocked by: Issue 1

Status: implemented locally

User stories covered: 10, 11, 23, 24

### Tracker Body

#### What to build

Create a reproducible split utility for the X-ray welding defect dataset so supplementary pressure-test results do not rely on the current tiny validation split.

#### Acceptance criteria

- [ ] The utility creates train/val/test splits from an existing YOLO dataset.
- [ ] Image-label pairing is preserved.
- [ ] The split is deterministic with a seed.
- [ ] A new data configuration is written for the split output.
- [ ] Tests verify deterministic split counts and label pairing.

#### Blocked by

Issue 1.

### TDD 实现计划

Public interface:

```bash
python tools/yolo_split_dataset.py \
  data/xray-welding-defect-yolo \
  --out data/xray-welding-defect-yolo-split \
  --train 0.70 \
  --val 0.15 \
  --test 0.15 \
  --seed 42
```

RED:

- Fixture with 10 image-label pairs.
- Assert split counts are deterministic and every output image has its label.

GREEN:

- Implement file pairing, seeded shuffle, copy/link mode, and data.yaml generation.

Refactor:

- Share image extension constants with data-health tooling if useful.

Suggested files:

- `tools/yolo_split_dataset.py`
- `tests/test_yolo_split_dataset.py`

Implemented files:

- `tools/yolo_split_dataset.py`
- `tests/test_yolo_split_dataset.py`

Code interface sketch:

```python
def split_yolo_dataset(root: Path, out: Path, ratios: SplitRatios, seed: int) -> SplitSummary:
    ...
```

Verification:

```bash
python -m unittest tests.test_yolo_split_dataset
```

## Issue 6: Implement FGDC-FPN as a Configurable Neck Module

Type: HITL

Blocked by: Issue 2

Status: implemented locally; Ultralytics YAML construction verified with `ultralytics==8.4.52` and `torch==2.12.0`

User stories covered: 12, 13, 14, 17, 18, 23, 24

### Tracker Body

#### What to build

Implement the first version of `FGDC-FPN` as a configurable YOLO11 Neck component. The slice should enable a short screening run that isolates the Neck contribution without changing Backbone, Head, or loss behavior.

#### Acceptance criteria

- [ ] The module can be selected through model configuration.
- [ ] A model using the module can be constructed by the training entry.
- [ ] Backbone and Head remain unchanged in the first version.
- [ ] A 20-epoch screening command is documented.
- [ ] Complexity reporting remains available.

#### Blocked by

Issue 2.

### TDD 实现计划

Public interface:

```bash
python train_yolo.py train \
  --model configs/tps_yolo11n_fgdc.yaml \
  --pretrained-weights yolo11n.pt \
  ...
```

RED:

- Add a test that registers custom YOLO modules and constructs the configured model.
- If full Ultralytics construction is too heavy for local tests, first test module input/output tensor shape through the public module class.

GREEN:

- Add `FGDC` or `FGDCBlock` to the custom module registry.
- Add a YOLO YAML variant that inserts the module in the Neck.

Refactor:

- Keep the module interface small: input tensor(s) in, enhanced tensor out.
- Avoid hardcoding dataset-specific class counts into the module.

Suggested files:

- `tools/yolo_custom_modules.py`
- `configs/tps_yolo11n_fgdc.yaml`
- `tests/test_yolo_custom_modules.py`

Implemented files:

- `tools/yolo_custom_modules.py`
- `configs/tps_yolo11n_fgdc.yaml`
- `tests/test_yolo_custom_modules.py`

Code interface sketch:

```python
class FGDC(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...
```

Verification:

```bash
python -m unittest tests.test_yolo_custom_modules
```

## Issue 7: Implement SAHB-Loss Training Option

Type: HITL

Blocked by: Issue 3, Issue 4

Status: implemented locally; first hard-background term uses focal BCE as the classification-side hard-example proxy

User stories covered: 15, 16, 17, 18, 23, 24

### Tracker Body

#### What to build

Implement `SAHB-Loss` as an opt-in training behavior that can be enabled without changing the default YOLO11 path. The loss must expose separable scale-aware and hard-background terms for ablation.

#### Acceptance criteria

- [ ] Default loss behavior remains unchanged unless explicitly enabled.
- [ ] Scale-aware localization can be enabled independently.
- [ ] Hard-background suppression can be enabled independently.
- [ ] Full `SAHB-Loss` can be enabled through training options.
- [ ] Tests verify option selection and simple loss-weighting behavior.

#### Blocked by

Issue 3 and Issue 4.

### TDD 实现计划

Public interface:

```bash
python train_yolo.py train \
  --extra custom_iou_loss=sahb sahb_scale_weight=1.0 sahb_hard_bg_weight=0.5
```

RED:

- Test default path: no `custom_iou_loss` leaves loss patch inactive.
- Test opt-in path: `custom_iou_loss=sahb` installs SAHB behavior.
- Test simple tensor case where smaller boxes receive larger localization weight.

GREEN:

- Extend loss patching to parse SAHB options.
- Implement scale-aware weighting first.
- Add hard-background term only after mining artifacts are defined.

Refactor:

- Separate option parsing from loss math.
- Keep fallback path to CIoU/default simple and safe.

Suggested files:

- `tools/yolo_custom_modules.py`
- `train_yolo.py`
- `tests/test_yolo_sahb_loss.py`

Implemented files:

- `tools/yolo_custom_modules.py`
- `train_yolo.py`
- `tests/test_yolo_sahb_loss.py`

Code interface sketch:

```python
def patch_detection_sahb_loss(
    enabled: bool,
    scale_weight: float,
    hard_bg_weight: float,
    warmup_epochs: int = 0,
) -> None:
    ...
```

Verification:

```bash
python -m unittest tests.test_yolo_sahb_loss
```

## Issue 8: Add Experiment Matrix and Command Generation

Type: AFK

Blocked by: Issue 6, Issue 7

Status: implemented locally

User stories covered: 17, 18, 19, 20, 23, 24, 25

### Tracker Body

#### What to build

Create a reproducible experiment matrix for TPS-YOLO11 baseline, component ablations, final confirmation, and X-ray pressure testing. The matrix should produce consistent commands or run records for AFK execution.

#### Acceptance criteria

- [ ] The matrix includes baseline, FGDC-FPN only, SAHB-Loss only, full TPS-YOLO11, and loss subterm ablations.
- [ ] Screening and final-run phases are clearly separated.
- [ ] Commands include dataset, model, image size, epochs, seed, and output name.
- [ ] Missing dependencies between runs are explicit.
- [ ] Tests verify that command generation is deterministic.

#### Blocked by

Issue 6 and Issue 7.

### TDD 实现计划

Public interface:

```bash
python tools/tps_experiment_matrix.py \
  --matrix idea-stage/tps_experiment_matrix.yaml \
  --print-commands
```

RED:

- Fixture matrix with two runs.
- Assert generated commands include required fields and stable order.

GREEN:

- Implement YAML loading and command rendering.

Refactor:

- Keep run schema narrow and explicit.

Suggested files:

- `idea-stage/tps_experiment_matrix.yaml`
- `tools/tps_experiment_matrix.py`
- `tests/test_tps_experiment_matrix.py`

Implemented files:

- `idea-stage/tps_experiment_matrix.yaml`
- `idea-stage/tps_screening_commands.sh`
- `idea-stage/tps_ablation_commands.sh`
- `tools/tps_experiment_matrix.py`
- `tests/test_tps_experiment_matrix.py`

Code interface sketch:

```python
@dataclass(frozen=True)
class ExperimentRun:
    id: str
    dataset: str
    model: str
    imgsz: int
    epochs: int
    seed: int
    extra: tuple[str, ...] = ()
```

Verification:

```bash
python -m unittest tests.test_tps_experiment_matrix
```

## Issue 9: Aggregate Results into Paper-Ready Tables

Type: AFK

Blocked by: Issue 2, Issue 4, Issue 8

Status: implemented locally

User stories covered: 20, 21, 22, 23, 24

### Tracker Body

#### What to build

Add a result aggregation tool that reads run metrics, area-bin metrics, hard-background metrics, and efficiency metrics, then exports paper-ready comparison and ablation tables.

#### Acceptance criteria

- [ ] The aggregator reads multiple run metric artifacts.
- [ ] It creates main comparison, ablation, tiny-object, hard-background, and efficiency tables.
- [ ] Missing optional metrics are represented clearly rather than crashing.
- [ ] Output is available in Markdown and CSV.
- [ ] Tests verify table output from small fake metric artifacts.

#### Blocked by

Issue 2, Issue 4, and Issue 8.

### TDD 实现计划

Public interface:

```bash
python tools/tps_result_tables.py \
  --runs runs/yolo \
  --extra-metrics idea-stage/artifacts/metrics \
  --out idea-stage/artifacts/tables
```

RED:

- Fixture metrics for baseline and TPS.
- Assert Markdown table contains expected run IDs and metric columns.
- Assert missing hard-background metric becomes `NA`, not an exception.

GREEN:

- Implement metrics collection and table rendering.

Refactor:

- Keep table schemas explicit and paper-oriented.

Suggested files:

- `tools/tps_result_tables.py`
- `tests/test_tps_result_tables.py`

Implemented files:

- `tools/tps_result_tables.py`
- `tests/test_tps_result_tables.py`

Code interface sketch:

```python
def build_result_tables(run_metrics: list[dict[str, Any]]) -> dict[str, str]:
    ...
```

Verification:

```bash
python -m unittest tests.test_tps_result_tables
```

## Issue 10: Export Qualitative Failure Analysis Cases

Type: AFK

Blocked by: Issue 3, Issue 4, Issue 9

Status: implemented locally

User stories covered: 21, 22, 23, 24

### Tracker Body

#### What to build

Create a qualitative analysis exporter that collects representative corrected misses, suppressed false positives, remaining failures, and auxiliary X-ray examples for paper figures and diagnosis.

#### Acceptance criteria

- [ ] The exporter reads baseline and TPS prediction artifacts.
- [ ] It identifies candidate improved detections and suppressed false positives.
- [ ] It saves a manifest of selected cases.
- [ ] It can optionally export image crops or visualization inputs.
- [ ] Tests verify case categorization from small synthetic prediction fixtures.

#### Blocked by

Issue 3, Issue 4, and Issue 9.

### TDD 实现计划

Public interface:

```bash
python tools/tps_qualitative_cases.py \
  --labels data/welding-defect-detection-yolo/labels/val \
  --baseline runs/predict/baseline/predictions.json \
  --tps runs/predict/tps/predictions.json \
  --out idea-stage/artifacts/qualitative/cases.json
```

RED:

- Fixture where baseline misses a ground-truth box but TPS detects it.
- Fixture where baseline has a false positive but TPS suppresses it.
- Assert cases are categorized correctly.

GREEN:

- Implement case matching and manifest writing.

Refactor:

- Reuse shared detection parsing and IoU utilities.

Suggested files:

- `tools/tps_qualitative_cases.py`
- `tests/test_tps_qualitative_cases.py`

Implemented files:

- `tools/tps_qualitative_cases.py`
- `tests/test_tps_qualitative_cases.py`

Code interface sketch:

```python
def select_qualitative_cases(
    labels: list[Detection],
    baseline_predictions: list[Detection],
    candidate_predictions: list[Detection],
) -> dict[str, list[Case]]:
    ...
```

Verification:

```bash
python -m unittest tests.test_tps_qualitative_cases
```

## Dependency Order

1. Issue 1: Establish YOLO11 Baseline Smoke Path
2. Issue 2: Add Area-Bin Detection Metrics
3. Issue 3: Mine Hard Background False Positives
4. Issue 4: Evaluate Hard Background False Positives
5. Issue 5: Re-split X-ray Welding Dataset for Pressure Test
6. Issue 6: Implement FGDC-FPN as a Configurable Neck Module
7. Issue 7: Implement SAHB-Loss Training Option
8. Issue 8: Add Experiment Matrix and Command Generation
9. Issue 9: Aggregate Results into Paper-Ready Tables
10. Issue 10: Export Qualitative Failure Analysis Cases

## TDD Priority

Start with Issue 2 before model architecture. The highest-value first implementation slice is area-bin metrics, because it defines the measurable target for both `FGDC-FPN` and `SAHB-Loss`.

Recommended first coding loop:

1. RED: `test_yolo_area_metrics` proves a perfect tiny prediction is counted as tiny success.
2. GREEN: implement minimal label/prediction parsing and tiny recall.
3. RED: add false prediction fixture.
4. GREEN: implement IoU matching.
5. RED: add AP-like confidence ordering or simplified AP behavior.
6. GREEN: complete area-bin summary artifact.
