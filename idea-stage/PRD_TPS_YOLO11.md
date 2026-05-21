# PRD: TPS-YOLO11 焊接微小缺陷检测实验系统

生成日期：2026-05-21

关联文档：

- `idea-stage/IDEA_BRIEF.md`
- `idea-stage/EXPERIMENT_PLAN.md`

## Problem Statement

用户现在已经冻结了一个可投稿论文 idea：基于 YOLO11 的焊接微小缺陷检测方法 `TPS-YOLO11`，核心是通过 `FGDC-FPN` 改进 Neck 特征融合，并通过 `SAHB-Loss` 提升小目标定位质量和伪缺陷背景误检抑制能力。

当前问题不是“有没有 idea”，而是 idea 还没有被拆成可执行、可测试、可复现的工程任务。现有仓库已经有 YOLO 数据集、训练入口、数据健康检查和 ClearML 训练路线，但还缺少支撑论文主张的关键工程能力：小目标分组指标、hard background 误检挖掘、伪缺陷误检评估、X-ray 辅助数据重划分、FGDC-FPN 模块、SAHB-Loss、消融运行矩阵和论文表格汇总。

如果不先把这些能力拆清楚，后续容易直接进入长训练或架构改动，导致实验不可解释、结果不可复现，甚至论文主张和指标证据脱节。

## Solution

构建一套围绕 `TPS-YOLO11` 的实验实现路线，把论文 idea 拆成一组可独立交给 agent 执行的纵向 issue。每个 issue 都必须提供一个可观察行为，并配套 TDD 风格的测试策略：先通过公开接口验证行为，再实现最小代码，最后再重构。

解决方案包含：

- 用现有 YOLO11 训练入口建立 baseline。
- 增加 area-bin 指标，直接评估 tiny/small 缺陷效果。
- 从 baseline 真实误检中挖掘 hard background subset。
- 增加 hard background false-positive 评估能力。
- 增加 X-ray 数据重划分能力，把 X-ray 数据作为补充压力测试而不是主贡献。
- 实现 `FGDC-FPN`，使 YOLO11 Neck 能进行细粒度特征补偿。
- 实现 `SAHB-Loss`，使训练过程对小框定位和 hard background 混淆更敏感。
- 增加消融矩阵、结果汇总和论文表格导出。

## User Stories

1. As a researcher, I want to run a YOLO11 baseline on the welding dataset, so that I can establish a trustworthy reference before changing the model.
2. As a researcher, I want baseline metrics saved in a stable format, so that later experiments can be compared without manually reading logs.
3. As a researcher, I want AP and recall split by tiny/small object bins, so that the paper can prove tiny defect improvements directly.
4. As a researcher, I want AP75 and AP75_tiny, so that the paper can support localization-quality claims.
5. As a researcher, I want per-class metrics, so that minority defect classes are not hidden by dominant air-hole results.
6. As a researcher, I want to mine high-confidence false positives from baseline YOLO11, so that the hard background subset reflects real model failures.
7. As a researcher, I want hard background candidates saved with confidence, class, box, and IoU metadata, so that I can audit the pseudo-defect failure cases.
8. As a researcher, I want false positives per image on hard background cases, so that the paper can quantify pseudo-defect suppression.
9. As a researcher, I want high-confidence false-positive counts, so that I can show whether the method suppresses risky industrial false alarms.
10. As a researcher, I want X-ray welding data re-split into reliable train/val/test splits, so that the auxiliary pressure test is not based on only 24 validation images.
11. As a researcher, I want the X-ray experiment framed as supplementary robustness evidence, so that the paper does not overclaim cross-modality generalization.
12. As a researcher, I want `FGDC-FPN` selectable through model configuration, so that I can compare baseline and Neck variants fairly.
13. As a researcher, I want `FGDC-FPN` to keep Backbone and Head unchanged initially, so that the Neck contribution is isolated.
14. As a researcher, I want `FGDC-FPN` complexity reported, so that accuracy gains can be balanced against deployment cost.
15. As a researcher, I want `SAHB-Loss` selectable through training options, so that I can compare default loss and proposed loss fairly.
16. As a researcher, I want the scale-aware and hard-background parts of `SAHB-Loss` separable, so that ablation can prove which part matters.
17. As a researcher, I want short screening runs before long training, so that weak method variants are eliminated cheaply.
18. As a researcher, I want 100-epoch ablations after screening, so that paper claims are based on mature training results.
19. As a researcher, I want multiple seeds for final confirmation, so that gains are not one lucky run.
20. As a researcher, I want result tables generated consistently, so that writing the paper does not depend on manual spreadsheet work.
21. As a researcher, I want qualitative examples of corrected misses and suppressed false positives, so that the paper can visually support the quantitative claims.
22. As a researcher, I want failure cases retained, so that the paper can honestly describe remaining limitations.
23. As a researcher, I want issue-level TDD instructions, so that different agents can implement tasks without drifting from the paper objective.
24. As a researcher, I want each issue to be independently verifiable, so that progress can be merged safely.
25. As a researcher, I want the implementation to respect the Linux/ClearML route, so that local planning aligns with the actual training environment.

## Implementation Decisions

- `TPS-YOLO11` remains the umbrella method name, with `FGDC-FPN` as the Neck component and `SAHB-Loss` as the optimization component.
- Main training and validation use YOLO-format welding defect data with YOLO11.
- X-ray welding data is used only as supplementary pressure testing, and must be re-split before serious reporting.
- Hard background samples are mined from baseline YOLO11 false positives, not hand-picked as a new training dataset.
- Evaluation tooling is a first-class part of the method, because the paper's claims require metrics beyond generic mAP.
- Area bins follow the existing data-health convention: tiny, small, medium, and large by normalized bounding-box area.
- Public command-line tools should be preferred for metrics, mining, split generation, and reporting, so tests can exercise behavior through stable interfaces.
- `FGDC-FPN` should be configurable and ablatable without changing the training script interface.
- `SAHB-Loss` should integrate through existing training options and preserve the default loss path.
- Final paper claims must be gated by ablation evidence, hard background false-positive results, and efficiency reporting.

## Testing Decisions

- Tests should verify external behavior through command-line tools, public functions, or saved artifacts, not private implementation details.
- The first test for each module should be a small tracer-bullet test using synthetic YOLO labels or tiny fake prediction files.
- Data parsing and metric tools should be tested with temporary directories and small deterministic fixtures.
- Model architecture tests should avoid long training; they should check that the model can be constructed and, where possible, can run a tiny forward path or dry validation.
- Loss tests should verify observable effects on simple tensors and training-option selection, not the exact internal call graph.
- Result aggregation tests should verify table content and missing-run handling from small fake metrics files.
- Existing tests already use `unittest`, temporary directories, and public tool interfaces; new tests should follow that pattern unless pytest fixtures become clearly useful.

## Out of Scope

- Launching long GPU training runs without explicit confirmation.
- Relabeling class definitions or changing dataset class policy.
- Treating X-ray pressure-test results as a main contribution.
- Writing the final paper manuscript.
- Claiming novelty before literature and experimental evidence are aligned.
- Publishing GitHub issues automatically before the issue tracker label workflow is confirmed.

## Further Notes

The immediate engineering route should not start with model architecture. It should start with baseline sanity, metric tooling, and hard-background mining. Those pieces define whether later `FGDC-FPN` and `SAHB-Loss` gains are meaningful.

