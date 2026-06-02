# Parallel S-Idea Diagnostics Execution Report

Date: 2026-06-02

## Status

Implemented and verified the execution layer for the Parallel S-Idea diagnostic harness after PR #1 was merged.

This report is a tooling and smoke-execution record, not a research result. The route decision still requires running the diagnostics on the real Steel-Defect-Mixed dataset with the real baseline weights.

## What Changed

- Shared YOLO data helpers now parse `data.yaml`, resolve split image paths, map image files to YOLO label files, and load normalized GT boxes.
- CLR-YOLO diagnostic can build reliability records from `--predictions-json` for local validation or from YOLO weights when available.
- OSD-YOLO diagnostic can build a leave-one-class split, preserve unknown labels for validation, evaluate predictions from JSON, and run YOLO inference when weights are available.
- FASD-YOLO audit now traverses labeled images and scores edge-proxy masks per GT box.
- Diagnostic outputs are JSON-serializable even when records contain `Box` values.

## Verification

Focused tests:

```bash
/Users/silya/vibe_coding/codex/Auto-research/.venv/bin/python -m pytest \
  tests/test_idea_diag_common.py \
  tests/test_clr_label_reliability.py \
  tests/test_osd_leave_one_class.py \
  tests/test_fasd_teacher_audit.py \
  tests/test_rank_parallel_ideas.py \
  tests/test_experiment_matrix.py \
  tests/test_ssh_submit.py -v
```

Result: `36 passed in 0.70s`.

Matrix rendering:

```bash
/Users/silya/vibe_coding/codex/Auto-research/.venv/bin/python tools/experiment_matrix.py \
  --matrix research/yolo11_welding_defect/experiments/parallel_s_idea_diagnostics.yaml \
  --phase parallel_s_idea_diagnostics \
  --print-commands
```

Result: rendered CLR, OSD, FASD, and ranking commands.

SSH dry-run:

```bash
/Users/silya/vibe_coding/codex/Auto-research/.venv/bin/python tools/ssh_submit.py \
  --matrix research/yolo11_welding_defect/experiments/parallel_s_idea_diagnostics.yaml \
  --phase parallel_s_idea_diagnostics \
  --remote-root '~/ar' \
  --dry-run
```

Result: printed three remote diagnostic commands and skipped local-only `idea_rank`.

## Local Smoke

A temporary synthetic YOLO dataset under `/tmp/parallel_s_smoke` was used to verify the full local diagnostic path:

- CLR wrote non-empty `summary.json`, `reliability_records.json`, and `report.md`.
- OSD wrote non-empty `summary.json`, `unknown_proposals.json`, and `report.md`.
- FASD wrote non-empty `summary.json`, `teacher_quality_records.json`, and `report.md`.
- Ranking wrote `ranking.md`.

The local smoke ranking recommended OSD-YOLO on the synthetic one-sample dataset. This is only proof that the execution chain works; it is not evidence that OSD-YOLO should be selected for the thesis route.

## Next Real Execution

Run the matrix on the Linux GPU host once `~/ar/data/steel-defect-mixed/data.yaml` and `~/ar/runs/yolo/baseline/weights/best.pt` are available:

```bash
python tools/ssh_submit.py \
  --matrix research/yolo11_welding_defect/experiments/parallel_s_idea_diagnostics.yaml \
  --phase parallel_s_idea_diagnostics \
  --remote-root '~/ar'
```

After the three diagnostic summaries exist on the host, run:

```bash
python tools/rank_parallel_ideas.py \
  --clr-summary idea-stage/artifacts/parallel_s/clr_label_reliability/summary.json \
  --osd-summary idea-stage/artifacts/parallel_s/osd_holdout_surface_classes/summary.json \
  --fasd-summary idea-stage/artifacts/parallel_s/fasd_edge_proxy_teacher_audit/summary.json \
  --out idea-stage/artifacts/parallel_s/ranking.md
```
