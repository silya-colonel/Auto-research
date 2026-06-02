# P45.3 — OSD Leave-One-Class Protocol

## Parent

P45 — Parallel S-Idea Diagnostics

## What to build

Build the OSD-YOLO open-set diagnostic protocol. It should support held-out
unknown defect classes, remove unknown labels from training splits when building
protocol data, and evaluate class-agnostic unknown-defect proposal recall.

## Acceptance criteria

- [ ] `tools/osd_leave_one_class.py` exists.
- [ ] `tests/test_osd_leave_one_class.py` verifies label filtering, unknown-class candidate selection, and class-agnostic unknown recall.
- [ ] The CLI help includes `--data-yaml`, `--unknown-class`, `--output-root`, `--mode build-split`, and `--mode evaluate`.
- [ ] Evaluation writes `summary.json`, `unknown_proposals.json`, and `report.md`.
- [ ] The focused tests pass with `pytest tests/test_osd_leave_one_class.py -v`.

## Blocked by

P45.1 — Shared Diagnostic Helpers.

## Timeout

AFK, 120s unit tests; GPU diagnostic timeout depends on model inference.
