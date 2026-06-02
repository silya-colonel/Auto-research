# P45.2 — CLR Label Reliability Diagnostic

## Parent

P45 — Parallel S-Idea Diagnostics

## What to build

Build the CLR-YOLO diagnostic CLI that estimates label reliability from
prediction consistency and checks whether low-reliability labels explain a
meaningful fraction of baseline errors on Steel-Defect-Mixed.

## Acceptance criteria

- [ ] `tools/clr_label_reliability.py` exists.
- [ ] `tests/test_clr_label_reliability.py` covers high-reliability, low-reliability, and gate-summary behavior.
- [ ] The CLI help includes `--weights`, `--data-yaml`, `--out-dir`, `--views`, and `--min-error-explained`.
- [ ] A diagnostic run writes `summary.json`, `reliability_records.json`, and `report.md`.
- [ ] The focused tests pass with `pytest tests/test_clr_label_reliability.py -v`.

## Blocked by

P45.1 — Shared Diagnostic Helpers.

## Timeout

AFK, 120s unit tests; GPU diagnostic timeout depends on dataset size.
