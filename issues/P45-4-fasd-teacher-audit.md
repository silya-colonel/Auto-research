# P45.4 — FASD Teacher Audit

## Parent

P45 — Parallel S-Idea Diagnostics

## What to build

Build the FASD-YOLO teacher-quality audit. The first provider should be an
edge/contrast proxy teacher that can audit whether GT-box regions contain
usable boundary-style supervision before any external foundation model is added.

## Acceptance criteria

- [ ] `tools/fasd_teacher_audit.py` exists.
- [ ] `tests/test_fasd_teacher_audit.py` verifies edge-proxy mask extraction and mask-quality scoring.
- [ ] The CLI help includes `--data-yaml`, `--split`, `--out-dir`, `--provider`, `--max-samples`, and `--min-usable-rate`.
- [ ] A diagnostic run writes `summary.json`, `teacher_quality_records.json`, `report.md`, and optional previews when requested.
- [ ] The focused tests pass with `pytest tests/test_fasd_teacher_audit.py -v`.

## Blocked by

P45.1 — Shared Diagnostic Helpers.

## Timeout

AFK, 120s unit tests; audit runtime depends on sample count.
