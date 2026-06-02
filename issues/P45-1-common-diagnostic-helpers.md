# P45.1 — Shared Diagnostic Helpers

## Parent

P45 — Parallel S-Idea Diagnostics

## What to build

Create shared helper functions for the new idea diagnostics so CLR-YOLO,
OSD-YOLO, and FASD-YOLO use the same label keys, class-agnostic IoU matching,
gate summaries, and JSON writing behavior.

## Acceptance criteria

- [ ] `tools/idea_diag_common.py` exists.
- [ ] `tests/test_idea_diag_common.py` verifies stable label keys, class-agnostic best IoU, gate pass/fail behavior, and JSON writing.
- [ ] The focused tests pass with `pytest tests/test_idea_diag_common.py -v`.

## Blocked by

None - can start immediately.

## Timeout

AFK, 120s unit tests.
