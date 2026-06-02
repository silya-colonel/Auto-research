# P45.5 — Parallel Idea Ranking Report

## Parent

P45 — Parallel S-Idea Diagnostics

## What to build

Create a ranking report that reads CLR-YOLO, OSD-YOLO, and FASD-YOLO diagnostic
summaries, prioritizes ideas that pass their gates, penalizes risk, and names
the recommended next training route.

## Acceptance criteria

- [ ] `tools/rank_parallel_ideas.py` exists.
- [ ] `tests/test_rank_parallel_ideas.py` verifies passing ideas outrank failed ideas and score/risk affect ordering.
- [ ] The CLI writes a Markdown report with a ranked table.
- [ ] The report contains a line beginning with `Recommended next training route:`.
- [ ] The focused tests pass with `pytest tests/test_rank_parallel_ideas.py -v`.

## Blocked by

P45.2, P45.3, and P45.4.

## Timeout

AFK, 120s unit tests.
