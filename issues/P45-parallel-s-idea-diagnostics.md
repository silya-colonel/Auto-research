# P45 — Parallel S-Idea Diagnostics

## What to build

Create a test-first diagnostic harness for three new steel-defect research ideas:
CLR-YOLO, OSD-YOLO, and FASD-YOLO.

This phase is diagnosis only. It should decide which idea deserves GPU training
without reviving failed historical routes such as MCA/DSA/QDH/HPR/MRS/RST/HNC/TPS.

## Acceptance criteria

- [ ] CLR-YOLO, OSD-YOLO, and FASD-YOLO each have an independently runnable diagnostic or audit CLI.
- [ ] Each diagnostic writes `summary.json`, detailed records, and `report.md`.
- [ ] A ranking report recommends one next training route or rejects all three.
- [ ] Experiment matrix and SSH dry-run support exist for all three diagnostics.
- [ ] No full multi-seed training is started before the ranking report is reviewed.

## Blocked by

P45.1 through P45.7.

## Timeout

HITL parent issue.
