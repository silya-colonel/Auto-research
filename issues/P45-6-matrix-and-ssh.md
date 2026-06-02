# P45.6 — Matrix and SSH Submission Support

## Parent

P45 — Parallel S-Idea Diagnostics

## What to build

Add experiment-matrix rendering and SSH dry-run support for CLR-YOLO,
OSD-YOLO, FASD-YOLO, and the final idea ranking report.

## Acceptance criteria

- [ ] `tools/experiment_matrix.py` renders commands for `clr_diag`, `osd_diag`, `fasd_audit`, and `idea_rank` run kinds.
- [ ] `tools/ssh_submit.py` dry-runs remote commands for `clr_diag`, `osd_diag`, and `fasd_audit`.
- [ ] `tests/test_experiment_matrix.py` covers the new run kinds.
- [ ] `tests/test_ssh_submit.py` covers the new remote command builders.
- [ ] `research/yolo11_welding_defect/experiments/parallel_s_idea_diagnostics.yaml` renders three diagnostic commands.

## Blocked by

P45.2, P45.3, and P45.4.

## Timeout

AFK, 120s unit tests; dry-run only.
