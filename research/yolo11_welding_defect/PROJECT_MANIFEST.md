# YOLO11 Welding Defect Method Project Manifest

Last updated: 2026-05-18

## Scope

This directory is the control center for the method paper:

> YOLO11 improvement for small and elongated welding defect detection.

It also links to the separate survey paper on YOLO-based welding defect detection, but the survey should keep its own writing plan once the Zotero literature export is ready.

## Fixed Decisions

1. `Auto-research` is the single controller for data, experiments, results, claims, and paper artifacts.
2. `AI-Researcher` is an idea generator and external advisor. It is not the final source of experimental facts or formal citations.
3. Zotero, exported via Better BibTeX, is the formal literature source.
4. AI-found papers go to a candidate pool first. They must be scored before entering Zotero and formal writing.
5. The method paper targets small-object and elongated welding defect robustness, not generic YOLO11 tweaking.
6. The method has two core contributions:
   - Welding defect morphology-aware localization loss.
   - Lightweight small-defect feature enhancement module.
7. Safe augmentation, class balancing, and hard-example strategies are auxiliary contributions.
8. Transferability validation means training baseline and improved variants per dataset, then comparing relative gains. It is not zero-shot testing across incompatible label spaces.

## Research Problem

Welding defects are often small, elongated, low contrast, and class imbalanced. Standard YOLO11 can under-optimize localization quality and recall for cracks, air holes, slag inclusion, unfused regions, and other narrow or sparse targets.

## Working Method Name

Temporary name:

```text
WD-YOLO11
```

Temporary loss name:

```text
WDLoss: Welding Defect Morphology-Aware Localization Loss
```

## Method Hypotheses

H1: A morphology-aware localization loss can improve `mAP50-95` and recall for small or elongated welding defects.

H2: A lightweight high-resolution feature enhancement module can preserve fine defect cues without causing large compute overhead.

H3: Combining the loss and feature module should outperform either component alone on the main welding dataset and retain positive gains on at least two transfer datasets.

## Primary Metrics

Primary metric:

```text
mAP50-95
```

Second primary metric:

```text
Recall / AP improvement for small or elongated defect classes
```

Auxiliary metrics:

```text
mAP50
Precision
Recall
per-class AP
per-class Recall
parameters
FLOPs
FPS or latency
```

## Human Gates

Gate 1: Idea selection

AI-Researcher and Codex may generate many candidates. Only selected ideas enter the experiment matrix.

Gate 2: Experiment matrix approval

Before launching long runs, confirm variables, compute budget, datasets, and ablation coverage.

Gate 3: Result-to-claim approval

Only claims supported by raw metrics, per-class results, and ablations may enter the paper.

Gate 4: Citation audit

Every formal citation must exist in the Zotero export and support the context where it is cited.

## Automation Flow

```text
Zotero / Better BibTeX export
  -> literature quality scoring
  -> curated topic packs
  -> AI-Researcher Socratic mode
  -> AI-Researcher reference-based mode
  -> candidate idea merge
  -> novelty and feasibility review
  -> experiment matrix
  -> YOLO training queue
  -> result analysis
  -> result-to-claim
  -> paper planning
  -> paper writing
  -> citation audit
```

## AI-Researcher Role

Use `AI-Researcher` for:

1. Socratic mode: generate plausible improvement directions from a high-level description.
2. Reference-based mode: generate candidate ideas from curated paper packs.

Do not use `AI-Researcher` as:

1. The final source of citations.
2. The final source of experimental results.
3. The authority on whether a claim is true.

## Immediate Next Tasks

1. Run a 1-epoch smoke test on `data/welding-defect-detection-yolo/data.yaml`.
2. Build object-size and class-imbalance reports for transfer datasets.
3. Prepare Zotero export sync: `.bib` plus `.json`.
4. Score the existing literature into A/B/C/D tiers.
5. Build the first executable experiment queue for baseline, loss variants, feature variants, and combined variants.
