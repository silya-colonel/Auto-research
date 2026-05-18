# Literature Workflow

## Source Policy

Formal literature source:

```text
Zotero -> Better BibTeX export -> GitHub private repository -> Auto-research
```

Expected synced files:

```text
zotero_library.bib
zotero_library.json
export_manifest.json
```

Formal citations in papers must come from `zotero_library.bib`.

Structured analysis should use `zotero_library.json` when available.

## Formal vs Candidate Literature

Formal library:

Papers already accepted into Zotero and exported.

Candidate library:

Papers discovered by AI-Researcher, Codex, arXiv, Semantic Scholar, Exa, or web search. Candidate papers are not formal citations until reviewed and added to Zotero.

## Generated Artifacts

Codex should generate and update:

```text
literature_index.csv
taxonomy.md
review_matrix.md
method_related_work.md
yolo11_improvement_candidates.md
citation_audit_report.md
```

## Topic Packs for AI-Researcher

Do not feed all papers at once. Build curated packs:

```text
01_welding_defect_yolo_core
02_yolo11_and_recent_yolo_improvements
03_small_object_detection
04_elongated_crack_defect_detection
05_localization_loss_functions
06_attention_neck_feature_fusion
07_transfer_validation_industrial_defects
```

Each pack should include high-quality papers first, with low-quality or weakly related papers excluded or marked as candidates.
