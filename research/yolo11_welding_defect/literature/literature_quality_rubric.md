# Literature Quality Rubric

## Tiers

A tier:

Core paper. High relevance and credible venue, strong method or dataset connection, useful for the survey framework or method-paper motivation.

B tier:

Useful citation. Relevant but narrower, older, incremental, or mainly supports background, definitions, or comparison tables.

C tier:

Candidate or weakly related. Useful for idea exploration but not central enough for formal argumentation.

D tier:

Reject or quarantine. Duplicated, low quality, off topic, metadata unreliable, or citation context too risky.

## Scoring Fields

Use 0 to 5 unless noted.

```text
relevance_score
quality_score
dataset_relevance
method_relevance
loss_related
small_object_related
elongated_defect_related
welding_related
yolo11_related
citation_risk
```

Suggested decision:

```text
A: relevance >= 4 and quality >= 4 and citation_risk <= 2
B: relevance >= 3 and quality >= 3
C: relevance >= 2 or useful for idea generation
D: duplicate, unreliable, or off topic
```

## Required Checks

1. Does the paper actually study detection, not only classification or segmentation?
2. Does it use welding defects, industrial surface defects, small objects, elongated defects, or YOLO improvements?
3. Are datasets and metrics clearly reported?
4. Is the method described with enough detail to support a claim?
5. Is there a DOI, arXiv ID, publisher page, or stable venue record?
6. Does the cited claim match what the paper actually shows?
