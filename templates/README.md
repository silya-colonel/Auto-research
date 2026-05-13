# Templates

Ready-to-use templates for each ARIS workflow. Copy, fill in your content, and run the corresponding skill.

### Workflow Input Templates

| Template | For Workflow | What to do |
|----------|-------------|------------|
| [RESEARCH_BRIEF_TEMPLATE.md](RESEARCH_BRIEF_TEMPLATE.md) | Workflow 1 | Detailed research direction as document input |
| [RESEARCH_CONTRACT_TEMPLATE.md](RESEARCH_CONTRACT_TEMPLATE.md) | Workflow 1 | Define problem boundaries, non-goals, timeline before starting |
| [EXPERIMENT_PLAN_TEMPLATE.md](EXPERIMENT_PLAN_TEMPLATE.md) | Workflow 1.5 | Claim-driven experiment roadmap with run order and budgets |
| [NARRATIVE_REPORT_TEMPLATE.md](NARRATIVE_REPORT_TEMPLATE.md) | Workflow 3 | Research narrative with claims, experiments, results |
| [PAPER_PLAN_TEMPLATE.md](PAPER_PLAN_TEMPLATE.md) | Workflow 3 | Pre-made outline to skip planning phase |
| [CLAUDE_MD_TEMPLATE.md](CLAUDE_MD_TEMPLATE.md) | All Workflows | Project dashboard with Pipeline Status — create once per project |
| [MANIFEST_TEMPLATE.md](MANIFEST_TEMPLATE.md) | All Workflows | Output tracking manifest — auto-maintained by skills |

### YOLO Templates (`/yolo-pipeline`)

| Template | What to do |
|----------|------------|
| [YOLO_EXPERIMENT_MANIFEST_TEMPLATE.md](YOLO_EXPERIMENT_MANIFEST_TEMPLATE.md) | Baseline/tuning/architecture experiment manifest for YOLO runs |
| [YOLO_GITIGNORE_TEMPLATE.txt](YOLO_GITIGNORE_TEMPLATE.txt) | Safety defaults to keep datasets and weights out of Git |
| [YOLO_REQUIREMENTS_TEMPLATE.txt](YOLO_REQUIREMENTS_TEMPLATE.txt) | Minimal Python dependency list for YOLO training |
| [train_yolo.py](train_yolo.py) | Ultralytics YOLO training wrapper with local metrics logging |

### Chinese Templates (中文模板)

| Template | For Workflow | What to do |
|----------|-------------|------------|
| [RESEARCH_BRIEF_TEMPLATE_CN.md](RESEARCH_BRIEF_TEMPLATE_CN.md) | Workflow 1 | 研究简报中文模板 |
| [IDEA_CANDIDATES_TEMPLATE_CN.md](IDEA_CANDIDATES_TEMPLATE_CN.md) | Workflow 1 | Idea 候选池中文模板 |
| [EXPERIMENT_PLAN_TEMPLATE_CN.md](EXPERIMENT_PLAN_TEMPLATE_CN.md) | Workflow 1.5 | 实验计划中文模板 |

### Patent Templates (`/patent-pipeline`)

| Template | For Workflow | What to do |
|----------|-------------|------------|
| [INVENTION_BRIEF_TEMPLATE.md](INVENTION_BRIEF_TEMPLATE.md) | Patent Pipeline | Invention disclosure with technical problem, solution, advantages, figures |
| [PATENT_CLAIMS_TEMPLATE.md](PATENT_CLAIMS_TEMPLATE.md) | `/claims-drafting` | Claims hierarchy worksheet with examples for CN/US/EP |
| [PATENT_SPECIFICATION_TEMPLATE.md](PATENT_SPECIFICATION_TEMPLATE.md) | `/specification-writing` | Skeleton specification with all required sections |

### Compact Mode Templates (`— compact: true`)

| Template | Written by | Purpose |
|----------|-----------|---------|
| [IDEA_CANDIDATES_TEMPLATE.md](IDEA_CANDIDATES_TEMPLATE.md) | `/idea-discovery` | Top 3-5 surviving ideas (lean, not full 12-idea report) |
| [EXPERIMENT_LOG_TEMPLATE.md](EXPERIMENT_LOG_TEMPLATE.md) | `/experiment-bridge` | Structured experiment record (results + reproduction commands) |
| [FINDINGS_TEMPLATE.md](FINDINGS_TEMPLATE.md) | `/auto-review-loop` | One-line-per-finding discovery log (anomalies, decisions) |

## Usage

### Research Pipeline

```bash
cp templates/EXPERIMENT_PLAN_TEMPLATE.md refine-logs/EXPERIMENT_PLAN.md
# Edit with your content, then:
/experiment-bridge
```

### YOLO Pipeline

```bash
cp templates/YOLO_EXPERIMENT_MANIFEST_TEMPLATE.md experiments/YOLO_EXPERIMENT_MANIFEST.md
cp templates/train_yolo.py train_yolo.py
cp templates/YOLO_REQUIREMENTS_TEMPLATE.txt requirements.txt
# Then invoke:
/yolo-pipeline "YOLO defect detection"
```

### Patent Pipeline

```bash
cp templates/INVENTION_BRIEF_TEMPLATE.md patent/INVENTION_BRIEF.md
# Edit with your invention details, then:
/patent-pipeline "patent/INVENTION_BRIEF.md -- CN"
```
