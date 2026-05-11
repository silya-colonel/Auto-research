# YOLO on Windows with ClearML

This runbook adapts ARIS to a practical YOLO defect detection setup:

```text
Mac editing/Codex -> GitHub private repo -> Windows native training -> ClearML UI
```

Linux, Vast.ai, Modal, OpenClaw, Cursor, Antigravity, Zotero/Obsidian, patent, grant, poster, and slides workflows remain available as optional layers. The Windows path is the temporary primary route when the Linux server is down.

For the complete Chinese step-by-step SOP, see [YOLO 缺陷检测全流程使用方法](YOLO_FULL_WORKFLOW_CN.md).

## Recommended Layout

```text
project/
  AGENTS.md
  requirements.txt
  data.yaml                  # may point to Windows absolute dataset paths
  train_yolo_clearml.py       # optional wrapper after smoke test
  experiments/
    YOLO_EXPERIMENT_MANIFEST.md
  results/
    DATA_HEALTH_REPORT.md
    RESULTS_SUMMARY.md
    RUN_TABLE.csv
    NEXT_EXPERIMENTS.md
```

Do not commit:

```text
datasets/
runs/
weights/
*.pt
*.pth
*.onnx
*.engine
.clearml/
```

Reusable ARIS assets:

```text
templates/train_yolo_clearml.py
templates/YOLO_EXPERIMENT_MANIFEST_TEMPLATE.md
templates/YOLO_GITIGNORE_TEMPLATE.txt
templates/YOLO_REQUIREMENTS_TEMPLATE.txt
tools/yolo_data_health.py
```

## Stage 1: Direct Windows Run

Use this stage first. It verifies GPU, Python, dataset paths, YOLO, and ClearML API credentials without depending on ClearML Agent or GitHub private-repo cloning.

### Required Preparation

On Windows, prepare:

```text
C:\work\yolo\
  AGENTS.md
  requirements.txt
  train_yolo_clearml.py
  experiments\
  results\
  tools\

C:\datasets\defect\
  data.yaml
  images\
  labels\
```

The minimum `requirements.txt` is:

```text
ultralytics
clearml
clearml-agent
```

Install and verify the environment:

```powershell
nvidia-smi
conda create -n yolo python=3.10 -y
conda activate yolo
pip install -r C:\work\yolo\requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If `torch.cuda.is_available()` prints `False`, stop here and fix NVIDIA driver / CUDA-enabled PyTorch before using ClearML.

### ClearML API Credentials

Create API credentials in ClearML Web UI:

```text
ClearML Web UI -> Settings/Profile -> Workspace -> API Credentials -> Create new credentials
```

Then initialize the Windows machine:

```powershell
clearml-init
```

Paste the ClearML Web UI values:

```text
api_server
web_server
files_server
access_key
secret_key
```

Without this step, `Task.init(...)`, ClearML logging, and ClearML Agent jobs will fail.

Verify the credentials before training:

```powershell
conda activate yolo
python -c "from clearml import Task; t=Task.init(project_name='yolo', task_name='clearml_connection_check'); print('task_id=', t.id); t.close()"
```

Success means `clearml_connection_check` appears in the ClearML Web UI.

If ClearML Server is also on this Windows machine, keep an eye on disk and RAM usage. Training gets priority; move ClearML Server later if it competes with GPU workflows.

### Smoke Test

Run directly once before automating:

```powershell
cd C:\work\yolo
conda activate yolo
python train_yolo_clearml.py --task-name smoke_yolo11n_1epoch --data-yaml C:\datasets\defect\data.yaml --model yolo11n.pt --imgsz 640 --epochs 1 --batch -1
```

Success means:

- CUDA is visible
- Ultralytics can read the dataset
- the output directory is created
- ClearML logs appear if ClearML integration is configured

Do not continue to Stage 2 until this works.

## Stage 2: ClearML Agent Queue

Use this stage only after Stage 1 succeeds. This stage adds automatic task dispatch, GitHub private-repo cloning, and queued baseline runs.

### Required Preparation

The GitHub private repo must contain:

```text
AGENTS.md
requirements.txt
train_yolo_clearml.py
experiments/YOLO_EXPERIMENT_MANIFEST.md
tools/yolo_data_health.py
```

The GitHub private repo must not contain:

```text
datasets/
runs/
weights/
*.pt
*.pth
*.onnx
*.engine
clearml.conf
```

Windows or the ClearML Agent user must be able to access the private repo.

Recommended SSH setup on Windows:

```powershell
ssh-keygen -t ed25519 -C "windows-yolo-clearml-agent"
type $env:USERPROFILE\.ssh\id_ed25519.pub
```

Add the public key to GitHub:

```text
GitHub -> Settings -> SSH and GPG keys
```

or for one repository only:

```text
Repo -> Settings -> Deploy keys
```

Verify:

```powershell
ssh -T git@github.com
git clone git@github.com:<your-user>/yolo-defect.git C:\work\yolo-agent-test
```

GitHub Desktop login or HTTPS + Personal Access Token also works. Do not write tokens into project files.

### Start the Agent

After the direct smoke test passes, use ClearML Agent for repeatable queued experiments:

```powershell
conda activate yolo
clearml-agent daemon --queue yolo-windows --foreground
```

Keep this PowerShell window open.

If the Agent reports virtual-environment creation or package-manager errors, install `clearml-agent` in `base` or system Python and start the Agent from there. Keep Stage 1 training in the `yolo` environment.

### Submit a Queued Task

```powershell
clearml-task --project yolo --name baseline_yolo11n_640 --repo git@github.com:<your-user>/yolo-defect.git --branch main --queue yolo-windows --script train_yolo_clearml.py --requirements requirements.txt --args "--task-name baseline_yolo11n_640 --data-yaml C:\datasets\defect\data.yaml --model yolo11n.pt --imgsz 640 --epochs 100 --batch -1"
```

ARIS should submit code/config through GitHub and let the Windows agent run from the queue. GitHub should not carry datasets or large weights.

Stage 2 success means:

- the task appears in the `yolo-windows` queue
- the Windows Agent picks it up
- the Agent clones the repo
- dependencies import successfully
- ClearML Web UI shows live console logs and metrics

Common failures:

- clone fails: fix Windows SSH key, Deploy key, GitHub Desktop login, or PAT
- import fails: confirm `requirements.txt` is in Git and contains `ultralytics` and `clearml`
- environment creation fails: start Agent from `base` or system Python instead of the active training environment
- data path fails: confirm `C:\datasets\defect\data.yaml` exists on Windows
- queue idle: confirm `--queue yolo-windows` matches the daemon queue name
- GPU OOM: use `--batch -1`, lower `--imgsz`, or set a smaller batch

## Automation Strategy

Use `/yolo-pipeline` as the top-level command.

The pipeline should:

- inspect `data.yaml` and label health
- run one smoke test
- run `yolo11n/s` baselines
- compare ClearML metrics
- recommend 3-8 next experiments instead of a blind grid
- only modify architecture/loss after baseline evidence exists
- write `NARRATIVE_REPORT.md` for paper/patent/grant handoff

## What Fully Automatic Means

ARIS can propose models, modules, losses, and experiment order, but it should not silently make expensive or high-risk decisions. Keep human checkpoints for:

- dataset taxonomy changes
- architecture/loss code changes
- cloud GPU spending
- publication or patent claims

This keeps the automation useful without turning it into a very confident GPU-burning gremlin.
