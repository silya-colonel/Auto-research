# ClearML GPU Queue Workflow

## Goal

Use one ClearML queue for all available GPU workers:

```text
gpu-any
```

Both the Windows machine GPU and the Linux server GPUs should run ClearML agents that listen to this queue. Training scripts submit tasks to the queue and then exit locally:

```python
task.execute_remotely(queue_name="gpu-any", clone=False, exit_process=True)
```

## Training Submission

Use `train_yolo.py train` with `--clearml-remote`:

```powershell
D:\app\conda\envs\sy\python.exe train_yolo.py train `
  --task-name smoke_welding_yolo11n_640 `
  --data-yaml data\welding-defect-detection-yolo\data.yaml `
  --model yolo11n.pt `
  --imgsz 640 `
  --epochs 1 `
  --clearml-remote `
  --clearml-queue gpu-any `
  --clearml-project yolo-welding-defect
```

Default queue:

```text
gpu-any
```

Default ClearML project:

```text
yolo-welding-defect
```

## Worker Requirement

The submission command does not itself use every GPU. It enqueues the task. To use local and remote GPUs, start ClearML agents on each GPU host and point them at the same queue.

Conceptually:

```text
Windows GPU agent -> gpu-any
Linux GPU 0 agent -> gpu-any
Linux GPU 1 agent -> gpu-any
Linux GPU 2 agent -> gpu-any
Linux GPU 3 agent -> gpu-any
```

ClearML then assigns queued training tasks to available workers.

## Local Direct Training

If you want to train immediately on the current process without queue submission, omit `--clearml-remote`.

```powershell
D:\app\conda\envs\sy\python.exe train_yolo.py train `
  --task-name local_smoke_welding_yolo11n_640 `
  --data-yaml data\welding-defect-detection-yolo\data.yaml `
  --model yolo11n.pt `
  --imgsz 640 `
  --epochs 1
```

## Notes

- Root-level scripts such as `clearml_cuda_test.py` and `clearml_smoke.py` are only connectivity tests.
- The reusable training integration lives in `train_yolo.py` and `tools/clearml_remote.py`.
- Keep `clone=False` so the current ClearML task is submitted instead of creating an additional cloned task.
- Keep `exit_process=True` so the Windows submission process exits after enqueueing rather than training locally and remotely at the same time.
