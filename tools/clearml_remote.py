from __future__ import annotations

import json
import shlex
import sys
from collections.abc import Callable
from typing import Any


TaskFactory = Callable[..., Any]


def maybe_execute_remotely(
    *,
    enabled: bool,
    project_name: str,
    task_name: str,
    queue_name: str = "gpu-any",
    clone: bool = False,
    exit_process: bool = True,
    task_factory: TaskFactory | None = None,
) -> bool:
    """Submit the current process to a ClearML queue when requested.

    Returns True after requesting remote execution. With ClearML's normal
    `exit_process=True` behavior, the local process exits inside
    `execute_remotely`; on a worker, ClearML continues past the call.
    """

    if not enabled:
        return False

    if task_factory is None:
        from clearml import Task

        task_factory = Task.init

    task = task_factory(project_name=project_name, task_name=task_name)

    # Store filtered CLI args so the remote worker can reconstruct the command.
    # Strip --clearml-remote / --clearml-queue / --clearml-project to avoid
    # infinite re-submission loops.
    _skip_values = {"--clearml-queue", "--clearml-project"}
    script_args = sys.argv[1:]
    filtered: list[str] = []
    skip_next = False
    for a in script_args:
        if skip_next:
            skip_next = False
            continue
        if a in ("--clearml-remote", "--enable-clearml", "--disable-clearml"):
            continue
        if a in _skip_values:
            skip_next = True
            continue
        if a.startswith(("--clearml-queue=", "--clearml-project=")):
            continue
        filtered.append(a)

    task.set_parameter("Args/script_args", json.dumps(filtered))

    task.execute_remotely(queue_name=queue_name, clone=clone, exit_process=exit_process)
    return True


def get_remote_script_args() -> list[str]:
    """Retrieve stored script args when running on a ClearML worker.

    Call this after the ``execute_remotely`` no-op path to reconstruct the
    command-line arguments that were originally passed on the submitter side.
    """
    from clearml import Task

    task = Task.current_task()
    if task is None:
        return []
    raw = task.get_parameter("Args/script_args")
    if raw is None:
        return []
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return shlex.split(raw)
