from __future__ import annotations

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
    task.execute_remotely(queue_name=queue_name, clone=clone, exit_process=exit_process)
    return True
