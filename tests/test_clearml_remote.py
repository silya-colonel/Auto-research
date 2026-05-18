from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class ClearMLRemoteTests(unittest.TestCase):
    def test_disabled_remote_execution_does_not_create_task(self) -> None:
        from tools.clearml_remote import maybe_execute_remotely

        calls: list[str] = []

        def task_factory(*args: object, **kwargs: object) -> object:
            calls.append("created")
            raise AssertionError("task should not be created when remote execution is disabled")

        submitted = maybe_execute_remotely(
            enabled=False,
            project_name="project",
            task_name="task",
            queue_name="gpu-any",
            task_factory=task_factory,
        )

        self.assertFalse(submitted)
        self.assertEqual(calls, [])

    def test_enabled_remote_execution_submits_to_gpu_any_queue(self) -> None:
        from tools.clearml_remote import maybe_execute_remotely

        calls: dict[str, object] = {}

        class FakeTask:
            def execute_remotely(self, **kwargs: object) -> None:
                calls["execute_remotely"] = kwargs

        def task_factory(**kwargs: object) -> FakeTask:
            calls["task_init"] = kwargs
            return FakeTask()

        submitted = maybe_execute_remotely(
            enabled=True,
            project_name="yolo-welding",
            task_name="baseline",
            queue_name="gpu-any",
            task_factory=task_factory,
        )

        self.assertTrue(submitted)
        self.assertEqual(calls["task_init"], {"project_name": "yolo-welding", "task_name": "baseline"})
        self.assertEqual(
            calls["execute_remotely"],
            {"queue_name": "gpu-any", "clone": False, "exit_process": True},
        )


if __name__ == "__main__":
    unittest.main()
