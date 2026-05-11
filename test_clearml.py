from clearml import Task

task = Task.init(
    project_name="my_project",
    task_name="test_from_mac_to_windows",
    output_uri=True,
)

task.execute_remotely(queue_name="win-gpu", exit_process=True)

print("hello from windows worker")
task.upload_artifact("result", {"status": "ok", "worker": "windows"})
