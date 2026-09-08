# Task1–4 evaluation initial states

Each recording contains `data.csv` with the original header and **exactly one data record** (the first recorded frame), plus unchanged `model_snapshot` XML files. No later frames, videos, or weights are included. These files initialize evaluation; they cannot replay demonstrations or produce future-window training labels.

Counts: Task1 69, Task2 167, Task3 100, Task4 150; 486 recordings total. `MANIFEST.json` lists exported files and SHA-256 hashes. Original source recordings were not changed.

SIMPLE `FullstateRecordingMixin` reads the first CSV row and requires each recording's `model_snapshot/mujoco/model/g1/scene_43dof.xml`. Keep the snapshots alongside the CSV. XML mesh/texture references also require this repository's `mujoco/model` assets and the existing viewer path-resolution setup.

Set the launcher's `TASK_RECORDINGS_DIR` and corresponding `TASK1_RECORDINGS_DIR` / `TASK2_RECORDINGS_DIR` / `TASK3_RECORDINGS_DIR` / `TASK4_RECORDINGS_DIR` to the selected directory under this checkout's `mujoco_recordings`. Launch wrappers with hardcoded paths must be updated locally. Alternatively, create individual task-directory symlinks under the launcher's expected recordings root, only when the destinations do not already exist.

HSSD backgrounds are external and are not included: Task1/4 use `SIMPLE/data/scenes/hssd/102344280/102344280.usd`; Task2/3 use `SIMPLE/data/scenes/hssd/102344250/102344250_local.usd`. Transfer the referenced textures/assets as well as the USD; a standalone USD file is not necessarily sufficient. Policy weights and Isaac Sim remain external dependencies.
