# ICRA 2027 Task1–6 evaluation resources

Minimal source-preserving extraction from `zhutengjie/HumanoidVLA_MJ`, local baseline `afb35c108c5633704378ba67fd9e5bcdab5233a7` plus current local assets/viewer, 2026-09-06. This repository is an extraction, not the complete upstream project. Original source ownership and applicable asset terms remain unchanged.

Includes `scripts/deploy/task3_isaac_hssd_viewer.py`, the nine MJCF task definitions used by SIMPLE Task1–6 and their mesh/texture dependencies, G1 robot assets, and supplementary assets referenced by the configured recording scene snapshots. `ASSET_MANIFEST.json` records exact hashes and source paths. Asset PNG/STL/OBJ files are scene resources, not recordings or model weights.

Task1–4 first-frame CSV initialization records and their XML snapshots are included under `mujoco_recordings`; see its README and manifest. The two Task1–4 HSSD backgrounds and their dependencies are published in `ql3c20/SIMPLE_eval_env:icra2027` under `data/scenes/hssd`. Full CSV trajectories, videos, policy weights and Python environments are not included. Supply those separately. Keep this checkout at `HUMANOID_VLA_MJ_ROOT`; install compatible Isaac Sim separately.

The exact SIMPLE G1 robot resource is stored under `mujoco/model/simple_g1_sonic`; on a fresh deployment, create a symlink at `$SIMPLE_ROOT/data/robots/g1_sonic` to that directory. Existing data directories must not be overwritten.

The viewer consumes recording `model_snapshot` XML and MuJoCo UDP state; it does not own physics or success evaluation. Rendering and full episode tests still require external data and weights.

SIMPLE also loads `data/robots/g1/curobo` during robot initialization. The required YAML, URDF and mesh resources are under `mujoco/model/simple_g1`; link this folder to `$SIMPLE_ROOT/data/robots/g1`. No AMO adapter `.pt` weights are included.
