# Isaac, ROS, and execution compatibility record

**Reviewed 2026-10-18.** This record separates a documented upstream combination from something actually run in this repository. It does not certify every lecture script on every backend.

**Current release:** Isaac Sim **6.2.0 GA**, released October 14, 2026. The course's existing 6.1 examples are a reproducible earlier baseline; they should not be read as the latest release. [Official 6.2 release](https://github.com/isaac-sim/IsaacSim/releases/tag/v6.2.0)

| Component or claim | Verified boundary | Primary source |
|---|---|---|
| Isaac Sim 6.2.0 GA | Kit 110.4.0, Newton 1.6.1, Warp 1.17.0; review the upgrade checks below | [6.2 release notes](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/overview/release_notes.html) |
| Isaac Sim 6.1.0 GA | Release notes identify Kit SDK 110.3.0 and experimental Newton 1.5.0 integration | [6.1 release notes](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/overview/release_notes.html) |
| Full simulator hardware | Published minimum includes RTX 4080 and 16 GB VRAM; more memory may be needed for training and complex sensors | [6.1 requirements](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/requirements.html) |
| Isaac Lab 3.0.0-EA | The release is early access. Its stack lists Python 3.12, PyTorch 2.11, Warp 1.16, Newton 1.5.2; full Sim workflows use Sim 6.1, while kit-less workflows can omit it | [EA release](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0-EA) |
| ROS bridge on Ubuntu | Jazzy on Ubuntu 24.04 is recommended. Humble on Ubuntu 22.04 is supported; generated interfaces loaded inside Sim must match its Python 3.12 runtime | [ROS installation](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/install_ros.html) |
| ROS-only CPU lab here | Jazzy with Fast DDS, Python 3.12.3; actual transport, TF, and four input-contract outcomes measured | [Lab](../Robotics%20Foundations/ROS2-Lab.md), [raw run](../Robotics%20Foundations/evidence/ros2-jazzy-local-run.json) |

## Current release and upgrade checks

6.2 changes sensor reads to owned snapshots, invalidates Newton tensor views after model rebuilds, removes Cortex, and changes some actuator ordering/limit behavior. It also fixes Newton substep/state-buffer problems. These changes can alter timing and controller inputs even when imports still succeed. [6.2 changes](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/overview/release_notes.html)

For an upgrade exercise, repeat the same known-pose, action-order, sensor-age, reset, and substep checks on both versions. Recreate invalid views, verify each new sensor sample, and compare trajectories before speed. Record the asset catalog and dependency lock too. Use the [cross-product compatibility table](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/compatibility.html) when adding Isaac ROS or other products; their version numbers do not need to match Sim's.

Isaac Sim's bundled Newton and Isaac Lab's Newton dependency are different entries in that table. Do not infer that they share the same patch version, solver settings, or numerical behavior. Record the dependency lock and chosen backend for each run.

The 8 GB RTX 4060 examples in this course are reduced-budget experiments below the published full-simulator minimum. A kit-less workflow has a different runtime, rendering, and memory footprint; its successful run would not establish that the full Kit/RTX simulator works on the same hardware. Choose based on the task's required backend and sensors, not just the product name.

## Pin the source, including the documentation

The tag `v3.0.0-EA` resolves to commit:

```text
ae37b028ea415c91ea2bc32609efcd759ed2b974
```

Its [`pyproject.toml`](https://github.com/isaac-sim/IsaacLab/blob/ae37b028ea415c91ea2bc32609efcd759ed2b974/pyproject.toml) pins `isaacsim==6.1.0.0` through the optional extra and `rsl-rl-lib==5.4.1`. By contrast, `release/3.0.0` at reviewed commit [`0c114a24a3b9e854d299d3527ada507139fc4e40`](https://github.com/isaac-sim/IsaacLab/blob/0c114a24a3b9e854d299d3527ada507139fc4e40/pyproject.toml) pins Sim `6.2.0.0` and RSL-RL `5.5.1`. The standalone [RSL-RL lecture](../Deep%20RL%20for%20Robot%20Learning/Lecture-06b.md) intentionally reads 5.5.1; that is not the EA tag's dependency set. Keep these environments separate unless you deliberately validate an upgrade.

**Upstream discrepancy:** the Sim compatibility page names `v3.0.0-GA`, but no published Lab tag/release by that name was found in this review. The [Lab repository's dependency table](https://github.com/isaac-sim/IsaacLab#isaac-sim-compatibility) instead maps the current release branch to Sim 6.2 and the EA tag to Sim 6.1. Treat the branch as a source snapshot; do not invent a GA release pin from a forward-looking compatibility row.

Several rendered 3.0 documentation links in the upstream release currently return 404. The course links these topics to the **same tag's source documents**, rather than substituting 2.x documentation:

- [Installation source](https://github.com/isaac-sim/IsaacLab/blob/ae37b028ea415c91ea2bc32609efcd759ed2b974/docs/source/setup/installation/index.rst)
- [Quickstart source](https://github.com/isaac-sim/IsaacLab/blob/ae37b028ea415c91ea2bc32609efcd759ed2b974/docs/source/setup/quickstart.rst)
- [2.x → 3.0 migration source](https://github.com/isaac-sim/IsaacLab/blob/ae37b028ea415c91ea2bc32609efcd759ed2b974/docs/source/migration/migrating_to_isaaclab_3-0.rst)
- [End-to-end tutorial source](https://github.com/isaac-sim/IsaacLab/blob/ae37b028ea415c91ea2bc32609efcd759ed2b974/docs/source/setup/tutorial.rst)

These files contain Sphinx references and includes, so some navigation works differently on GitHub. Read neighbouring files at the same commit when following an include. If adopting a later `release/3.0.0` commit for a bug fix, record that exact commit, dependency lock, task configuration, and source changes; the moving branch and EA tag are not interchangeable pins.

## Avoid Python ABI mixing

An external Humble node on Ubuntu 22.04 can use its normal Python 3.10 environment while communicating with the simulator over middleware. That does not make its generated Python modules importable inside the simulator's Python 3.12 process. Use a separate terminal and the documented 3.12 workspace when loading custom interfaces into Sim. Do not copy the system Humble environment into the Sim process merely because the ROS distribution name matches. [NVIDIA ROS/Python setup](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/install_ros.html)

## Executed CPU evidence

The [simulation CPU lab](Simulation-CPU-Lab.md) now includes actual Newton 1.5.2/Warp 1.16.0 free-fall and contact runs, OpenUSD 26.8 layer/payload checks, six ROS message outcomes, a separate-process known-weight policy adapter, and extracted lesson methods executed with CPU PyTorch. Standard-library contracts pass 27 tests. Each result states what was simulated or faked; none is a full Isaac Sim or Isaac Lab training run. [Exact validation record](labs/validation-2026-10-18.json)

## Evidence still required on your machine

Before a full simulator milestone, save the compatibility-check result, actual GPU/driver/runtime versions, a zero-action scene run, a reset check, and a clock/frame test. Then measure sensor and policy timing under the intended workload. The repository's CPU checks and measured ROS-only lab establish useful prerequisites; **Isaac Sim, Isaac Lab training, RTX rendering, and robot actuation were not executed by this compatibility review**.

Return to the [course overview](README.md) or [version and hardware contract](Lab-Guide.md#version-and-hardware-contract).
