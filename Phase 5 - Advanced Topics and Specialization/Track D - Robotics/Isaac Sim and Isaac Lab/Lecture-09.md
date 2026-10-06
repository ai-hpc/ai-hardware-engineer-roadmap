# Lecture 09: Isaac Lab 3.0 Architecture

## Overview

Lectures 01-08 drove one scene from your own loop. Isaac Lab inverts that: you describe **one** environment as configuration, and the framework clones it thousands of times on one GPU, steps it in lockstep, and hands batched tensors to an RL library. The point is throughput. A policy that needs 100 million environment steps will not train at one robot per process.

Isaac Lab 3.0 (Early Access, `release/3.0.0`, September 2026, requires Isaac Sim 6.1 and Python 3.12) is also the biggest break the framework has had. Physics is now **multi-backend**: PhysX through Isaac Sim, Newton with MuJoCo-Warp, or the experimental kit-less OvPhysX, chosen at launch with `physics=...`. Install is `uv`-driven. Training goes through one `isaaclab train` command. Data properties return `ProxyArray`. Quaternions flip from wxyz to **xyzw**. Most tutorials and forum answers you will find are 2.x, so this lecture teaches the 3.0 shape and gives you a migration table for reading the old one.

By the end you should be able to:

* name the 3.0 packages and say which one owns physics backends, tasks, RL wrappers, and visualizers
* install Isaac Lab 3.0 with `uv`, run a task kit-less (Newton) and with Kit (Isaac Sim PhysX), and explain the difference
* choose between the manager-based and direct workflows, and trace one `step()` through either
* read an `InteractiveSceneCfg` and explain `{ENV_REGEX_NS}`, cloning, and per-env memory
* select backends with `physics=`, `renderer=` and `presets=`, and author a `PresetCfg`
* use the 3.0 data API (`.torch` / `.warp`, `_index` / `_mask` writes, xyzw) and the EventManager randomization modes
* port a 2.x task or command line to 3.0

---

## 1. Why it matters: from one scene to thousands

Recall the rollout-regime argument from [Deep RL Lecture 01](../Deep%20RL%20for%20Robot%20Learning/Lecture-01.md): on-policy algorithms like PPO ([Deep RL Lecture 06](../Deep%20RL%20for%20Robot%20Learning/Lecture-06.md)) are limited by **env-steps/s**, and the only way to get millions per second is to vectorize the simulator itself. Isaac Lab does that with three ideas:

| Idea | What it buys | Where it lives |
|---|---|---|
| **Cloning** | One environment authored once, replicated N times with physics parsing shared | `InteractiveScene`, `isaaclab.cloner` |
| **Batched GPU state** | Observations, actions, rewards and resets as `(N, ...)` tensors that never leave the GPU | Asset `.data` (`ProxyArray`), managers, direct env methods |
| **Configuration over code** | Tasks are `@configclass` trees that you override from the CLI (Hydra) without editing Python | `ManagerBasedRLEnvCfg`, `DirectRLEnvCfg`, `PresetCfg` |

3.0 adds a fourth: **backend independence.** The same task config runs on PhysX or Newton, because asset classes such as `Articulation` are factories that load the active backend's implementation.

---

## 2. Mental model

### 2.1 Packages

The `release/3.0.0` source tree has these packages under `source/`:

| Package | Owns |
|---|---|
| `isaaclab` | Core: `envs`, `managers`, `scene`, `assets`, `sensors`, `sim`, `actuators`, `cloner`, `physics`, `renderers`, `benchmark`, `cli`, `utils` (incl. `ProxyArray`) |
| `isaaclab_physx` | PhysX backend implementations (assets, sensors, `PhysxCfg`, PhysX schema cfgs, `SurfaceGripper`) |
| `isaaclab_newton` | Newton backend (`NewtonCfg`, `MJWarpSolverCfg`, Kamino solver cfg; articulations, rigid objects, deformables) |
| `isaaclab_ov` | Omniverse kit-less integrations: `OvPhysxCfg`, OVRTX renderer |
| `isaaclab_tasks` | Registered gym tasks (`core/` and `contrib/`) and `PresetCfg` utilities |
| `isaaclab_rl` | Wrappers and the unified train/play entrypoints for RSL-RL, RL-Games, skrl, SB3, RLinf, TorchRL |
| `isaaclab_assets` | Robot and object configs (`CARTPOLE_CFG`, ...) |
| `isaaclab_visualizers` | Kit, Newton (GL/RTX), Rerun and Viser viewers |
| `isaaclab_teleop` / `isaaclab_mimic` | Teleoperation (replaces `isaaclab.devices.openxr`) and Mimic data generation (Lecture 11) |
| `isaaclab_contrib`, `isaaclab_experimental`, `isaaclab_tasks_experimental` | Staged contributions and early-access features |
| `isaaclab_ppisp` | Renderer-agnostic HDR→LDR post-processing |

### 2.2 Install: `uv` first, Kit optional

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/isaac-sim/IsaacLab.git --branch release/3.0.0 && cd IsaacLab

uv run isaaclab train --rl_library rsl_rl --task Isaac-Cartpole-Direct physics=newton_mjwarp          # kit-less
uv run --extra ovphysx isaaclab train --rl_library rsl_rl --task Isaac-Cartpole-Direct physics=ovphysx  # kit-less PhysX
uv run --extra isaacsim isaaclab train --rl_library rsl_rl --task Isaac-Cartpole-Direct physics=isaacsim_physx
```

`uv run` creates and syncs the project environment on every call (Python 3.12, PyTorch CUDA 13.0 wheels in the source checkout). `--extra NAME` goes **before** `isaaclab`. Here is what the extras pull in (`pyproject.toml` at `release/3.0.0`):

* `isaacsim` adds `isaacsim[all,extscache]==6.1.0.0` (Kit, RTX, Isaac Sim PhysX)
* `ovphysx`, `ovrtx` and `ov` add the kit-less Omniverse PhysX and RTX runtimes
* `rl-games`, `skrl`, `sb3`, `rlinf`, `torchrl`, `rerun`, `viser`, `mimic`, `teleop` and `video` add those integrations
* `all` covers `ov` plus the common RL and visualizer extras, but **not** Isaac Sim

`rsl_rl` is in the base environment and is the default library for most core tasks. `./isaaclab.sh -i` still exists for manually managed environments; the docs call it legacy. The docs list 16 GB VRAM for full Isaac Sim workflows. Kit-less Newton runs do not load Isaac Sim at all, which is what makes them attractive on an 8 GB card.

Kit is in the process **only** when something needs it: `physics=isaacsim_physx`, `renderer=isaacsim_rtx`, `--viz kit`, or livestreaming. Newton (`newton_mjwarp`, `newton_kamino`) and OvPhysX run kit-less. Do not combine OvPhysX with a Kit runtime in one process.

### 2.3 Two workflows

```text
 Manager-based (ManagerBasedRLEnv + ManagerBasedRLEnvCfg)     Direct (DirectRLEnv / DirectMARLEnv + *Cfg)
 ┌────────────────────────────────────────────────────┐      ┌──────────────────────────────────────────┐
 │ scene         InteractiveSceneCfg                  │      │ scene         InteractiveSceneCfg        │
 │ actions       ActionManager    (terms → targets)   │      │ _pre_physics_step(actions)               │
 │ observations  ObservationManager (groups of terms) │      │ _apply_action()        × decimation      │
 │ rewards       RewardManager    (weighted terms)    │      │ _get_dones()  → terminated, time_out     │
 │ terminations  TerminationManager                   │      │ _get_rewards()                           │
 │ events        EventManager     (randomization)     │      │ _reset_idx(env_ids)                      │
 │ commands      CommandManager   (goals, optional)   │      │ _get_observations() → {"policy": ...}    │
 │ curriculum    CurriculumManager (optional)         │      │ (you write the logic; cfg holds scalars) │
 └────────────────────────────────────────────────────┘      └──────────────────────────────────────────┘
```

| Choose | When |
|---|---|
| **Manager-based** | Prototyping, swapping reward or observation terms, team projects, reusing the `mdp` library (`reset_joints_by_offset`, `randomize_rigid_body_mass`, ...) |
| **Direct** | Performance-critical or hard-to-decompose logic, ports from IsaacGymEnvs, multi-agent (`DirectMARLEnv`), fused reward kernels (TorchScript or Warp) |

Both Cartpoles in `isaaclab_tasks/core/cartpole/` implement the same MDP (same reward scales, resets and termination), one per workflow. Reading them side by side is Lab 9b.

**One `ManagerBasedRLEnv.step()`**, in the order in `manager_based_rl_env.py`:

1. `action_manager.process_action(action)`
2. `decimation` times: `apply_action()`, `scene.write_data_to_sim()`, `sim.step(render=False)`, render only every `sim.render_interval` steps if rendering, then `scene.update(dt)`
3. `termination_manager.compute()`, then `reward_manager.compute(dt=step_dt)`
4. `_reset_idx(...)` for terminated envs, which triggers `event_manager.apply(mode="reset", ...)`
5. `command_manager.compute(...)`, then `event_manager.apply(mode="interval", ...)`
6. `observation_manager.compute()`, then return `(obs, reward, terminated, time_out, extras)`

So `step_dt = decimation × sim.dt`. The Cartpole has `sim.dt = 1/120` and `decimation = 2`, so the policy acts at 60 Hz. Direct envs follow the same skeleton, calling your `_pre_physics_step` once, then `_apply_action` every physics substep.

### 2.4 The scene: one env, cloned

```python
# from isaaclab_tasks/core/cartpole/cartpole_manager_env_cfg.py (release/3.0.0)
@configclass
class CartpoleSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)))
    robot: ArticulationCfg = replace(CARTPOLE_CFG, prim_path="{ENV_REGEX_NS}/Robot")
    distant_light = AssetBaseCfg(prim_path="/World/DistantLight", ...)

scene: CartpoleSceneCfg = CartpoleSceneCfg(num_envs=4096, env_spacing=4.0, clone_in_fabric=True)
```

* **`{ENV_REGEX_NS}`** expands to a regex over the per-env namespaces `/World/envs/env_0`, `env_1`, .... Anything under it is cloned per env. Anything at `/World/...` (ground, lights) exists once and is shared.
* **`replicate_physics=True`** (default) means every clone has identical USD and the physics parser handles them in one optimized pass. Setting it False allows per-env asset differences at a setup-time cost, and it is **not supported on Newton**. **`filter_collisions=True`** stops envs from colliding with their neighbours. `clone_in_fabric` is a **deprecated** legacy flag in 3.0; the Cartpole configs still set it, but the replicator ignores it.
* **Asset configs** are backend-neutral: `AssetBaseCfg` (non-physics), `RigidObjectCfg`, `RigidObjectCollectionCfg`, `ArticulationCfg` (+ `actuators={...: ImplicitActuatorCfg(...)}`), and `DeformableObjectCfg`. Backend-specific physics properties are now **schema fragments** passed as lists. `CARTPOLE_CFG` passes `rigid_props=[UsdPhysicsRigidBodyCfg(...), PhysxRigidBodyCfg(...)]` and `articulation_props=[PhysxArticulationCfg(...), NewtonArticulationCfg(...)]`. The 2.x `RigidBodyPropertiesCfg`-style classes are deprecated aliases scheduled for removal in 3.2.
* **Order matters.** Entities are added in attribute order. The docstring recommends terrain, then articulations and rigid bodies, then sensors, then lights.

3.0 sources use `isaaclab.utils.replace(cfg, prim_path=...)`. You will see the `CARTPOLE_CFG.replace(prim_path=...)` method form in 2.x code.

### 2.5 Backends and presets

A task exposes alternatives as a `PresetCfg`: typed fields are named variants, and `default` is used when you pass nothing. This is the Cartpole's (abridged from `cartpole_common.py`):

```python
from isaaclab.physics import PhysxAutoCfg
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_tasks.utils import PresetCfg

@configclass
class CartpolePhysicsCfg(PresetCfg):
    isaacsim_physx: PhysxCfg = PhysxCfg()
    ovphysx: OvPhysxCfg = OvPhysxCfg()
    physx: PhysxAutoCfg = PhysxAutoCfg(isaacsim_physx=isaacsim_physx, ovphysx=ovphysx)
    newton_mjwarp: NewtonCfg = NewtonCfg(solver_cfg=MJWarpSolverCfg(njmax=5, nconmax=3, ...),
                                         num_substeps=1, use_cuda_graph=True)
    default: NewtonCfg = newton_mjwarp          # Cartpole runs on Newton unless told otherwise

# in the env cfg: self.sim.physics = CartpolePhysicsCfg()
```

| Selector (Hydra token, no dashes) | Examples | Changes |
|---|---|---|
| `physics=` | `isaacsim_physx`, `physx` (auto: Isaac Sim PhysX if Kit is needed, else OvPhysX), `newton_mjwarp`, `newton_kamino` (beta), `ovphysx` (experimental) | The whole `sim.physics` section |
| `renderer=` | `isaacsim_rtx`, `rtx` (auto), `newton_renderer`, `ovrtx` | Camera rendering backend |
| `presets=` | `rgb`, `depth`, ... (task-specific) | Observation modes, camera layouts, bundles |

The order is: each preset's `default` first, then `presets=`, then path-targeted presets (`env.sim.physics=newton_mjwarp`), then scalar overrides (`env.sim.dt=0.002`). `uv run isaaclab train --task <id> --help` lists the names a task supports, and passing an unlisted name fails validation. Defaults are task-specific. Cartpole's is Newton, while tasks with an established PhysX default keep Isaac Sim PhysX, so **always write `physics=` explicitly in anything you benchmark.** Solver settings do not transfer numerically between families. Joint and body ordering can differ between backends, too; the docs have a PhysX↔Newton policy-transfer how-to and an articulation-ordering guide (Lecture 05).

### 2.6 The 3.0 data API

```python
robot = env.unwrapped.scene["robot"]
q  = robot.data.joint_pos.torch        # torch.Tensor (N, num_joints): cached zero-copy view of the Warp buffer
qw = robot.data.joint_pos.warp         # the original wp.array, for Warp kernels
rq = robot.data.root_quat_w.torch      # (N, 4) in (x, y, z, w)
robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)          # index-based write
robot.actuators.target_command.set_effort_index(value=effort, joint_ids=cart_ids)  # 3.0 command API; the direct
# Cartpole still calls the deprecated robot.set_joint_effort_target_index(target=..., joint_ids=...)
```

* **`ProxyArray`.** Every asset and sensor `.data` property returns one. `.torch` is a cached zero-copy tensor view (via `wp.to_torch`), and `.warp` is the underlying array. Arithmetic directly on the `ProxyArray` still works through a deprecation bridge that warns once. **Don't cache `.torch` across resets**: Newton can reallocate buffers on a full reset, and the old tensor then points at stale memory. PhysX refreshes on first access each step; Newton refreshes every step automatically.
* **Writes name their selector.** `write_*_index(...)` takes env or joint indices and `write_*_mask(...)` takes boolean masks. The 2.x single method that took either is gone.
* **Quaternions are (x, y, z, w)** everywhere in Lab 3.0, chosen to match Warp, PhysX and Newton without conversion. Isaac Sim's Core Experimental prims still return **(w, x, y, z)**. If you mix Isaac Sim API calls into a Lab task, or read old 2.x configs, convert. The repo's `scripts/tools/find_quaternions.py` flags likely-wxyz literals in your own code. Setting `WARN_ON_TORCH_QUATF_ACCESS=1` warns on reads of quaternion buffers through `.torch`.

### 2.7 Events: where randomization lives

`EventManager` terms are `EventTermCfg(func=..., mode=..., params=...)`:

| Mode | When it fires | Typical use |
|---|---|---|
| `"prestartup"` | Once, before the simulation starts | USD edits physics must parse at start, e.g. `randomize_rigid_body_scale` (which also needs `replicate_physics=False`) |
| `"startup"` | Once, after the simulation starts | Per-env friction, mass, CoM randomization |
| `"reset"` | On every env reset (the env calls it) | Initial joint and root states |
| `"interval"` | Every `interval_range_s` seconds (the manager handles it) | Random pushes |

From the 3.0 velocity-tracking config (`Isaac-Velocity-*-AnymalD`): `randomize_rigid_body_material` (startup; `static_friction_range`, `dynamic_friction_range`, `restitution_range`, `num_buckets`), `randomize_rigid_body_mass` (startup; `mass_distribution_params=(1/1.25, 1.25)`, `operation="scale"`, `distribution="log_uniform"`), `randomize_rigid_body_com`, `reset_root_state_uniform` and `reset_joints_by_scale` (reset), and `push_by_setting_velocity` (interval, `interval_range_s=(10.0, 15.0)`). Domain randomization theory and evaluation are in [Deep RL Lecture 13](../Deep%20RL%20for%20Robot%20Learning/Lecture-13.md). The mechanism is just these terms.

---

## 3. 2.x → 3.0 migration

| Isaac Lab 2.x | Isaac Lab 3.0 | Source |
|---|---|---|
| conda, Python 3.11, `./isaaclab.sh --install`; Isaac Sim 4.5-5.1 | `uv run ...`, Python 3.12, Isaac Sim 6.1 only (5.1 and older unsupported) | Install docs, migration guide |
| `./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py` / `play.py` | `uv run isaaclab train\|play --rl_library rsl_rl` (also `rl_games`, `skrl`, `sb3`, `rlinf`); library-specific scripts removed | Migration guide |
| `--headless` | Omit `--viz` or pass `--viz none`; `--headless` now only controls Kit's rendering mode and no longer suppresses visualizers | Migration guide |
| `--resume --load_run RUN`, `--use_pretrained_checkpoint` | `--checkpoint <path>\|latest\|best\|pretrained` | Migration guide |
| `Isaac-Cartpole-v0`, `Isaac-Velocity-Rough-Anymal-C-v0`, `Isaac-Lift-Cube-Franka-v0` | `Isaac-Cartpole`, `Isaac-Velocity-Rough-AnymalD`, `Isaac-Lift-Franka`; ANYmal-C only as `IsaacContrib-Velocity-Rough-AnymalC-Direct`. **UNVERIFIED** whether old `-v0` IDs remain as aliases | Task `__init__.py` files, EA notes |
| Quaternions (w, x, y, z), identity `(1, 0, 0, 0)` | (x, y, z, w), identity `(0, 0, 0, 1)` | Migration guide |
| `robot.data.root_pos_w` is a `torch.Tensor` | `robot.data.root_pos_w.torch` (`ProxyArray`) | Migration guide, ProxyArray how-to |
| `write_root_pose_to_sim(pose, env_ids)` | `write_root_pose_to_sim_index(...)` / `..._mask(...)` | Migration guide |
| `self.sim.physics = sim_utils.PhysxCfg(...)` | `isaaclab_physx.physics.PhysxCfg` inside a `PresetCfg`; select with `physics=` | Migration guide |
| `RigidBodyPropertiesCfg`, `JointDrivePropertiesCfg(max_effort=, max_velocity=)` | Schema fragments (`PhysxRigidBodyPropertiesCfg`, `JointDriveBaseCfg(max_force=, max_joint_velocity=)`); old names warn, removed in 3.2 | Migration guide |
| `TiledCamera` / `TiledCameraCfg` | Deprecated alias; `Camera` includes the tiled path | `sensors/camera/tiled_camera.py` |
| `Imu` (full state) | Renamed `Pva`; new lightweight `Imu` (angular velocity + linear acceleration with gravity) | Migration guide |
| `isaaclab.devices.openxr` (`OpenXRDevice`) | `isaaclab_teleop` (`IsaacTeleopDevice`, `IsaacTeleopCfg`) | Migration guide, EA notes |
| `SurfaceGripper` in `isaaclab.assets` | `isaaclab_physx.assets.SurfaceGripper` | Migration guide |
| `env_cfg.viewer.eye = ...`; `gym.wrappers.RecordVideo` | `sim.default_visualizer_cfg = VisualizerCfg(eye=...)`; `env_cfg.video_recorders = [VideoRecorderCfg(...)]` | Migration guide |
| `scripts/benchmarks/benchmark_*.py` | `uv run isaaclab benchmark runtime\|training\|startup\|play` | Migration guide |

Port in this order: install, backend config, data access and quaternions, then training commands and visualization. Validate behavior after the quaternion step, because a wrong quaternion produces a task that runs and quietly learns nothing.

---

## 4. The hardware view: memory per environment, and who owns the GPU

**Where VRAM goes in a training run.** As a working model:

$$
M(N) \approx M_{\text{fixed}} + N \cdot (m_{\text{phys}} + m_{\text{buf}}) + M_{\text{rollout}}(N) + M_{\text{policy}}
$$

* \( M_{\text{fixed}} \) is the process baseline. With Kit, it includes the Kit app and RTX renderer you measured in Lecture 01. Kit-less, it is the CUDA context, Warp and PyTorch. The difference is the first thing Lab 9c measures.
* \( m_{\text{phys}} \) is per-env physics state: bodies, joints, contact pairs, solver buffers. PhysX GPU capacities are **static** (contact-rich or large scenes may need explicit buffer sizing). Newton with `use_cuda_graph=True` pays a one-time graph-capture cost.
* \( m_{\text{buf}} \) is per-env observation, action, reward and reset buffers, plus manager scratch.
* Rollout storage for on-policy RL is arithmetic:

$$
M_{\text{rollout}} \approx N \cdot T \cdot (d_{\text{obs}} + d_{\text{act}} + k) \cdot 4\ \text{B}
$$

  For the Cartpole RSL-RL config (\( T = \) `num_steps_per_env` \( = 16 \), \( d_{\text{obs}} = 4 \), \( d_{\text{act}} = 1 \), and \( k \approx 6 \) for rewards, dones, values, log-probs, returns and advantages), 4,096 envs need about 2.9 MB. For state-based tasks, rollout memory is a rounding error, and the fixed cost plus physics dominate. Cameras change the picture completely, because each env's render product adds \( W \cdot H \cdot \text{channels} \) per step (Lectures 07 and 10).

**Instancing still matters.** With `replicate_physics=True` and instanceable assets (the 6.x importers' default, Lecture 02), N robots share one set of meshes. Per-env cost is then state, not geometry.

**Keep the loop on the GPU.** The loop is physics buffers → `ProxyArray` zero-copy view → observation tensor → policy → action tensor → `write_*_index` → physics, with no host round trip. One stray `.item()`, `.cpu()`, `print(tensor)` or Python `if` on a GPU value forces a device sync every step. The 3.0 direct Cartpole carries comments on exactly this ("device indices avoid per-step host uploads", "no .item(): avoids a sync on every reset"). Rendering is skipped unless a visualizer, camera or video asks for it: with `--viz none` and no cameras, `step()` never renders.

**Two processes cannot share 8 GB politely.** Isaac Lab training and an Isaac Sim GUI (or a second training run) each hold their own fixed cost and caching allocators. Run one GPU job at a time on the 4060.

> **8 GB budget.** State-only tasks are the sweet spot on a 4060. Kit-less Newton (`physics=newton_mjwarp`, `--viz none`) avoids the Kit and RTX baseline entirely, so classic-control and many locomotion tasks should fit thousands of envs. Find the knee with Lab 9c rather than trusting any published number. With Kit (`--extra isaacsim`, `physics=isaacsim_physx`), subtract your Lecture 01 fixed cost first. Turn down `--num_envs` before anything else, then drop visualizers (`--viz none`), and avoid camera presets until Lecture 10's budget. Move to a cloud L40S / RTX PRO 6000 when you need Kit plus RTX cameras across many envs, or when Kit and an evaluation policy must share one card. The 4060 is below the official 16 GB minimum for full Isaac Sim workflows.

---

## 5. Build it

### Lab 9a — Install and run `Isaac-Cartpole`, kit-less and with Kit

```bash
cd IsaacLab
uv run python scripts/environments/list_envs.py | grep -i cartpole        # registered IDs (any -v0 left?)
uv run isaaclab train --task Isaac-Cartpole --help                       # look for the physics= choices

# kit-less Newton, no viewer
uv run isaaclab train --rl_library rsl_rl --task Isaac-Cartpole --num_envs 4096 --viz none physics=newton_mjwarp
# the same task through Isaac Sim PhysX (Kit in the process)
uv run --extra isaacsim isaaclab train --rl_library rsl_rl --task Isaac-Cartpole --num_envs 4096 --viz none physics=isaacsim_physx
# watch the result
uv run isaaclab play --rl_library rsl_rl --task Isaac-Cartpole --checkpoint latest --num_envs 16 --viz newton_gl
```

Log `nvidia-smi` throughout. Note time-to-first-iteration, iteration time, and peak VRAM for both runs. Checkpoints land under `logs/`. Confirm in the startup log that the Kit run reports Isaac Sim starting and the Newton run does not. (`--viz newton` also works but is a deprecated alias for `newton_gl`.)

### Lab 9b — Read a task end to end

Open `source/isaaclab_tasks/isaaclab_tasks/core/cartpole/` and annotate, in `ANNOTATED_CARTPOLE.md`:

1. **Manager-based** (`cartpole_manager_env_cfg.py`). For each config class (`CartpoleSceneCfg`, `ActionsCfg`, `ObservationsCfg.PolicyCfg`, `EventCfg`, `RewardsCfg`, `TerminationsCfg`, `CartpoleEnvCfg.__post_init__`), write which manager consumes it, at which step of §2.3 it runs, and the tensor shape it produces for N envs. Note what is absent (`commands`, `curriculum`). Then do the same for `EventsCfg`, `CommandsCfg` and `CurriculumCfg` in `core/velocity/velocity_env_cfg.py`.
2. **Direct** (`cartpole_direct_env.py`). Map each method to the manager it replaces. Find where `episode_length_buf` drives `time_out` and where the reward is fused into one TorchScript function.
3. **Backends** (`cartpole_common.py`). List the physics presets and say which one runs when you pass nothing.

Then check your annotations against a live env. This probe follows the launcher pattern of 3.0's own `zero_agent` / `random_agent` entrypoint (`isaaclab_rl/entrypoints/simple_agents.py`):

```python
# isaac_bench/lab_probe.py
# uv run python isaac_bench/lab_probe.py --task Isaac-Cartpole --num_envs 1024 --viz none physics=newton_mjwarp
import argparse, sys, time
import gymnasium as gym, torch
from isaaclab.app import add_launcher_args, launch_simulation
import isaaclab_tasks  # noqa: F401  (registers the gym IDs)
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Cartpole")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--steps", type=int, default=500)
add_launcher_args(parser)
args, hydra_args = setup_preset_cli(parser, None)
sys.argv = [sys.argv[0]] + hydra_args                  # physics= / renderer= / presets= go to Hydra

env_cfg, _ = resolve_task_config(args.task, "")
env_cfg.scene.num_envs = args.num_envs
args.device = env_cfg.sim.device

with launch_simulation(env_cfg, args):
    env = gym.make(args.task, cfg=env_cfg)
    env.reset()
    u = env.unwrapped
    robot = u.scene["robot"]                           # "cartpole" in Isaac-Cartpole-Direct
    jp = robot.data.joint_pos
    print(f"{type(jp).__name__} torch={tuple(jp.torch.shape)} on {jp.torch.device}")
    print("root quat (x, y, z, w):", [round(v, 3) for v in robot.data.root_quat_w.torch[0].tolist()])
    print(f"physics_dt={u.physics_dt:.5f} step_dt={u.step_dt:.5f} action_space={u.action_space.shape}")

    act = torch.zeros(u.action_space.shape, device=u.device)
    with torch.inference_mode():
        for _ in range(50):                            # warm-up (CUDA graphs, allocator)
            env.step(act)
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(args.steps):
            env.step(act)
        torch.cuda.synchronize(); dt = time.perf_counter() - t
    print(f"env-steps/s={u.num_envs * args.steps / dt:,.0f}  ms/step={dt / args.steps * 1e3:.3f}  "
          f"torch peak={torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
    env.close()
```

The startup log prints every manager's term table (`[INFO] Event Manager: ...`, `Observation Manager`, `Reward Manager`, ...). Diff it against your annotation. `torch peak` counts only PyTorch's allocator, so compare it with the `nvidia-smi` total to see how much belongs to physics and the runtime.

### Lab 9c — Kit vs kit-less: VRAM and env-steps/s

`isaaclab benchmark runtime` steps a task with random actions, no policy, and writes a result bundle. Its flags (`--num_envs`, `--num_steps`, `--warmup_steps`, `--benchmark_formatter`, `--output_path`, `--measure_sync_step`) come from `isaaclab/benchmark/entrypoints/runtime.py`:

```bash
# isaac_bench/lab_sweep.sh
set -e; mkdir -p results
for physics in newton_mjwarp isaacsim_physx; do
  extra=""; [ "$physics" = isaacsim_physx ] && extra="--extra isaacsim"
  for n in 256 1024 4096 16384; do
    tag=cartpole_${physics}_${n}
    nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -lms 200 > results/$tag.vram.csv &
    smi=$!
    uv run $extra isaaclab benchmark runtime --task Isaac-Cartpole-Direct \
        --num_envs $n --num_steps 1000 --warmup_steps 50 --visualizer none \
        --benchmark_formatter json,summary --output_path results/$tag physics=$physics || echo "FAILED $tag"
    kill $smi
  done
done
```

Add `physics=ovphysx` (with `--extra ovphysx`) if it installs on your machine. For each run, record env-steps/s from the summary and peak VRAM minus idle desktop. Plot both against N on log axes. Then rerun the largest N that fits with `--measure_sync_step` to split simulation time from outside-simulation time.

What to expect qualitatively: at small N, both backends are dominated by fixed per-step overhead and throughput grows almost linearly with N. At large N, the GPU saturates and the curve flattens. The Kit run starts from a higher VRAM floor. Where the two cross in speed is task- and GPU-specific; that is why you measure.

---

## 6. Use it in the real stack

* **Training** ([Lecture 10](Lecture-10.md)) is the same commands with longer runs, sweeps over `--num_envs` and `physics=`, and your own direct task.
* **Imitation and evaluation** (Lecture 11) use `isaaclab_teleop`, `isaaclab_mimic`, and Isaac Lab-Arena, which composes a scene, an embodiment and a task into a `ManagerBasedRLEnvCfg`. The manager vocabulary from §2.3 is what Arena generates.
* **The Deep RL course** uses Isaac Lab as its rung-2 simulator. The `ProxyArray` → policy → `write_*_index` loop is where its PPO implementation ([Deep RL Lecture 06](../Deep%20RL%20for%20Robot%20Learning/Lecture-06.md)) attaches.
* **Extension projects** depend on the released `isaaclab` wheel (`uv pip install --index https://pypi.nvidia.com isaaclab==3.0.0rc1` plus extras) instead of a checkout.

---

## 7. Measure it

| Metric | How | Why it matters |
|---|---|---|
| **Env-steps/s vs N** | Lab 9c summary; `lab_probe.py` | The throughput curve that sets training wall-clock |
| **Peak VRAM vs N** | `nvidia-smi` log per run | Slope = VRAM per env; intercept = fixed cost |
| **Kit vs kit-less fixed cost** | Intercept difference between backends | How many envs Kit costs you on 8 GB |
| **Torch vs total VRAM** | `torch.cuda.max_memory_allocated()` vs `nvidia-smi` | Separates policy/rollout memory from physics and runtime |
| **Sim vs outside-sim time** | `--measure_sync_step` | Whether physics or your Python/managers bound the step |
| **Time to first step** | `isaaclab benchmark startup` | Iteration speed when you change configs often |

---

## 8. Ship it

Commit to `isaac_bench/`:

* `lab_probe.py`, `lab_sweep.sh`, and the `results/` directory (benchmark JSON, VRAM CSVs)
* `lab_scaling.png` — env-steps/s and peak VRAM vs N for each backend
* `ANNOTATED_CARTPOLE.md` — the Lab 9b annotation, with the manager-to-method map
* `MIGRATION_NOTES.md` — one 2.x snippet of your choice (from a tutorial or forum post) ported to 3.0, line by line, citing the table in §3
* An update to `BASELINE.md` — kit-less vs Kit fixed cost, VRAM per env for Cartpole, and the N where throughput stops scaling on your GPU

---

## Exit criteria

You can move on when you can:

* install Isaac Lab 3.0 with `uv` and run a task kit-less and with Kit, explaining what each loads
* trace one `step()` through the managers (or the direct methods) and compute `step_dt` from a config
* explain `{ENV_REGEX_NS}`, `replicate_physics`, and what is shared vs per-env in a cloned scene
* select and author physics presets, and say which preset runs when none is given
* use `.torch` / `.warp`, `_index` writes and xyzw quaternions correctly, and port a 2.x command line
* report VRAM per env and env-steps/s for Cartpole on both backends on your GPU

---

## Self-check

1. A 2.x tutorial runs `./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Lift-Cube-Franka-v0 --headless --resume --load_run 2025-10-01`. Write the 3.0 command, and name each change.
2. A ported task trains but the robot spawns lying on its side, and the reward never improves. The config has `rot=(1.0, 0.0, 0.0, 0.0)` in `init_state`. Explain the bug, and how you would find every other instance in your code.
3. Two teammates benchmark `Isaac-Cartpole` and get very different env-steps/s on identical GPUs. One passed no `physics=` token and the other passed `physics=isaacsim_physx`. Explain the difference, and what every benchmark command in your repo should include.
4. Your direct task's `_get_rewards` caches `self.joint_pos = robot.data.joint_pos.torch` in `__init__`, and on Newton the rewards become nonsense after the first full reset, while on PhysX they look fine. Why?
5. You want per-env friction randomization drawn once per training run, plus a random push every 10-15 s. Which EventManager modes and which `mdp` functions do you use, and which component triggers each mode?
6. On an 8 GB card, Kit-mode Cartpole runs out of memory at an N where Newton is still comfortable. Using the \( M(N) \) model, explain which term differs, and how Lab 9c's two curves show it.

---

## References

* Isaac Lab 3.0 installation (uv, extras, kit-less, legacy `isaaclab.sh`) — [docs](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/setup/installation/index.html)
* Quickstart (commands, RL libraries, backends, visualizers, benchmarks) — [docs](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/setup/quickstart.html)
* Migrating to 3.0 — [docs](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/migration/migrating_to_isaaclab_3-0.html)
* Backends and presets — [docs](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/concepts/backends_and_presets.html); physics backends — [docs](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/concepts/physics_backends.html)
* Task workflows (manager-based vs direct) — [docs](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/concepts/task_workflows.html)
* Working with `ProxyArray` — [docs](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/how-to/proxy_array.html)
* Isaac Lab v3.0.0-EA release notes — [GitHub](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0-EA)
* Source at `release/3.0.0` — [cartpole tasks](https://github.com/isaac-sim/IsaacLab/tree/release/3.0.0/source/isaaclab_tasks/isaaclab_tasks/core/cartpole), [`manager_based_rl_env.py`](https://github.com/isaac-sim/IsaacLab/blob/release/3.0.0/source/isaaclab/isaaclab/envs/manager_based_rl_env.py), [`interactive_scene_cfg.py`](https://github.com/isaac-sim/IsaacLab/blob/release/3.0.0/source/isaaclab/isaaclab/scene/interactive_scene_cfg.py), [`event_manager.py`](https://github.com/isaac-sim/IsaacLab/blob/release/3.0.0/source/isaaclab/isaaclab/managers/event_manager.py), [benchmark `runtime.py`](https://github.com/isaac-sim/IsaacLab/blob/release/3.0.0/source/isaaclab/isaaclab/benchmark/entrypoints/runtime.py), [`pyproject.toml`](https://github.com/isaac-sim/IsaacLab/blob/release/3.0.0/pyproject.toml)

---

## Next in this special course

* Next: [Lecture 10 — Training Policies in Isaac Lab](Lecture-10.md)
* Previous: [Lecture 08 — ROS 2 and OmniGraph](Lecture-08.md)
* Back: [Isaac Sim and Isaac Lab — Overview](README.md)
