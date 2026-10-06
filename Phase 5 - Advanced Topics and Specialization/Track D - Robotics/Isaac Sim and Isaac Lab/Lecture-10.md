# Lecture 10: Training Policies in Isaac Lab

## Overview

[Lecture 09](Lecture-09.md) took Isaac Lab 3.0 apart: packages, managers, scenes, backends. This lecture runs it as a training system. One `isaaclab train` command starts a loop that steps thousands of cloned environments, runs a policy network on every step, and updates the network with PPO. All three share one GPU. On an 8 GB RTX 4060 the questions that matter are concrete: how many environments fit, where each millisecond of an iteration goes, and how long it takes in wall-clock time to reach a good policy.

You will train the stock tasks (Cartpole, Franka lift, ANYmal-D locomotion, a camera Cartpole), read their logs like an engineer rather than a spectator, then write and train your own **direct-workflow** task: a Franka reaching a random target.

By the end you should be able to:

* drive `isaaclab train` / `play` / `benchmark` from the command line, including backend selectors and Hydra overrides
* read an RSL-RL run: losses, action std, throughput, checkpoints, videos
* split an iteration's time into environment stepping, policy inference, and PPO update, using numbers you measured
* predict the VRAM of a state task and a camera task before launching them, then check the prediction
* write, register, and train a `DirectRLEnv` task, and explain the order in which its methods run
* produce wall-clock-to-reward and VRAM tables for your GPU

---

## 1. Why it matters: training is a GPU pipeline with three stages

The PPO loop for massively parallel simulation was popularized by "Learning to Walk in Minutes" (Rudin et al., 2021). It has three stages per iteration, and all three run on your GPU:

| Stage | What runs | Scales with | Lever |
|---|---|---|---|
| **Collect** | `num_steps_per_env` × (policy forward + `env.step()`) for all envs | `num_envs`, physics cost per env, sensors | `--num_envs`, physics preset, decimation, cameras |
| **Store** | Rollout buffer: observations, actions, rewards, values for every step and env | `num_envs × num_steps_per_env × obs size` | Observation size (pixels!), rollout length |
| **Update** | PPO epochs × minibatches over the buffer | Batch size, network size | `num_learning_epochs`, `num_mini_batches`, network width |

A faster simulator does not guarantee faster training. If the update stage dominates, doubling env throughput saves little. If the rollout buffer for a camera task does not fit, nothing runs at all. The theory of PPO is covered in [Deep RL Lecture 06](../Deep%20RL%20for%20Robot%20Learning/Lecture-06.md), and the rollout-regime view (env-steps/s as the currency) in [Deep RL Lecture 01](../Deep%20RL%20for%20Robot%20Learning/Lecture-01.md). This lecture is the measured, hardware-side version of both.

---

## 2. Mental model: one `isaaclab train` call, end to end

### 2.1 The command surface (Isaac Lab 3.0)

Every option below was read from the `release/3.0.0` source (`isaaclab_rl/entrypoints/common.py`, `backends/cli_args_rsl_rl.py`, the Kit launcher) or the 3.0 docs:

```bash
uv run isaaclab train --rl_library rsl_rl --task Isaac-Cartpole \
    --num_envs 4096 --seed 42 --max_iterations 150 \
    --run_name seed42 --viz none \
    physics=newton_mjwarp \
    agent.algorithm.learning_rate=0.0005
```

| Piece | Kind | What it does |
|---|---|---|
| `--rl_library rsl_rl\|skrl\|rl_games\|sb3\|rlinf\|torchrl` | dispatcher flag | Picks the backend module. If omitted, the task's registered `default_agent` is used (RSL-RL for most core tasks) |
| `--task`, `--num_envs`, `--seed`, `--max_iterations` | common flags | `--seed -1` draws a random seed |
| `--video`, `--video_length`, `--video_interval` | common flags | Records clips (needs `--extra video`); turns camera rendering on |
| `--checkpoint latest\|best\|pretrained\|<path>` | RSL-RL flag | Resume (train) or select (play) |
| `--experiment_name`, `--run_name`, `--logger tensorboard\|wandb\|neptune` | RSL-RL flags | Log folder naming and backend |
| `--viz none\|newton_gl\|newton_rtx\|kit\|viser\|rerun` | launcher flag | Visualizer; omit or `none` for headless. The docs' `--viz newton` is a deprecated alias of `newton_gl` |
| `--device cuda:0`, `--deterministic` | launcher flags | Device; reproducible Torch/rendering/physics settings |
| `physics=`, `renderer=`, `presets=` | Hydra tokens (no dashes) | Backend and task-mode presets; list them with `--task X --help` |
| `env.<path>=…`, `agent.<path>=…` | Hydra overrides | Any config field, e.g. `env.sim.dt=0.002` |

Optional libraries are `uv` extras placed *before* `isaaclab`: `uv run --extra skrl isaaclab train --rl_library skrl ...`. Isaac Sim itself is an extra too (`--extra isaacsim`), needed for `physics=isaacsim_physx`, `renderer=isaacsim_rtx` and `--viz kit`.

> **You will see this in older code.** Isaac Lab 2.x ran `./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Cartpole-v0 --headless --enable_cameras`. In 3.0 the per-library scripts are replaced by `isaaclab train|play`, `--headless` by `--viz none` (or simply omitting `--viz`), task IDs lose `-v0`, and the `release/3.0.0` Kit launcher defines no `--enable_cameras` flag: rendering is enabled automatically when the scene contains Kit camera sensors, or when you pass `--video`.

### 2.2 What one iteration does

```text
 iteration k
 ├── collect: repeat num_steps_per_env times
 │     obs ──► actor MLP/CNN (GPU) ──► actions ──► env.step(actions)
 │                                                   ├── _pre_physics_step(actions)      once
 │                                                   ├── repeat `decimation` times:
 │                                                   │     _apply_action(); write_data_to_sim()
 │                                                   │     sim.step(); scene.update()
 │                                                   ├── _get_dones()                     ← before rewards!
 │                                                   ├── _get_rewards()
 │                                                   ├── _reset_idx(done env ids)
 │                                                   └── _get_observations()
 ├── store: write (obs, action, reward, value, log-prob, done) into the rollout buffer
 └── update: PPO for num_learning_epochs × num_mini_batches; log; maybe save model_<k>.pt
```

The env-step order comes from `DirectRLEnv.step()` in `release/3.0.0`. Two consequences. First, one env step equals `decimation` physics steps, so the policy runs at \( 1 / (\text{decimation} \cdot dt) \) Hz. Second, `_get_dones()` runs **before** `_get_rewards()`, so per-step quantities that both need (distances, poses) belong in `_get_dones()`. The stock Cartpole refreshes its joint-state caches there.

The batch per iteration is \( N_{\text{env}} \times T \) transitions, where \( T \) is `num_steps_per_env`. For the stock Cartpole agent (`num_steps_per_env = 16`, default 4,096 envs) that is 65,536 transitions per update.

### 2.3 Libraries

RSL-RL is the default and is installed with Isaac Lab. It covers PPO, teacher-student distillation, symmetry augmentation, RND and CNN policies, and it is what this course uses. For a line-by-line reading of its PPO (adaptive learning rate, clipped value loss, time-out bootstrapping, observation groups), see [Deep RL Lecture 06b](../Deep%20RL%20for%20Robot%20Learning/Lecture-06b.md). The others are extras: skrl (`--extra skrl`; JAX, MAPPO/IPPO, AMP), RL-Games (`--extra rl-games`), Stable-Baselines3 (`--extra sb3`), TorchRL (`--extra torchrl`; in the 3.0 source and docs, not the EA notes), and RLinf for VLA post-training ([Lecture 11](Lecture-11.md)). The 3.0 docs advise choosing a library by the features you need, not by a single throughput number.

### 2.4 Tasks used in this lecture

| Task ID (3.0) | Workflow | Default physics in source | What it stresses |
|---|---|---|---|
| `Isaac-Cartpole` / `Isaac-Cartpole-Direct` | manager / direct | `newton_mjwarp` | Almost nothing: measures framework overhead |
| `Isaac-Lift-Franka` | manager | `newton_mjwarp` | Contacts, an articulated arm, a graspable object |
| `Isaac-Velocity-Flat-AnymalD` / `-Rough-AnymalD` | manager | (check `--help`) | Legged contacts; Rough adds terrain and a height scan |
| `Isaac-Cartpole-Camera-Direct` | direct | `newton_mjwarp` | Rendering and pixel observations (96×96, 2-frame stack, 512 envs by default) |

In the `release/3.0.0` configs of Cartpole, Lift and the cabinet task, the `default` physics preset is **Newton with MJWarp**, and PhysX is one `physics=isaacsim_physx` (or `ovphysx`) away. That matters on an 8 GB card: the default path is kit-less and does not load Isaac Sim or RTX unless a camera, `--viz kit` or a PhysX preset asks for them.

---

## 3. Reading a training run

**Where things land.** RSL-RL writes to `logs/rsl_rl/<experiment_name>/<YYYY-MM-DD_HH-MM-SS>[_<run_name>]/`: TensorBoard events, `model_<iteration>.pt` checkpoints every `save_interval` iterations, the dumped env and agent configs, a run manifest, and `videos/` when `--video` is on. Watch with `uv run python -m tensorboard.main --logdir logs`.

**The scalars RSL-RL 5.5 logs** (from `rsl_rl/utils/logger.py` at `v5.5.1`, the version pinned by the 3.0 `pyproject.toml`):

| Key | Read it as | Healthy shape |
|---|---|---|
| `Train/mean_reward`, `Train/mean_episode_length` | Return and length over recently finished episodes | Rising reward; length task-dependent (long for balancing, short for reach-and-done) |
| `Train/mean_reward/time` | The same reward, but against **wall-clock seconds** | Your wall-clock-to-reward curve, logged for free |
| `Loss/value`, `Loss/surrogate`, `Loss/entropy`, `Loss/learning_rate` | Critic fit, PPO objective, exploration, adaptive LR | Value loss falls then stabilizes; LR moves under `schedule="adaptive"` |
| `Policy/mean_std` | Mean action standard deviation | Shrinks slowly; collapsing early means exploration died |
| `Perf/total_fps`, `Perf/collection_time`, `Perf/learning_time` | Transitions per second; seconds collecting vs updating per iteration | Stable; the ratio tells you which stage to optimize |
| `Episode/*`, `Metrics/*` | Task extras (e.g. Cartpole's `Metrics/success_rate`) | Task-specific; trust success over reward |

The adaptive learning rate and the KL leash behind `desired_kl` are explained in [Deep RL Lecture 06](../Deep%20RL%20for%20Robot%20Learning/Lecture-06.md). A reward curve that climbs while `Metrics/success_rate` stays flat is reward hacking. Fix the reward, not the hyperparameters.

**Playing a checkpoint.** `play` loads a checkpoint and applies the env config's `play_mode()` (the docs say it caps the scene at 50 envs and disables observation noise):

```bash
uv run isaaclab play --rl_library rsl_rl --task Isaac-Cartpole --checkpoint latest --viz newton_gl
uv run --extra video isaaclab play --rl_library rsl_rl --task Isaac-Cartpole \
    --checkpoint best --video --video_length 200
```

Use the same `physics=`/`renderer=`/`presets=` for play as for training. An observation preset changes tensor shapes, and the checkpoint will refuse to load.

---

## 4. Writing your own direct task

The direct workflow puts the whole MDP in one class: you implement the step functions yourself instead of composing manager terms. The task below is "move the Franka hand to a random point". It is small enough to train on a 4060 and has every piece a real task has.

### 4.1 Generate a project

The 3.0 template generator creates an installable `uv` project that advertises its tasks through an `isaaclab.tasks` entry point, so the shared `isaaclab` CLI finds them:

```bash
cd ~/IsaacLab   # your release/3.0.0 checkout
uv run isaaclab --new --non_interactive \
    --project_path ~/work --name isaac_reach --author "Your Name" \
    --workflow direct:single-agent --rl_library rsl_rl --rl_algorithm ppo
cd ~/work/isaac_reach && uv sync          # kit-less Newton by default; no Isaac Sim
uv run isaaclab list_envs --show_presets  # the generated Cartpole example should appear
```

The generated package has `src/isaac_reach/tasks/<family>/...`. Add a sibling family `src/isaac_reach/tasks/reach/` with the four files below. The generated task importer discovers registrations under `tasks/`, skipping packages named `mdp`.

### 4.2 Config: scene, spaces, physics, task constants

```python
# src/isaac_reach/tasks/reach/reach_env_cfg.py
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass, replace
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG
from isaaclab_tasks.utils import preset
# Known-good Franka physics/decimation presets, borrowed from the core cabinet task:
from isaaclab_tasks.core.cabinet.cabinet_env_cfg import (
    LIGHT_CFG, PLANE_CFG, CabinetDecimationCfg, CabinetSimCfg)


@configclass
class ReachSceneCfg(InteractiveSceneCfg):
    plane: AssetBaseCfg = PLANE_CFG
    light: AssetBaseCfg = LIGHT_CFG
    robot: ArticulationCfg = replace(FRANKA_PANDA_CFG, prim_path="{ENV_REGEX_NS}/Robot")
    # Newton needs the asset's MuJoCo physics variant; the PhysX backends need the PhysX one.
    robot.spawn.variants["Physics"] = preset(
        default="mujoco", isaacsim_physx="physx", physx="physx", ovphysx="physx")


@configclass
class FrankaReachEnvCfg(DirectRLEnvCfg):
    episode_length_s = 4.0
    decimation: int = CabinetDecimationCfg()  # policy at 60 Hz on either backend
    sim: CabinetSimCfg = CabinetSimCfg()      # Newton: dt 1/600 s; PhysX: dt 1/60 s
    action_space = 7                          # arm joint-position offsets
    observation_space = 20                    # q(7) + qd(7) + ee_pos(3) + (target - ee)(3)
    state_space = 0
    scene: ReachSceneCfg = ReachSceneCfg(num_envs=1024, env_spacing=2.0)

    arm_joint_names = "panda_joint[1-7]"
    ee_body_name = "panda_hand"
    action_scale = 0.5                        # rad per unit action
    target_low = (0.35, -0.25, 0.20)          # [m], env frame
    target_high = (0.65, 0.25, 0.55)
    success_radius = 0.05                     # [m]
    rew_distance = -1.0
    rew_near = 1.0
    rew_action_rate = -0.01
```

Reusing `CabinetSimCfg`/`CabinetDecimationCfg` is deliberate. They are maintained presets that already step a Franka stably on Newton and PhysX (Newton at a finer timestep, with decimation chosen so the policy rate matches). Write your own `PresetCfg` once the task works. The concepts page shows the pattern.

### 4.3 The environment, method by method

```python
# src/isaac_reach/tasks/reach/reach_env.py
from collections.abc import Sequence
import torch
from isaaclab.envs import DirectRLEnv
from .reach_env_cfg import FrankaReachEnvCfg


class FrankaReachEnv(DirectRLEnv):
    cfg: FrankaReachEnvCfg

    def __init__(self, cfg: FrankaReachEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)        # builds + clones the scene
        self.robot = self.scene["robot"]
        ids, _ = self.robot.find_joints(self.cfg.arm_joint_names)
        self.arm_ids = torch.tensor(ids, dtype=torch.long, device=self.device)
        self.ee_idx = self.robot.find_bodies(self.cfg.ee_body_name)[0][0]
        self.q0 = self.robot.data.default_joint_pos.torch[:, self.arm_ids]   # ProxyArray → torch
        self.lo, self.hi = (torch.tensor(v, device=self.device) for v in (cfg.target_low, cfg.target_high))
        z = lambda *shape, **kw: torch.zeros(*shape, device=self.device, **kw)
        self.targets, self.ee_pos, self.q_targets = z(self.num_envs, 3), z(self.num_envs, 3), z(self.num_envs, len(ids))
        self.dist, self.success = z(self.num_envs), z(self.num_envs, dtype=torch.bool)
        self.prev_actions = torch.zeros_like(self.actions)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:      # once per env step
        self.prev_actions[:] = self.actions
        self.actions[:] = actions.clamp(-1.0, 1.0)
        self.q_targets[:] = self.q0 + self.cfg.action_scale * self.actions

    def _apply_action(self) -> None:                                  # `decimation` times
        self.robot.actuators.target_command.set_position_index(
            value=self.q_targets, joint_ids=self.arm_ids)

    def _update_ee(self) -> None:
        self.ee_pos[:] = self.robot.data.body_pos_w.torch[:, self.ee_idx] - self.scene.env_origins
        self.dist[:] = torch.linalg.norm(self.targets - self.ee_pos, dim=-1)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:        # runs BEFORE rewards
        self._update_ee()
        self.success |= self.dist < self.cfg.success_radius
        time_out = self.episode_length_buf >= self.max_episode_length
        return torch.zeros_like(time_out), time_out                  # no early termination

    def _get_rewards(self) -> torch.Tensor:
        r = (self.cfg.rew_distance * self.dist
             + self.cfg.rew_near * (1.0 - torch.tanh(self.dist / 0.1))
             + self.cfg.rew_action_rate * torch.sum((self.actions - self.prev_actions) ** 2, dim=-1))
        return r * self.step_dt

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.extras.setdefault("log", {})["Metrics/success_rate"] = self.success[env_ids].float().mean()
        super()._reset_idx(env_ids)                                    # scene reset + reset events
        self.targets[env_ids] = self.lo + (self.hi - self.lo) * torch.rand(len(env_ids), 3, device=self.device)
        q = self.robot.data.default_joint_pos.torch[env_ids].clone()
        self.robot.write_joint_position_to_sim_index(position=q, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q), env_ids=env_ids)
        self.actions[env_ids], self.prev_actions[env_ids], self.success[env_ids] = 0.0, 0.0, False

    def _get_observations(self) -> dict:
        self._update_ee()   # recompute; just-reset envs may show pre-reset poses until the next physics step
        q = self.robot.data.joint_pos.torch[:, self.arm_ids] - self.q0
        qd = self.robot.data.joint_vel.torch[:, self.arm_ids]
        return {"policy": torch.cat((q, qd, self.ee_pos, self.targets - self.ee_pos), dim=-1)}
```

Every Isaac Lab call above appears in the `release/3.0.0` sources of the stock Cartpole and cabinet direct tasks: `self.scene["…"]`, `find_joints`/`find_bodies`, `.data.*.torch`, `actuators.target_command.set_position_index`, `write_joint_*_to_sim_index`, `episode_length_buf`, `step_dt`, `extras["log"]`. Two 3.0 details matter here. `.data.*` returns a `ProxyArray`, so take `.torch` (or `.warp`). The older `set_joint_position_target_index` still exists but is marked deprecated in favor of the actuator command API. If you extend the task to target *orientations*, Isaac Lab 3.0 quaternions are **xyzw**, while the Isaac Sim experimental prim API is wxyz.

### 4.4 Agent config and registration

```python
# src/isaac_reach/tasks/reach/agents/rsl_rl_ppo_cfg.py   (+ an empty agents/__init__.py)
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

@configclass
class ReachPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 500
    save_interval = 50
    experiment_name = "franka_reach_course"
    actor = RslRlMLPModelCfg(hidden_dims=[128, 64], activation="elu", obs_normalization=True,
                             distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0))
    critic = RslRlMLPModelCfg(hidden_dims=[128, 64], activation="elu", obs_normalization=True)
    algorithm = RslRlPpoAlgorithmCfg(value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2,
                                     entropy_coef=0.005, num_learning_epochs=5, num_mini_batches=4,
                                     learning_rate=1.0e-3, schedule="adaptive", gamma=0.99, lam=0.95,
                                     desired_kl=0.01, max_grad_norm=1.0)
```

```python
# src/isaac_reach/tasks/reach/__init__.py
import gymnasium as gym
from . import agents

gym.register(
    id="Course-Reach-Franka-Direct",
    entry_point=f"{__name__}.reach_env:FrankaReachEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.reach_env_cfg:FrankaReachEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ReachPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
```

The field names mirror the stock Cartpole agent config. If the reach learns poorly, start from the maintained `Isaac-Reach-Franka` agent config instead of tuning these numbers blind. Smoke-test before training: `uv run isaaclab random_agent --task Course-Reach-Franka-Direct --num_envs 16 --viz newton_gl` should show arms twitching and no NaNs.

---

## 5. The hardware view: where the time and the memory go

**Time per transition.** The benchmark tool reports three rates. Turn them into seconds per transition and subtract:

$$
t_{\text{infer}} \approx \frac{1}{\text{FPS}_{\text{collect}}} - \frac{1}{\text{FPS}_{\text{env}}},
\qquad
t_{\text{update}} \approx \frac{1}{\text{FPS}_{\text{total}}} - \frac{1}{\text{FPS}_{\text{collect}}}
$$

Here \( \text{FPS}_{\text{env}} \) is `benchmark runtime` (random actions, env only), \( \text{FPS}_{\text{collect}} \) is rollout collection with the policy (`benchmark training`, `runtime.collection_fps`), and \( \text{FPS}_{\text{total}} \) includes the PPO update (`runtime.total_fps`, the same quantity as RSL-RL's `Perf/total_fps`). Physics alone can be split out further with `ISAACLAB_PHYSICS_PROFILE=1` on a runtime benchmark, but only for tasks whose config sets a non-`None` `benchmark_mode`. That profiled run synchronizes the device, so treat it as a diagnostic, not a throughput number. Profiling a full run is [Lecture 12](Lecture-12.md)'s job.

**Expected shapes as you sweep `num_envs`.** At small \( N \), FPS grows almost linearly, because per-step launch overhead (Python, kernel launches) is amortized over more envs. Then it bends as the SMs saturate, and contact-rich tasks bend earlier. Update time grows with the batch \( N \times T \) and the network, roughly independent of physics. Wall-clock-to-reward is U-shaped in \( N \). Too few envs means slow collection and noisy gradients. Too many means each iteration takes longer, and PPO's sample efficiency per transition falls.

**Backends are not like-for-like.** In the cabinet presets reused above, Newton steps at \( dt = 1/600 \) s with decimation 10, and PhysX at \( 1/60 \) s with decimation 1. Both act at 60 Hz, but Newton does ten physics steps per env step. Report env-steps/s **and** physics-steps/s (env-steps/s × decimation) before you call one engine faster.

**A VRAM model you can check.**

$$
\text{VRAM} \approx V_{\text{fixed}} + N \left( v_{\text{phys}} + v_{\text{render}} \right) + \underbrace{N \cdot T \cdot (d_{\text{obs}} + d_{\text{act}} + c) \cdot 4\,\text{B}}_{\text{rollout buffer}} + V_{\text{net+opt}}
$$

\( V_{\text{fixed}} \) is small on the kit-less Newton path and grows by the Kit + RTX baseline (Lecture 01) once Isaac Sim is loaded. For state tasks the rollout term is negligible: 4,096 envs × 24 steps × 20 floats × 4 B is under 10 MB. For pixels it dominates. RSL-RL 5.5 preallocates its rollout storage as `(num_steps_per_env, num_envs, *obs_shape)` in the observation's dtype, and the camera Cartpole feeds normalized float images with a 2-frame stack (6×96×96 for RGB). With the shipped camera agent (`num_steps_per_env = 64`) and the default 512 envs:

$$
64 \times 512 \times (6 \cdot 96 \cdot 96) \times 4\,\text{B} \approx 7.2\,\text{GB} \; (6.75\,\text{GiB})
$$

That is more than a 4060 has left after the renderer, before counting the CNN or the render products. This is arithmetic from the shipped config, not a measurement, so verify it with `nvidia-smi` in Lab 10c. It tells you which knob to turn: `--num_envs` and resolution scale this term linearly and quadratically. Slimmer render settings do not touch it.

> **8 GB budget.** State tasks (Cartpole, reach, lift) on the default kit-less Newton path should fit thousands of envs, so throughput, not memory, is the limit. Close the desktop GPU apps anyway. Camera tasks are memory-bound: compute the rollout buffer first, then start from roughly 64-128 envs at 64-96 px and grow. Keep `--viz none` for every measured run (a Kit viewport adds the RTX baseline). `physics=isaacsim_physx` loads Isaac Sim and costs you that fixed VRAM, so do backend comparisons at a modest `num_envs`. Rough-terrain locomotion and Lift at full default env counts are where you may need to halve `num_envs`. Fall back to a cloud L40S / RTX PRO 6000 (48 GB) for camera tasks at the default 512 envs, RTX-rendered training, or hyperparameter sweeps that run several seeds concurrently. Remember that the 4060 is below Isaac Sim's 16 GB minimum. These labs are sized to work around that, not because it is a supported configuration.

---

## 6. Build it

Results go in `isaac_bench/train/`, next to Lecture 01's harness. Log VRAM for every run exactly as in Lab 1a (`nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -lms 500`).

### Lab 10a — Cartpole: smoke test, train, and sweep

```bash
uv run isaaclab random_agent --task Isaac-Cartpole --num_envs 16                 # resets and steps?
uv run isaaclab train --rl_library rsl_rl --task Isaac-Cartpole --num_envs 64 --max_iterations 10
uv run isaaclab train --rl_library rsl_rl --task Isaac-Cartpole --seed 1 --run_name s1   # full default run
uv run isaaclab play  --rl_library rsl_rl --task Isaac-Cartpole --checkpoint latest --viz newton_gl
```

Then sweep env-only throughput and VRAM:

```bash
# isaac_bench/train/sweep_runtime.sh  TASK  "N1 N2 ..."  [hydra tokens]   env: TAG, UV_EXTRA
TASK=$1; NS=$2; shift 2
for N in $NS; do
  OUT=isaac_bench/train/runs/${TASK}_N${N}_${TAG:-default}
  mkdir -p "$OUT"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 500 > "$OUT/vram.log" &
  SMI=$!
  uv run ${UV_EXTRA:+--extra $UV_EXTRA} isaaclab benchmark runtime --task "$TASK" --num_envs "$N" \
      --warmup_steps 50 --num_steps 1000 --seed 42 --visualizer none \
      --benchmark_formatter schema,summary --output_path "$OUT" "$@"
  kill $SMI
done
```

```bash
# isaac_bench/train/collect.sh — one CSV row per run (schema path from the 3.0 benchmark docs)
for d in isaac_bench/train/runs/*/; do
  fps=$(jq '.runtime.environment_step_timing.environment_step_fps.mean' "$d"/*schema*.json)
  echo "$(basename "$d"),$fps,$(sort -n "$d/vram.log" | tail -1)"
done > isaac_bench/train/runtime_sweep.csv      # run, env_fps, peak_vram_mib
```

Run `sweep_runtime.sh Isaac-Cartpole "256 1024 4096 16384"`. Plot env-FPS and peak VRAM against `num_envs` on log axes. The slope of the VRAM line is **VRAM per env**. The intercept is your kit-less fixed cost, to compare with Lecture 01's Kit fixed cost.

### Lab 10b — Franka lift and the backend comparison

Train `Isaac-Lift-Franka` at a `num_envs` that fits (start at 1,024), with three seeds (`--seed 1/2/3`). Run `sweep_runtime.sh Isaac-Lift-Franka "256 512 1024 2048 4096"`. Then repeat one size with PhysX: `TAG=physx UV_EXTRA=isaacsim sweep_runtime.sh Isaac-Lift-Franka "1024" physics=isaacsim_physx`. Record env-steps/s, physics-steps/s, and peak VRAM for both. Run `benchmark training` once per backend at the same size:

```bash
uv run isaaclab benchmark training --rl_library rsl_rl --task Isaac-Lift-Franka --num_envs 1024 \
    --max_iterations 50 --warmup_steps 50 --seed 42 --visualizer none \
    --benchmark_formatter schema,summary --output_path isaac_bench/train/runs/lift_train_newton
```

Fill in the \( t_{\text{infer}} \) / \( t_{\text{update}} \) split from §5. Optional: `Isaac-Velocity-Flat-AnymalD` at 1,024 and 4,096 envs, then Rough at the size that fits. Note how much the terrain and height scan cost per env.

### Lab 10c — A camera task inside 8 GB

First write down the §5 rollout-buffer estimate for your planned `num_envs`, resolution, and `num_steps_per_env`. Then:

```bash
uv run isaaclab train --rl_library rsl_rl --task Isaac-Cartpole-Camera-Direct \
    --num_envs 64 --max_iterations 10 \
    physics=newton_mjwarp renderer=newton_renderer presets=rgb \
    env.scene.tiled_camera.width=64 env.scene.tiled_camera.height=64
```

The env config replaces the observation height/width with the camera's at init, so the resolution override flows into the CNN input. Confirm with the 10-iteration smoke run. Sweep `num_envs ∈ {32, 64, 128, 256}` × resolution `∈ {64, 96}`, measure peak VRAM, and compare with your estimate. Find the largest configuration that trains without OOM and record it. If you have the `isaacsim` extra, repeat one point with `renderer=isaacsim_rtx` and note the jump in fixed VRAM.

### Lab 10d — Train your direct reach task

```bash
cd ~/work/isaac_reach
for S in 1 2 3; do
  uv run isaaclab train --rl_library rsl_rl --task Course-Reach-Franka-Direct --seed $S --run_name s$S
done
uv run isaaclab play --rl_library rsl_rl --task Course-Reach-Franka-Direct --checkpoint latest --viz newton_gl
```

Extract wall-clock-to-threshold from the free `Train/mean_reward/time` and the task's success metric:

```python
# isaac_bench/train/wallclock.py  <run_dir> <success_threshold>
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ea = EventAccumulator(sys.argv[1]); ea.Reload()
succ = ea.Scalars("Metrics/success_rate")
t0 = succ[0].wall_time
hit = next((e for e in succ if e.value >= float(sys.argv[2])), None)
print(f"final success={succ[-1].value:.3f}  "
      f"time-to-{sys.argv[2]}={'never' if hit is None else f'{hit.wall_time - t0:.0f}s'} "
      f"(iteration {None if hit is None else hit.step})")
```

Repeat at `--num_envs 256`, `1024` and `4096`. That gives the wall-clock-to-reward U-curve for your GPU.

---

## 7. Use it in the real stack

* **Overrides without editing files:** `agent.algorithm.learning_rate=…`, `env.scene.num_envs=…`, `env.sim.dt=…` are Hydra paths, and the 3.0 docs cover multirun sweeps. For your own sweep drivers there is a programmatic API: `from isaaclab_rl import TrainingRequest, train`.
* **Distillation** (state teacher → vision student) uses the same CLI: `--agent rsl_rl_distillation_cfg_entry_point --checkpoint teacher.pt`. It is the standard way to get a camera policy without paying for camera RL from scratch.
* **Scale and rigor:** multi-GPU (`isaaclab train_multigpu`) and full-run profiling are in [Lecture 12](Lecture-12.md). Seeds and confidence intervals follow [Deep RL Lecture 13](../Deep%20RL%20for%20Robot%20Learning/Lecture-13.md). The reach task seeds the [Lecture 13](Lecture-13.md) capstone, which adds an object, domain randomization (`events` with `EventTermCfg`, modes `startup`/`reset`/`interval`), and a ROS 2 bridge.

---

## 8. Measure it

| Metric | How | Why it matters |
|---|---|---|
| **Env-steps/s** | `benchmark runtime`, `environment_step_fps` | Simulator ceiling for this task and `num_envs` |
| **Physics-steps/s** | env-steps/s × decimation | Fair cross-backend comparison |
| **Collection / total FPS** | `benchmark training`; `Perf/total_fps` | Gives \( t_{\text{infer}} \) and \( t_{\text{update}} \) |
| **Collection vs learning time** | `Perf/collection_time`, `Perf/learning_time` | Which stage to optimize |
| **Peak VRAM, VRAM per env** | `nvidia-smi` log, slope over `num_envs` | How many envs fit |
| **Rollout-buffer estimate vs measured** | §5 formula vs Lab 10c | Validates your memory model for pixels |
| **Wall-clock to threshold** | `wallclock.py` on `Metrics/success_rate` | The number that matters to a team |
| **Final success over seeds** | 3 seeds, mean and spread | Avoid trusting one lucky run |

---

## 9. Ship it

Commit to `isaac_bench/train/`:

* `sweep_runtime.sh`, `collect.sh`, `wallclock.py`, plus `runtime_sweep.csv` (task, backend, `num_envs`, env-steps/s, physics-steps/s, peak VRAM)
* `training_split.csv` (the three FPS values and the derived \( t_{\text{infer}} \), \( t_{\text{update}} \)) and `camera_budget.csv` (`num_envs`, resolution, estimated buffer, measured peak VRAM, OOM yes/no)
* `TRAINING.md`: the throughput and VRAM plots, the wall-clock-to-success table for the reach task across `num_envs` and seeds, and one paragraph on which stage dominates on your 4060 and why

And the `isaac_reach/` project with the reach task, its agent config, and the three-seed logs (or their TensorBoard exports).

---

## Exit criteria

You can move on when you can:

* launch, resume, play and benchmark any core task, with backend selectors and Hydra overrides, without looking up flags
* read an RSL-RL TensorBoard run and say whether it is learning, exploring, unstable, or reward hacking
* state, for one task on your GPU, the split between env stepping, inference, and update, with measured numbers
* estimate a camera task's rollout-buffer memory before launching it, and be within the right order of magnitude
* write a direct task from scratch, explain why `_get_dones()` computes the distance, and train it to a stated success rate over three seeds

---

## Self-check

1. Your Lift run shows `Perf/collection_time` at a small fraction of `Perf/learning_time`. A teammate proposes switching physics backends to go faster. What do you tell them, and which two agent fields would you look at first?
2. `Isaac-Cartpole-Camera-Direct` OOMs at the default 512 envs on your 4060, even with `--viz none` and a Newton renderer. Using the rollout-buffer formula, propose two changes and estimate the memory each saves.
3. You compare `physics=newton_mjwarp` and `physics=isaacsim_physx` on your reach task and Newton reports fewer env-steps/s. Before concluding that PhysX is faster, what must you check in the config, and what number should you report alongside?
4. A custom direct task computes the end-effector distance in `_get_rewards()` and uses it in `_get_dones()` for success-based termination. Training is unstable and episodes end one step "late". Explain the bug from the `DirectRLEnv.step()` order and fix it.
5. Doubling `num_envs` from 2,048 to 4,096 raised env-steps/s by a large fraction, but time-to-90%-success barely changed. Give two reasons from the PPO loop.

---

## References

* Isaac Lab 3.0 — [Reinforcement Learning (train/play/checkpoints/troubleshooting)](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/concepts/reinforcement_learning.html), [Training with an RL agent](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/how-to/run_rl_training.html), [Quickstart](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/setup/quickstart.html)
* Isaac Lab 3.0 — [Backends and Presets](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/concepts/backends_and_presets.html), [Run benchmarks](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/developer-tools/benchmarking/run_benchmarks.html), [Reproducibility](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/features/reproducibility.html)
* Isaac Lab 3.0 — [Creating a Direct Workflow RL Environment](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/how-to/create_direct_rl_env.html), [Build your own project or task (template generator)](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/developer-tools/template_generator.html), [SO-101 end-to-end tutorial](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/setup/tutorial.html)
* Isaac Lab `release/3.0.0` source — [`direct_rl_env.py`](https://github.com/isaac-sim/IsaacLab/blob/release/3.0.0/source/isaaclab/isaaclab/envs/direct_rl_env.py), [`cartpole_direct_env.py`](https://github.com/isaac-sim/IsaacLab/blob/release/3.0.0/source/isaaclab_tasks/isaaclab_tasks/core/cartpole/cartpole_direct_env.py), [`cabinet_direct_env.py`](https://github.com/isaac-sim/IsaacLab/blob/release/3.0.0/source/isaaclab_tasks/isaaclab_tasks/core/cabinet/cabinet_direct_env.py)
* Isaac Lab v3.0.0-EA release notes — [GitHub](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0-EA)
* RSL-RL — [GitHub](https://github.com/leggedrobotics/rsl_rl) (logger and rollout storage read at tag `v5.5.1`)
* Rudin, Hoeller, Reist, Hutter, "Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning," CoRL 2021 — [arXiv:2109.11978](https://arxiv.org/abs/2109.11978)
* Isaac Lab paper, "Isaac Lab: A GPU-Accelerated Simulation Framework for Multi-Modal Robot Learning," 2025 — [arXiv:2511.04831](https://arxiv.org/abs/2511.04831)

---

## Next in this special course

* Next: [Lecture 10b — Worked Example: Spot Waypoint Navigation](Lecture-10b.md)
* Previous: [Lecture 09 — Isaac Lab 3.0 Architecture](Lecture-09.md)
* Back: [Isaac Sim and Isaac Lab — Overview](README.md)
