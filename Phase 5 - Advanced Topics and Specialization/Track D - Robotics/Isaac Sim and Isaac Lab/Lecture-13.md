# Lecture 13: Capstone: A Task, a Policy, and a Bridge

## Overview

The capstone joins the course into one artifact another engineer can run. You build a **Franka push-to-goal task** as a direct Isaac Lab 3.0 environment with domain randomization, and train it with RSL-RL PPO inside 8 GB. You evaluate it with fixed seeds and confidence intervals. Then you **take the policy out of Isaac Lab**: you export it, run it as a ROS 2 Jazzy node, and drive a separate Isaac Sim 6.1 scene through the ROS 2 bridge, measuring end-to-end latency and real-time factor. A `PERFORMANCE.md` ties it together: what fit in 8 GB, where the time went, and what you would change on a 16 GB or 48 GB card.

The hard part is not any single step. It is the **seams** between them, where the same policy meets a different runtime. Most capstones that "work in training and fail in the bridge" fail on a seam: a joint-order mismatch, a quaternion convention, a missing solver override, or a latency nobody modeled. This lecture treats every seam as a contract that is written down and checked.

By the end you should be able to:

* specify a manipulation MDP precisely (observations, actions, rewards, terminations, resets, randomization) and implement it as a `DirectRLEnv`
* size training to an 8 GB VRAM budget from measurements, not guesses
* report success rates with Wilson intervals over disjoint training and evaluation seeds, on nominal and shifted dynamics
* export a policy and run it outside Isaac Lab against Isaac Sim over ROS 2, with the policy contract checked on both sides
* measure end-to-end latency and real-time factor, and separate contract bugs from timing effects
* write a performance report that predicts what changes on bigger GPUs

---

## 1. Why it matters: three runtimes, one policy

| Runtime | What runs | Who owns time | What can silently differ |
|---|---|---|---|
| **Isaac Lab training** | 1,000+ cloned envs, PhysX on GPU, PPO | The env loop (fixed `dt`, decimation) | Nothing yet; this defines the contract |
| **Isaac Lab evaluation** | Same task, DR off or shifted, exported `policy.pt` | Same | Normalization, action clipping, play-mode overrides |
| **Isaac Sim + ROS 2** | One robot, one cube, standalone Isaac Sim script, policy in another process | Wall clock, DDS, two processes | Joint order, quaternion order, solver settings, gains, friction, latency |

The third row is a simulator integration exercise with separate processes, DDS transport and independently scheduled work. Isaac Sim also reads the raw USD without every Isaac Lab spawner override. Passing this exercise does not isolate all remaining failures to physics: hardware timing, state estimation, calibration, driver behavior and operating conditions still need their own validation.

---

## 2. Mental model: the policy contract

A checkpoint maps an ordered observation vector to an ordered action vector. It does not include the physics engine, the timing or the joint names. Isaac Lab 3.0's PhysX↔Newton transfer guide lists what must match exactly between runtimes. The capstone uses the same list:

| Contract item | Capstone value | Checked by |
|---|---|---|
| Joint order and names | `panda_joint1..7, panda_finger_joint1, panda_finger_joint2` | Assertion in the env; the ROS node maps by **name** |
| Action | 7 arm-joint target **deltas**, clipped to [-1, 1], × 0.05 rad, integrated and clamped to soft limits; gripper held closed | Equivalent target integration in training and the tested simulator-owned `CommandGate` |
| Observation | `[q − q0 (9), 0.1·q̇ (9), cube_pos_b (3), goal_pos_b (3), last_action (7)]` = 31 | Shared `build_obs()` layout + lockstep parity gate |
| Frames | Robot base = env origin, identity rotation; positions in the base frame | Same placement in the bridge scene |
| Quaternions | Isaac Lab 3.0 **xyzw**; Isaac Sim experimental prims **wxyz** | Explicit conversion at every crossing |
| Timing | Physics `dt` = 1/120 s, decimation 2 → policy at 60 Hz; zero-order hold | `contract_runtime.json` read by the bridge |
| Mechanism | Same USD (`.../IsaacLab/Robots/FrankaEmika/franka_panda.usda`, Physics=`physx`), 8/0 solver iterations, self-collisions off | The bridge sets solver iterations and self-collisions; it does **not** yet copy the rest of `FRANKA_PANDA_CFG` (passive `panda_finger_joint2`, effort/velocity limits, joint friction, `max_depenetration_velocity`) or the cube's physics material. Copy those too, or expect the lockstep gate to show the gap |
| Policy state | No observation normalizer (`obs_normalization=False`) and no recurrence | Agent config |

```text
   capstone_tasks/push/contract.py  (pure Python + NumPy, no Isaac imports)
        │                         │                               │
        ▼                         ▼                               ▼
  PushEnv (Isaac Lab)     eval_policy.py (Lab 12)        capstone_policy_node.py (ROS 2 Jazzy, CPU torch)
  writes contract_runtime.json  ─────────────────────►   isaac_capstone_bridge.py (Isaac Sim 6.1) reads it
```

Turning off observation normalization is a deliberate simplification: a normalizer is policy *state* that has to travel with the checkpoint. The observation terms are scaled to order 1 by hand instead.

---

## 3. Project specification

| Item | Specification |
|---|---|
| Robot | Franka Panda, `FRANKA_PANDA_CFG` from `isaaclab_assets` (PhysX variant, primitive colliders, implicit actuators with USD gains) |
| Object | 5 cm cube, nominal 0.2 kg, on the ground plane, spawned at x ∈ [0.40, 0.60] m, y ∈ [−0.15, 0.15] m |
| Goal | 0.10-0.20 m from the cube start in a random direction, x clamped to [0.30, 0.70] m |
| Episode | 6 s (360 policy steps), terminates early if the cube leaves the workspace |
| Success | Final cube-goal distance in the plane < 3 cm |
| Reward | Reach a push point behind the cube (tanh kernel) + cube-to-goal (tanh) + success bonus − action rate − joint velocity, × `step_dt` |
| Randomization | Startup: cube friction (static 0.5-1.0) and mass (×0.5-2.0). Per reset: action delay of 0-2 policy steps. Always: Gaussian observation noise σ = 0.005 |
| Shifted eval | Friction 0.3, mass ×3, delay 3 steps, all **outside** the training ranges |
| Algorithm | RSL-RL PPO, MLP actor/critic [256, 128, 64], ≥ 3 training seeds |
| Bridge | Isaac Sim 6.1 standalone scene + rclpy publishers; policy as a ROS 2 Jazzy node on CPU |

---

## 4. The hardware view: the 8 GB plan

Plan VRAM with the Lecture 01 vocabulary before writing reward code:

| Consumer | Training | Bridge run | Source of the number |
|---|---|---|---|
| Desktop + Kit + renderer fixed cost | Yes | Yes (a second, separate instance) | Lecture 01 Lab 1a |
| Per-env physics (Franka primitives + cube + contacts) | × `num_envs` | × 1 | Slope of the Lab 13c sweep |
| PyTorch: policy, critic, Adam, rollout | Small: the rollout is \(N S (o + a + 4)\) floats, a few MB at \(N\) = 1,024 and \(S\) = 24 | 0 on GPU (policy on CPU) | `torch.cuda.max_memory_reserved()` |
| Cameras | None (state observations) | None (headless) or one viewport | Lecture 07 if you add the camera variant |

The decisions that follow from the table:

* **Training and the bridge never run at the same time on 8 GB.** Each pays the Kit fixed cost.
* **The policy node runs on CPU.** A CPU policy avoids a second CUDA context, but its latency must be measured. Even the supplied tiny known-weight fixture has a much slower cold call than subsequent calls; do not infer a deadline from the network dimensions.
* Choose `num_envs` from the measured sweep at about **80% of the VRAM that is left** after fixed cost. PPO needs headroom for the update, and the profiler from Lab 12a needs more.

> **8 GB experiment.** The 4060 is below the published full-simulator minimum. Training fit and throughput have not been established here; first pass the startup check and use Lab 13c to find a measured budget. If startup itself fails, reducing the environment count cannot necessarily remove the fixed cost. Turn down: `num_envs` first, then contact-heavy colliders (`FRANKA_MINIMAL_CFG` keeps gripper colliders only, but you must then re-qualify the policy, because arm-floor contact disappears). Never train with `--viz kit` or a livestream attached. The optional camera variant is the first thing that will not fit comfortably; size it with Lecture 07's per-render-product cost, or rent an L40S / RTX PRO 6000 for it. Budget any VLA server separately using its actual weights, cache and runtime; use a remote or separate-process server when the combined measurement exceeds memory.

**On 16 GB or 48 GB:** repeat the same memory and timing sweep before increasing environments, adding cameras, or overlapping a bridge with training. Capacity alone does not establish that concurrent workloads fit or meet deadlines.

---

## 5. Phases and go/no-go gates

| Phase | Work | Go/no-go gate (all must pass) |
|---|---|---|
| **P0 Scaffold** | External project, `contract.py`, empty task | `isaaclab random_agent --task Capstone-Push-Franka-Direct --num_envs 16` runs; joint-order assertion passes; cubes and goals spawn in reach |
| **P1 Task** | Rewards, terminations, resets, DR | Per-term episode sums logged; random policy success near zero; no NaNs or exploding cubes in 10⁵ random steps |
| **P2 Train** | `num_envs` sweep, 3 seeds | All seeds learn; mean success on the nominal eval clears a target you **pre-register** in `PLAN.md` before training |
| **P3 Evaluate** | Nominal and shifted, disjoint seeds | Wilson CIs for every (seed, condition); checkpoints chosen on validation seeds only |
| **P4 Bridge** | Export and the atomic bridge assets below | Known-observation action parity first; matched held-out scenarios with a predeclared regression margin; expired/duplicate/old-episode replies rejected; observed deadlines reported with failures |
| **P5 Report** | `PERFORMANCE.md`, manifest | A second person reproduces one number from `RUN_MANIFEST.json` |

---

## 6. Risk register

| Risk | Early signal | Mitigation |
|---|---|---|
| **Sim instability** (cube jitter, gripper penetration) | Cube z drifts at rest; reward spikes; NaNs | Primitive colliders; check contact/rest offsets and iterations ([Lecture 04](Lecture-04.md)); smaller `dt` before larger gains |
| **Reward hacking** | Reward rises, success flat; the cube is flicked or pinned | Watch `--video` clips every few hundred iterations; log per-term sums; success bonus only on final distance |
| **VRAM OOM** | Crash at startup or first PPO update | Lower `num_envs`; `--viz none`; close desktop GPU apps; profile at reduced `num_envs` |
| **Isaac Lab 3.0 EA churn** | GA (targeted for late October 2026) renames or deprecates APIs | Pin the Isaac Lab commit SHA in the manifest; keep task code on documented APIs (`actuators.target_command`, `*_index` writers); rerun P0-P3 gates after any upgrade |
| **Contract drift** | Bridge success far below Lab even in lockstep | Joint-name assertion, shared `contract.py`, runtime JSON, lockstep parity gate |
| **DDS / ROS 2 setup** | No messages, or bursty arrival | Same `ROS_DOMAIN_ID`; `ros2 topic hz`; Lecture 08's bridge checklist |
| **No convergence in budget** | Flat success after a large fraction of iterations | Curriculum: start with small goal offsets and no delay, widen later; check reward scales before tuning PPO |

---

## 7. Build it

Layout (an external project created from Isaac Lab's template with `uv run isaaclab --new`, as in [Lecture 10](Lecture-10.md) §4.1, whose `pyproject.toml` declares the `isaaclab.tasks` entry point):

```text
capstone/
  capstone_tasks/push/{__init__.py, contract.py, push_env_cfg.py, push_env.py, agents/rsl_rl_ppo_cfg.py}
  bridge/{isaac_capstone_bridge.py, capstone_policy_node.py, capstone_protocol.py, ros2_capstone_replay.py}
  results/  PLAN.md  PERFORMANCE.md  RUN_MANIFEST.json
```

The template puts task code under `<name>/tasks/`. Map this layout onto it however you like. What matters is that the `isaaclab.tasks` entry point imports the module that calls `gym.register`, since `isaaclab train` and `eval_policy.py` both load tasks through that entry point.

### Lab 13a — The contract and the scaffold

```python
# capstone_tasks/push/contract.py — imported by the Isaac Lab task AND the ROS 2 node; no Isaac imports
import json, os
import numpy as np

JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)] + ["panda_finger_joint1", "panda_finger_joint2"]
ARM_JOINTS = JOINT_NAMES[:7]
PHYSICS_DT, DECIMATION = 1 / 120, 2            # policy at 60 Hz
ACTION_SCALE, QD_SCALE, GRIPPER_CLOSED = 0.05, 0.1, 0.0
CUBE_SIZE, CUBE_MASS = 0.05, 0.2

def integrate_target(target, a, lo, hi):       # same code on torch tensors and NumPy arrays
    return (target + ACTION_SCALE * a).clip(lo, hi)

def build_obs(q, qd, q0, cube_b, goal_b, last_a):
    return np.concatenate([q - q0, QD_SCALE * qd, cube_b, goal_b, last_a]).astype(np.float32)

def write_runtime_contract(path, joint_names, q0, limits):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "contract_runtime.json"), "w") as f:
        json.dump(dict(joint_names=list(joint_names), q0=q0.tolist(), limits=limits.tolist(),
                       physics_dt=PHYSICS_DT, decimation=DECIMATION, action_scale=ACTION_SCALE), f, indent=1)
```

Register three IDs in `push/__init__.py` with `gym.register`, following Isaac Lab's `core/cartpole/__init__.py`. All three use `entry_point=f"{__name__}.push_env:PushEnv"`, `rsl_rl_cfg_entry_point=...rsl_rl_ppo_cfg:PushPPORunnerCfg` and `default_agent="rsl_rl"`. Their `env_cfg_entry_point`s are `PushEnvCfg` for `Capstone-Push-Franka-Direct`, `PushEvalNominalCfg` for `Capstone-Push-Franka-Eval-Nominal`, and `PushEvalShiftedCfg` for `Capstone-Push-Franka-Eval-Shifted`.

### Lab 13b — The direct task

```python
# capstone_tasks/push/push_env_cfg.py
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg, mdp
from isaaclab.managers import EventTermCfg as EventTerm, SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass, replace
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG
from . import contract as C

@configclass
class PushSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))
    robot: ArticulationCfg = replace(FRANKA_PANDA_CFG, prim_path="{ENV_REGEX_NS}/Robot")
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(size=(C.CUBE_SIZE,) * 3, rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
                                  mass_props=sim_utils.MassCfg(mass=C.CUBE_MASS),
                                  collision_props=sim_utils.UsdPhysicsCollisionCfg(),
                                  physics_material=sim_utils.RigidBodyMaterialCfg(),
                                  visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1))),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, C.CUBE_SIZE / 2)))  # rot default (0,0,0,1) = xyzw

def _friction(lo, hi):                          # startup mode: see Lecture 12 §5
    return EventTerm(func=mdp.randomize_rigid_body_material, mode="startup",
                     params={"asset_cfg": SceneEntityCfg("cube"), "static_friction_range": (lo, hi),
                             "dynamic_friction_range": (lo - 0.1, hi - 0.1), "restitution_range": (0.0, 0.0),
                             "num_buckets": 16})

def _mass(lo, hi):
    return EventTerm(func=mdp.randomize_rigid_body_mass, mode="startup",
                     params={"asset_cfg": SceneEntityCfg("cube"), "mass_distribution_params": (lo, hi),
                             "operation": "scale"})

@configclass
class TrainEvents:
    friction = _friction(0.5, 1.0)
    mass = _mass(0.5, 2.0)

@configclass
class ShiftEvents:                               # outside the training ranges
    friction = _friction(0.3, 0.3)
    mass = _mass(3.0, 3.0)

@configclass
class PushEnvCfg(DirectRLEnvCfg):
    decimation = C.DECIMATION
    episode_length_s = 6.0
    action_space = 7
    observation_space = 31
    state_space = 0
    sim: SimulationCfg = SimulationCfg(dt=C.PHYSICS_DT, render_interval=C.DECIMATION, physics=PhysxCfg())
    scene: PushSceneCfg = PushSceneCfg(num_envs=1024, env_spacing=2.0, replicate_physics=True)
    events: TrainEvents = TrainEvents()
    observation_noise_model = NoiseModelCfg(noise_cfg=GaussianNoiseCfg(std=0.005))
    min_action_delay: int = 0
    max_action_delay: int = 2                    # policy steps
    cube_xy_range = ((0.40, 0.60), (-0.15, 0.15))
    goal_offset_range = (0.10, 0.20)
    success_tol = 0.03
    ee_offset = (0.0, 0.0, 0.1034)               # panda_hand → fingertip center (as in Isaac Lab's Franka cabinet task)
    w_reach = 1.0
    w_push = 4.0
    w_success = 10.0
    w_rate = -0.01
    w_qd = -1.0e-4

@configclass
class PushEvalNominalCfg(PushEnvCfg):
    events = None
    observation_noise_model = None
    max_action_delay = 0

@configclass
class PushEvalShiftedCfg(PushEnvCfg):
    observation_noise_model = None             # match nominal evaluation noise; vary declared dynamics only
    events: ShiftEvents = ShiftEvents()
    min_action_delay = 3
    max_action_delay = 3
```

```python
# capstone_tasks/push/push_env.py
import math, torch
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import combine_frame_transforms, sample_uniform
from . import contract as C

class PushEnv(DirectRLEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.robot, self.cube = self.scene["robot"], self.scene["cube"]
        if list(self.robot.joint_names) != C.JOINT_NAMES:                  # the contract is checked, not assumed
            raise RuntimeError(f"joint order {self.robot.joint_names} != {C.JOINT_NAMES}")
        dev, n = self.device, self.num_envs
        self.arm_ids, self.finger_ids = torch.arange(7, device=dev), torch.tensor([7], device=dev)  # finger2 = passive mimic
        self.finger_target = torch.full((n, 1), C.GRIPPER_CLOSED, device=dev)
        self.hand_idx = self.robot.find_bodies("panda_hand")[0][0]
        self.q0 = self.robot.data.default_joint_pos.torch.clone()                  # never cache a live .torch view
        lim = self.robot.data.soft_joint_pos_limits.torch[:, :7].clone()
        self.q_lo, self.q_hi = lim[..., 0], lim[..., 1]
        self.target, self.last_a = self.q0[:, :7].clone(), torch.zeros(n, 7, device=dev)
        self.prev_applied = torch.zeros_like(self.last_a)
        self.hist = torch.zeros(n, max(cfg.max_action_delay, 1) + 1, 7, device=dev)  # ≥ 2 slots: hist[:, 1] = previous action
        self.delay = torch.zeros(n, dtype=torch.long, device=dev)
        self.goal = torch.zeros(n, 3, device=dev)
        self.last_episode_success = torch.zeros(n, dtype=torch.bool, device=dev)      # read by eval_policy.py
        self.ee_off = torch.tensor(cfg.ee_offset, device=dev).repeat(n, 1)
        C.write_runtime_contract(cfg.log_dir or ".", self.robot.joint_names, self.q0[0].cpu(), lim[0].cpu())

    def _frames(self):                          # positions in the robot base frame (= env origin, by contract)
        o = self.scene.env_origins
        ee, _ = combine_frame_transforms(self.robot.data.body_pos_w.torch[:, self.hand_idx],
                                         self.robot.data.body_quat_w.torch[:, self.hand_idx], self.ee_off)
        return ee - o, self.cube.data.root_pos_w.torch - o

    def _pre_physics_step(self, actions):
        a = actions.clamp(-1.0, 1.0)
        self.hist = torch.roll(self.hist, 1, dims=1); self.hist[:, 0] = a
        delayed = self.hist[torch.arange(self.num_envs, device=self.device), self.delay]
        self.target = C.integrate_target(self.target, delayed, self.q_lo, self.q_hi)
        self.prev_applied = self.last_a.clone()
        self.last_a = delayed.clone()             # observed history is the action actually applied, as in the bridge

    def _apply_action(self):
        cmd = self.robot.actuators.target_command
        cmd.set_position_index(value=self.target, joint_ids=self.arm_ids)
        cmd.set_position_index(value=self.finger_target, joint_ids=self.finger_ids)

    def _get_observations(self):                # same order as C.build_obs
        _, cube = self._frames()
        d = self.robot.data
        obs = torch.cat((d.joint_pos.torch - self.q0, C.QD_SCALE * d.joint_vel.torch, cube, self.goal, self.last_a), -1)
        return {"policy": obs}

    def _get_rewards(self):
        ee, cube = self._frames()
        to_goal = self.goal[:, :2] - cube[:, :2]
        d_goal = to_goal.norm(dim=-1)
        push_pt = cube.clone()
        push_pt[:, :2] -= C.CUBE_SIZE * to_goal / (d_goal[:, None] + 1e-6)          # just behind the cube
        c = self.cfg
        r = (c.w_reach * (1 - torch.tanh((ee - push_pt).norm(dim=-1) / 0.1))
             + c.w_push * (1 - torch.tanh(d_goal / 0.1)) + c.w_success * (d_goal < c.success_tol).float()
             + c.w_rate * (self.last_a - self.prev_applied).square().sum(-1)
             + c.w_qd * self.robot.data.joint_vel.torch.square().sum(-1))
        return r * self.step_dt

    def _get_dones(self):
        _, cube = self._frames()
        lost = (cube[:, 2] < -0.05) | (cube[:, :2].norm(dim=-1) > 0.9)
        return lost, self.episode_length_buf >= self.max_episode_length

    def _reset_idx(self, env_ids):
        ids = self.robot._ALL_INDICES if env_ids is None else env_ids
        _, cube = self._frames()
        self.last_episode_success[ids] = (self.goal[ids, :2] - cube[ids, :2]).norm(dim=-1) < self.cfg.success_tol
        super()._reset_idx(ids)
        n, dev = len(ids), self.device
        q = self.q0[ids].clone(); q[:, :7] += sample_uniform(-0.05, 0.05, (n, 7), dev)
        q[:, :7] = q[:, :7].clamp(self.q_lo[ids], self.q_hi[ids]); q[:, 7:] = C.GRIPPER_CLOSED
        self.robot.write_joint_position_to_sim_index(position=q, env_ids=ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q), env_ids=ids)
        self.target[ids] = q[:, :7]
        (x0, x1), (y0, y1) = self.cfg.cube_xy_range
        xy = torch.stack((sample_uniform(x0, x1, (n,), dev), sample_uniform(y0, y1, (n,), dev)), -1)
        pose = torch.zeros(n, 7, device=dev)
        pose[:, :2], pose[:, 2], pose[:, 6] = xy + self.scene.env_origins[ids, :2], C.CUBE_SIZE / 2, 1.0  # quat xyzw
        self.cube.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
        self.cube.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=dev), env_ids=ids)
        r, th = sample_uniform(*self.cfg.goal_offset_range, (n,), dev), sample_uniform(-math.pi, math.pi, (n,), dev)
        self.goal[ids, :2] = xy + torch.stack((r * th.cos(), r * th.sin()), -1)
        self.goal[ids, 0] = self.goal[ids, 0].clamp(0.30, 0.70); self.goal[ids, 2] = C.CUBE_SIZE / 2
        self.last_a[ids] = 0.0; self.prev_applied[ids] = 0.0; self.hist[ids] = 0.0
        self.delay[ids] = torch.randint(self.cfg.min_action_delay, self.cfg.max_action_delay + 1, (n,), device=dev)
        self.sim.forward()                       # refresh kinematics before reset observations; do not advance time
```

The CPU fixture [push_tensor_cpu.py](labs/push_tensor_cpu.py) extracts this class and checks delayed applied-action history, action-rate reward, reset isolation and limits with real CPU PyTorch and an explicit fake scene. It does not validate Isaac physics or asset APIs. In particular, issuing action 1 with delay two leaves both the target and observed applied action unchanged for two transitions.

The action and reset calls (`actuators.target_command.set_position_index`, `write_*_to_sim_index`, `.data.*.torch`) are the 3.0 forms used by Isaac Lab's own direct Cartpole and Franka-cabinet tasks. In older code you will see `set_joint_position_target`, `write_root_state_to_sim`, and wxyz root quaternions. For the agent, subclass `RslRlOnPolicyRunnerCfg` the way `core/cartpole/agents/rsl_rl_ppo_cfg.py` does. Use `RslRlMLPModelCfg(hidden_dims=[256, 128, 64], activation="elu", obs_normalization=False)` for the actor and critic, `num_steps_per_env=24`, `experiment_name="capstone_push"`, and the Cartpole PPO hyperparameters as a starting point.

### Lab 13c — Train within 8 GB

```bash
cd capstone
uv run --extra isaacsim isaaclab random_agent --task Capstone-Push-Franka-Direct --num_envs 16 --viz kit   # P0 smoke test
for n in 256 512 1024 2048 4096; do                                                                    # VRAM / throughput sweep
  uv run --extra isaacsim isaaclab benchmark runtime --task Capstone-Push-Franka-Direct --num_envs $n \
    --num_steps 500 --warmup_steps 50 --visualizer none --benchmark_formatter schema,summary --output_path results/sweep_$n
done
for s in 1 2 3; do
  uv run --extra isaacsim isaaclab train --rl_library rsl_rl --task Capstone-Push-Franka-Direct \
    --num_envs $N --seed $s --run_name seed$s --viz none
done
```

Log `nvidia-smi` throughout the sweep. Fit VRAM against `num_envs` (fixed cost + slope × N, as in Lecture 01). Choose `$N` with headroom, and note where env-steps/s stops rising. That knee, not the OOM point, is the useful number. Then run `play` once per seed with an explicit `--checkpoint logs/rsl_rl/capstone_push/<run>/model_<iter>.pt` to export `exported/policy.pt` and `policy.onnx` beside it. `--checkpoint latest` would pick the newest run only. Add `--video` to the play run to keep a clip for the report.

### Lab 13d — Evaluate with fixed seeds and confidence intervals

Pre-register seed lists in `PLAN.md`: training {1, 2, 3}, validation {501-505} (checkpoint selection), test {1001-1005} (reported numbers). Run the Lab 12 evaluator (it reads `last_episode_success`):

```bash
for s in 1 2 3; do for e in 1001 1002 1003 1004 1005; do for c in Nominal Shifted; do
  uv run --extra isaacsim python ../isaac_bench/eval_policy.py --task Capstone-Push-Franka-Eval-$c \
    --policy logs/rsl_rl/capstone_push/*_seed$s/exported/policy.pt --seed $e --num_envs 256 --out results/eval.jsonl
done; done; done
```

Report, per training seed and condition, \(k/n\) with the **Wilson 95% interval** (`stats.py` from [Deep RL Lecture 13](../Deep%20RL%20for%20Robot%20Learning/Lecture-13.md)). Three training seeds give little information about run-to-run variation; show all three episode intervals and per-seed results rather than disguising them as one pooled trial. Also report the **robustness drop** (nominal − shifted) per seed. A collapse under shifted dynamics demonstrates sensitivity to that tested shift. It does not by itself identify memorization, a particular parameter, or the outcome of the bridge experiment.

### Lab 13e — Export and bridge over ROS 2

Start with the [atomic protocol and measured ROS exercise](Simulation-CPU-Lab.md#3-atomic-observations-and-simulator-owned-actions). The simulator sends **one complete observation per tick**, including episode, sequence, controller-clock timestamp, expiry, frame, ordered joint state, cube, goal, current target, and last applied action. A policy reply echoes the observation digest and returns seven normalized actions. The simulator owns integration and accepts each reply at most once.

This prevents a slow reply from advancing a policy-side target that the simulator never applied. It also removes the cross-topic ordering ambiguity of separately publishing goal, cube and joint state. The standard-library gate has deliberate duplicate, expired, future, previous-episode, shape, finite-value and joint-order tests. A six-case Jazzy transport exercise and a separate-process policy adapter with a known-weight TorchScript fixture actually ran. No learned checkpoint or full Isaac scene was executed.

| Supplied asset | Role | Validation status |
|---|---|---|
| [capstone_protocol.py](labs/capstone_protocol.py) | Strict message contract, simulator-owned target/history, freshness and identity checks | 11 portable tests plus real ROS replay |
| [ros2_capstone_replay.py](labs/ros2_capstone_replay.py) | Two localhost ROS nodes and scripted actions | Executed with Jazzy; six outcomes saved |
| [capstone_policy_node.py](labs/capstone_policy_node.py) | Loads the exported 31→7 TorchScript policy on CPU | Executed in a separate process with a known-weight fixture; trained artifact still required for the capstone |
| [isaac_capstone_bridge.py](labs/isaac_capstone_bridge.py) | Creates the Franka/cube scene, publishes atomic observations, applies gated targets | APIs reviewed against 6.1 source; simulator execution pending |

Keep the files together so their imports resolve. From this lesson directory, start the policy in a ROS-sourced Python 3.12 environment that also has CPU PyTorch. Set paths to the artifacts produced by **your** completed training/export run:

```bash
source /opt/ros/jazzy/setup.bash
: "${POLICY_PT:?Set the exported TorchScript policy path}"
: "${RUNTIME_CONTRACT:?Set the matching contract_runtime.json path}"
python3 labs/capstone_policy_node.py --policy "$POLICY_PT" --contract "$RUNTIME_CONTRACT" \
  --domain-id 77 --namespace /course_push_trial
```

In a second terminal, use the pinned Isaac Sim Python environment, with `RUNTIME_CONTRACT` set to the same file:

```bash
: "${RUNTIME_CONTRACT:?Set the matching contract_runtime.json path}"
python labs/isaac_capstone_bridge.py --contract "$RUNTIME_CONTRACT" \
  --domain-id 77 --namespace /course_push_trial --headless --lockstep \
  --output push-lockstep.json
```

For a workstation installation, replace `python` with its `python.sh` launcher. Both adapters constrain discovery to localhost and use the specified private namespace; choose a distinct namespace for concurrent experiments. The replay needs only `rclpy` and `std_msgs`; the policy additionally needs PyTorch and the simulator additionally needs the actual pinned Isaac environment and asset access.

**Lockstep:** physics waits for the reply to the current observation. The 500 ms wall-clock transport timeout is a diagnostic bound, not a 60 Hz control guarantee. A timeout fails the run; it does not silently apply an older reply. Compare exported and training actions on saved identical observations before evaluating rollouts.

**Free-running:** omit `--lockstep`. Each request expires after one policy period, and physics advances on an absolute wall-clock schedule. A rejected or missing response leaves the last bounded joint target in place; it does not integrate another delta. Logs separate accepted action age, expiration, deadline misses and success. Holding a target is the defined **simulator** fallback, not proof of a suitable physical emergency response. Inspect cold-start inference separately and warm the policy before a timing comparison.

**Reconcile the physics before interpreting success.** The adapter sets the named USD/variants, masses, solver iterations and reset pose, but it does not automatically reproduce every `FRANKA_PANDA_CFG` actuator/limit/friction override or training cube material. Compare resolved values, including passive mimic joints, max efforts, velocities, friction and depenetration settings, and port the required overrides. The initial-state RNG in the adapter is an example, not a paired match to all Lab evaluation seeds. Feed the same saved scenario table to both environments for paired evaluation. The action-delay buffer and accumulated target make the training task partially observed under its 31-value observation; adding target/history or recurrent state is a separate, documented policy-design experiment requiring retraining.

**Do not use overlapping confidence intervals as a parity test.** Overlap can occur despite meaningful differences, and a mismatch can arise from several causes. First check numerical action parity and frame/ordering contracts. Then define a tolerated regression on matched held-out scenarios before seeing results. The [paired CPU example](labs/simulation_semantics.py) shows why zero regressions in 100 episodes still cannot establish a one-percent upper regression bound at 95% confidence. This statistical gate is evidence about its specified outcome/distribution, not a proof of dynamics equivalence.

**Optional camera variant.** Replace the cube position in the observation with a small RGB or depth camera (`CameraCfg`; in 3.0 `TiledCamera` is a deprecated alias because `Camera` includes the tiled path). Budget it with Lecture 07's per-render-product cost before training, and expect to need a CNN encoder and many more samples.

---

## 8. Use it in the real stack

* **Real robot:** the bridge scene is where `franka_ros`/`ros2_control` would go ([Lecture 08](Lecture-08.md), [Advanced ROS](../Advanced%20Robot%20Operating%20System/Lecture-01.md)). The contract is a starting specification; revalidate physical units, limits, controller period, feedback freshness and fallback behavior on the real stack. Friction, mass and actuator dynamics are what sysid ([Lecture 12](Lecture-12.md) §6.2) should pin down first.
* **Sim-to-sim as a habit:** Isaac Lab's PhysX↔Newton transfer guide uses exactly this contract-first method across physics engines. Running your policy under `physics=newton_mjwarp` is a cheap second rehearsal. It needs the Franka `mujoco` physics variant and joint-ordering settings from that guide.
* **Rung 2 of the Deep RL ladder:** this task, with its measured env-steps/s, is the simulator rung in [Deep RL for Robot Learning](../Deep%20RL%20for%20Robot%20Learning/README.md).

---

## 9. Measure it

| Metric | How | Target or use |
|---|---|---|
| **Fixed VRAM, VRAM per env, chosen `num_envs`** | Lab 13c sweep + `nvidia-smi` | The 8 GB budget table |
| **Env-steps/s at chosen `num_envs`** | `benchmark runtime` | Throughput knee |
| **Step-time breakdown** | Lab 12a method on this task | Physics vs env Python vs update |
| **Wall-clock to target success, per seed** | Training logs | The number that matters for cost |
| **Success, nominal and shifted** (Wilson 95%) | Lab 13d | Per seed; robustness drop |
| **Bridge success: lockstep / free-running** | `isaac_capstone_bridge.py` | Contract gap / latency cost |
| **Achieved RTF and deadline misses (free-running)** | `physics_steps · dt / wall_s`; `step_deadline_misses` | Overall RTF includes reset/setup; step misses concern the per-episode schedule |
| **Accepted action age; inference and holds** | `age_p50_ns`, `age_p99_ns`, `expired_requests_held`; node `inference_ns` | Accepted ages exclude rejected/expired replies, so report hold counts too |

---

## 10. Ship it

The capstone deliverables are the README's *What you ship*, completed:

* **`isaac_bench/`** with results for every lab, including `eval_policy.py` and the Lecture 12 breakdown
* **USD assets** from Lectures 02 and 06 (your imported, validated robot and work cell). Stretch goal: swap them into the capstone scene and requalify the policy.
* **`capstone/capstone_tasks/push/`**: task, contract, agent config, trained checkpoints and exported `policy.pt` for 3 seeds, learning curves (`results/curves.png`), eval table with CIs, and a video
* **`capstone/bridge/`**: all four local bridge/protocol/helper files and saved `push-lockstep.json` / `push-free-running.json` runs
* **`PERFORMANCE.md`** and `RUN_MANIFEST.json`:

```markdown
# PERFORMANCE — Capstone push (GPU, driver, Isaac Sim 6.1.0, Isaac Lab <SHA>, PhysX, dt 1/120, decimation 2)
## VRAM budget (8 GB): fixed cost | per env | PyTorch | chosen num_envs | headroom
## Step-time breakdown: physics | env Python | policy | PPO update | other   (% of iteration)
## Training: wall-clock to target success, seeds 1-3
## Evaluation: nominal / shifted, k/n and Wilson 95% CI, per seed; robustness drop
## Bridge: lockstep vs free-running success (CI), RTF, step_deadline_misses, expired_requests_held, accepted action age p50/p99, cold/warm inference
## On 16 GB / 48 GB: predicted num_envs, camera variant, bridge-while-training, VLA server — and why
```

---

## Exit criteria

You are done when you can:

* hand the repo to another engineer who reproduces one training curve and one eval number from the manifest
* explain each step function of `PushEnv` and justify each reward term and randomization range
* state the policy contract from memory and show where each item is checked
* show numerical action parity on known observations, then evaluate a predeclared paired task-regression gate and explain its limits
* report RTF and p99 latency for the free-running bridge, and say what would break on a real robot first

---

## Self-check

1. The lockstep bridge succeeds far less often than the Lab nominal eval, but free-running and lockstep agree with each other. Is latency the problem? List the first three contract items you would check, in order.
2. A teammate exports the policy with `obs_normalization=True` and the bridge output is nonsense, while `isaaclab play` looks perfect. Explain using the transfer guide's "policy state" row, and give two fixes.
3. Your `num_envs` sweep shows env-steps/s flat above 1,024 but VRAM still fits 4,096. Which `num_envs` do you train with on the 4060, and what does the flat curve tell you about the bottleneck (Lecture 12, Lab 12a)?
4. Seed 2 reaches nominal success comparable to seeds 1 and 3, but its shifted-dynamics interval is far lower. What do you report, and what does this suggest about how many training seeds a claim needs?
5. In free-running mode the p99 latency exceeds the 16.7 ms policy period while the p50 is small. What does the zero-order hold do to the effective delay the policy sees, and how does your training-time delay range need to relate to that distribution?
6. You upgrade to Isaac Lab 3.0 GA and `set_position_index` has moved. Which gates do you rerun before trusting any old number?

---

## References

* Isaac Lab 3.0 — [Creating a direct workflow RL environment](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/how-to/create_direct_rl_env.html) · [Registering an environment](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/how-to/register_rl_env_gym.html) · [Actuators](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/concepts/actuators.html)
* Isaac Lab 3.0 — [Transfer policies between PhysX and Newton](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/how-to/transfer_policies_between_physx_and_newton.html) (the policy-contract table) · [Policy inference in a USD environment](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/how-to/policy_inference_in_usd.html) · [Reproducibility](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/features/reproducibility.html)
* Isaac Lab 3.0 — [Run benchmarks](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/developer-tools/benchmarking/run_benchmarks.html) · [Migration guide 2.x → 3.0](https://github.com/isaac-sim/IsaacLab/blob/ae37b028ea415c91ea2bc32609efcd759ed2b974/docs/source/migration/migrating_to_isaaclab_3-0.rst)
* Isaac Lab source (`release/3.0.0`) — [direct Cartpole task](https://github.com/isaac-sim/IsaacLab/blob/release/3.0.0/source/isaaclab_tasks/isaaclab_tasks/core/cartpole/cartpole_direct_env.py) · [direct Franka cabinet task](https://github.com/isaac-sim/IsaacLab/blob/release/3.0.0/source/isaaclab_tasks/isaaclab_tasks/core/cabinet/cabinet_direct_env.py) · [Franka asset configs](https://github.com/isaac-sim/IsaacLab/blob/release/3.0.0/source/isaaclab_assets/isaaclab_assets/robots/franka.py)
* Isaac Sim 6.1 — [ROS 2 joint control (Franka)](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/ros2_tutorials/robot_control/tutorial_ros2_manipulation.html) · [ROS 2 bridge in standalone workflow](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/ros2_tutorials/bridge_configuration/tutorial_ros2_python.html) · [`subscriber.py` standalone example](https://github.com/isaac-sim/IsaacSim/blob/v6.1.0/source/standalone_examples/api/isaacsim.ros2.bridge/subscriber.py)
* Isaac Sim 6.1 — [Experimental `Articulation` source](https://github.com/isaac-sim/IsaacSim/blob/v6.1.0/source/extensions/isaacsim.core.experimental.prims/python/impl/articulation.py) (DOF targets, solver iterations; wxyz poses)
* Agarwal et al., *Deep Reinforcement Learning at the Edge of the Statistical Precipice* — [arXiv:2108.13264](https://arxiv.org/abs/2108.13264)
* Peng et al., *Sim-to-Real Transfer of Robotic Control with Dynamics Randomization* — [arXiv:1710.06537](https://arxiv.org/abs/1710.06537)

---

## Next in this special course

* Next: [Deep RL for Robot Learning](../Deep%20RL%20for%20Robot%20Learning/README.md). Use your capstone task as its rung-2 simulator. Then [VLA Optimization and Action-Parity Harness](../VLA%20Optimization%20and%20Action-Parity%20Harness/README.md), where this bridge pattern evaluates VLA policies.
* Previous: [Lecture 12 — Performance, Sim-to-Real, and Scale-Out](Lecture-12.md)
* Back: [Isaac Sim and Isaac Lab — Overview](README.md)
