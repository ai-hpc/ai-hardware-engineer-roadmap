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

The third row is a stand-in for the real robot. It has the same software shape as hardware: a separate process, messages over DDS, and a clock nobody controls. It also adds physics you did not train in, because Isaac Sim reads the raw USD without Isaac Lab's spawner overrides. If the policy survives this seam, the remaining sim-to-real gap is the physical one from [Lecture 12](Lecture-12.md) §6.

---

## 2. Mental model: the policy contract

A checkpoint maps an ordered observation vector to an ordered action vector. It does not include the physics engine, the timing or the joint names. Isaac Lab 3.0's PhysX↔Newton transfer guide lists what must match exactly between runtimes. The capstone uses the same list:

| Contract item | Capstone value | Checked by |
|---|---|---|
| Joint order and names | `panda_joint1..7, panda_finger_joint1, panda_finger_joint2` | Assertion in the env; the ROS node maps by **name** |
| Action | 7 arm-joint target **deltas**, clipped to [-1, 1], × 0.05 rad, integrated and clamped to soft limits; gripper held closed | Shared `integrate_target()` function |
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
  PushEnv (Isaac Lab)     eval_policy.py (Lab 12)        policy_node.py (ROS 2 Jazzy, CPU torch)
  writes contract_runtime.json  ─────────────────────►   sim_bridge.py (Isaac Sim 6.1) reads it
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
* **The policy node runs on CPU.** A 31→7 MLP costs microseconds to milliseconds on a CPU, and a CUDA context in the node would take VRAM from the simulator.
* Choose `num_envs` from the measured sweep at about **80% of the VRAM that is left** after fixed cost. PPO needs headroom for the update, and the profiler from Lab 12a needs more.

> **8 GB budget.** Expect state-only Franka push training to fit on the 4060 somewhere in the hundreds-to-low-thousands of environments. That is an order-of-magnitude guess; your Lab 13c sweep sets the real number. Turn down: `num_envs` first, then contact-heavy colliders (`FRANKA_MINIMAL_CFG` keeps gripper colliders only, but you must then re-qualify the policy, because arm-floor contact disappears). Never train with `--viz kit` or a livestream attached. The optional camera variant is the first thing that will not fit comfortably; size it with Lecture 07's per-render-product cost, or rent an L40S / RTX PRO 6000 for it. A VLA in the loop (the [VLA course](../VLA%20Optimization%20and%20Action-Parity%20Harness/README.md)) does not fit beside Isaac Sim on 8 GB.

**On 16 GB:** more environments per update (check that wall-clock to reward actually improves), the camera variant at small resolution, and the bridge can run while a training job continues. **On 48 GB:** camera policies at useful resolution, many evaluation seeds in parallel, and a VLA policy server beside the simulator.

---

## 5. Phases and go/no-go gates

| Phase | Work | Go/no-go gate (all must pass) |
|---|---|---|
| **P0 Scaffold** | External project, `contract.py`, empty task | `isaaclab random_agent --task Capstone-Push-Franka-Direct --num_envs 16` runs; joint-order assertion passes; cubes and goals spawn in reach |
| **P1 Task** | Rewards, terminations, resets, DR | Per-term episode sums logged; random policy success near zero; no NaNs or exploding cubes in 10⁵ random steps |
| **P2 Train** | `num_envs` sweep, 3 seeds | All seeds learn; mean success on the nominal eval clears a target you **pre-register** in `PLAN.md` before training |
| **P3 Evaluate** | Nominal and shifted, disjoint seeds | Wilson CIs for every (seed, condition); checkpoints chosen on validation seeds only |
| **P4 Bridge** | Export, `sim_bridge.py`, `policy_node.py` | Lockstep bridge success CI overlaps the Lab nominal CI (else it is a contract bug); free-running achieved RTF ≈ 1 with `late_frac` ≈ 0; p99 latency below the 16.7 ms policy period |
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
  bridge/{sim_bridge.py, policy_node.py}
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
        self.last_a = a

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
             + c.w_rate * (self.last_a - self.hist[:, 1]).square().sum(-1)
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
        q = self.q0[ids].clone(); q[:, :7] += sample_uniform(-0.05, 0.05, (n, 7), dev); q[:, 7:] = C.GRIPPER_CLOSED
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
        self.last_a[ids] = 0.0; self.hist[ids] = 0.0
        self.delay[ids] = torch.randint(self.cfg.min_action_delay, self.cfg.max_action_delay + 1, (n,), device=dev)
```

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

Report, per training seed and condition, \(k/n\) with the **Wilson 95% interval** (`stats.py` from [Deep RL Lecture 13](../Deep%20RL%20for%20Robot%20Learning/Lecture-13.md)). Three training seeds are too few for a bootstrap, so show all three intervals rather than one pooled number. Also report the **robustness drop** (nominal − shifted) per seed. A policy that is excellent on nominal and collapses on shifted dynamics has memorized the simulator, and the bridge will expose it.

### Lab 13e — Export and bridge over ROS 2

The bridge scene is a standalone Isaac Sim 6.1 script. It follows the official `isaacsim.ros2.bridge/subscriber.py` example pattern: enable the bridge extension, then use `rclpy` directly inside the simulator process.

```python
# bridge/sim_bridge.py — run with Isaac Sim 6.1's python (bundled Jazzy libs are auto-selected on 24.04)
import argparse, json, time
ap = argparse.ArgumentParser()
ap.add_argument("--contract", required=True); ap.add_argument("--episodes", type=int, default=100)
ap.add_argument("--lockstep", action="store_true"); ap.add_argument("--headless", action="store_true")
args = ap.parse_args()
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": args.headless})
import numpy as np
import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.experimental.objects import Cube, GroundPlane
from isaacsim.core.experimental.prims import Articulation, GeomPrim, RigidPrim
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.storage.native import get_assets_root_path
app_utils.enable_extension("isaacsim.ros2.bridge"); simulation_app.update()
import rclpy
from rclpy.time import Time
from geometry_msgs.msg import PointStamped, PoseStamped
from sensor_msgs.msg import JointState

K = json.load(open(args.contract))
GroundPlane("/World/GroundPlane")
stage_utils.add_reference_to_stage(get_assets_root_path() + "/Isaac/IsaacLab/Robots/FrankaEmika/franka_panda.usda",
                                   "/World/Robot", variants=[("Physics", "physx"), ("Colliders", "primitives")])
robot = Articulation("/World/Robot")
robot.set_solver_iteration_counts(8, 0); robot.set_enabled_self_collisions(False)  # Isaac Lab's spawner overrides
Cube("/World/Cube", sizes=0.05, positions=np.array([[0.5, 0.0, 0.025]]))
GeomPrim("/World/Cube", apply_collision_apis=True)
cube = RigidPrim("/World/Cube", masses=np.array([0.2]))
SimulationManager.setup_simulation(dt=K["physics_dt"], device="cuda")

rclpy.init(); node = rclpy.create_node("capstone_sim")
pub_js = node.create_publisher(JointState, "/capstone/joint_states", 10)
pub_cube = node.create_publisher(PoseStamped, "/capstone/cube_pose", 10)
pub_goal = node.create_publisher(PointStamped, "/capstone/goal", 10)
latest, lat_ms = {}, []
def on_cmd(m):
    latest["cmd"] = m
    lat_ms.append((node.get_clock().now() - Time.from_msg(m.header.stamp)).nanoseconds / 1e6)
node.create_subscription(JointState, "/capstone/joint_command", on_cmd, 10)

app_utils.play(); simulation_app.update()
names = list(robot.dof_names)                                     # map by NAME: Isaac Sim's order need not match Lab's
q_init = np.array([0.0 if "finger" in nm else K["q0"][K["joint_names"].index(nm)] for nm in names])  # gripper closed
rng, succ, n_steps, late, t0 = np.random.default_rng(7), [], 0, 0, time.perf_counter()
for ep in range(args.episodes):
    xy = rng.uniform([0.40, -0.15], [0.60, 0.15]); th = rng.uniform(-np.pi, np.pi)
    goal = xy + rng.uniform(0.10, 0.20) * np.array([np.cos(th), np.sin(th)]); goal[0] = np.clip(goal[0], 0.30, 0.70)
    cube.set_world_poses(positions=[[*xy, 0.025]], orientations=[[1.0, 0.0, 0.0, 0.0]])   # wxyz in Isaac Sim!
    cube.set_velocities(np.zeros((1, 3)), np.zeros((1, 3)))
    robot.set_dof_positions(q_init[None]); robot.set_dof_position_targets(q_init[None]); latest.clear()
    g = PointStamped(); g.point.x, g.point.y, g.point.z = float(goal[0]), float(goal[1]), 0.025; pub_goal.publish(g)
    for k in range(int(6.0 / K["physics_dt"])):
        if k % K["decimation"] == 0:
            stamp = node.get_clock().now().to_msg()
            js = JointState(name=names, position=robot.get_dof_positions().numpy()[0].tolist(),
                            velocity=robot.get_dof_velocities().numpy()[0].tolist()); js.header.stamp = stamp
            p, q = (a.numpy()[0] for a in cube.get_world_poses())
            cp = PoseStamped(); cp.header.stamp = stamp
            cp.pose.position.x, cp.pose.position.y, cp.pose.position.z = map(float, p)
            cp.pose.orientation.w, cp.pose.orientation.x, cp.pose.orientation.y, cp.pose.orientation.z = map(float, q)  # wxyz → msg
            pub_cube.publish(cp); pub_js.publish(js)
            deadline = time.perf_counter() + 0.5
            while True:                                            # lockstep waits for the answer to THIS observation
                rclpy.spin_once(node, timeout_sec=0.0 if not args.lockstep else 0.002)
                c = latest.get("cmd")
                if not args.lockstep or (c and c.header.stamp == stamp) or time.perf_counter() > deadline:
                    break
            if (c := latest.get("cmd")) is not None:                # zero-order hold on the latest command
                robot.set_dof_position_targets(np.array([c.position]), dof_indices=[names.index(nm) for nm in c.name])
        SimulationManager.step(); n_steps += 1
        if not args.lockstep:                                       # pace to the wall clock, like a real robot
            slack = t0 + n_steps * K["physics_dt"] - time.perf_counter()
            if slack > 0: time.sleep(slack)
            else: late += 1
    succ.append(float(np.linalg.norm(cube.get_world_poses()[0].numpy()[0, :2] - goal) < 0.03))
rtf = n_steps * K["physics_dt"] / (time.perf_counter() - t0)
print(json.dumps(dict(lockstep=args.lockstep, k=int(sum(succ)), n=len(succ), rtf=rtf, late_frac=late / n_steps,
                      lat_p50=float(np.percentile(lat_ms, 50)), lat_p99=float(np.percentile(lat_ms, 99)))))
simulation_app.close()
```

```python
# bridge/policy_node.py — system ROS 2 Jazzy shell (Python 3.12) + CPU torch; NOT inside Isaac Sim
import json, sys, time, numpy as np, rclpy, torch
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, PoseStamped
from sensor_msgs.msg import JointState
sys.path.insert(0, "capstone_tasks/push")              # import contract.py by path: the package __init__ pulls in Isaac Lab
import contract as C

class PolicyNode(Node):
    def __init__(self, policy_path, contract_path):
        super().__init__("capstone_policy")
        K = json.load(open(contract_path)); self.names = K["joint_names"]
        self.q0, lim = np.array(K["q0"], np.float32), np.array(K["limits"], np.float32)
        self.lo, self.hi = lim[:, 0], lim[:, 1]
        self.policy = torch.jit.load(policy_path, map_location="cpu").eval()
        self.cube = self.goal = None; self.infer_ms = []; self.reset_state()
        self.pub = self.create_publisher(JointState, "/capstone/joint_command", 10)
        self.create_subscription(PoseStamped, "/capstone/cube_pose", self.on_cube, 10)
        self.create_subscription(PointStamped, "/capstone/goal", self.on_goal, 10)
        self.create_subscription(JointState, "/capstone/joint_states", self.on_joints, 10)

    def reset_state(self):
        self.target, self.last_a = self.q0[:7].copy(), np.zeros(7, np.float32)

    def on_cube(self, m): p = m.pose.position; self.cube = np.array([p.x, p.y, p.z], np.float32)
    def on_goal(self, m): self.goal = np.array([m.point.x, m.point.y, m.point.z], np.float32); self.reset_state()

    def on_joints(self, m):
        if self.cube is None or self.goal is None:
            return
        i = {n: j for j, n in enumerate(m.name)}
        q = np.array([m.position[i[n]] for n in self.names], np.float32)
        qd = np.array([m.velocity[i[n]] for n in self.names], np.float32)
        t = time.perf_counter()
        with torch.inference_mode():
            a = self.policy(torch.from_numpy(C.build_obs(q, qd, self.q0, self.cube, self.goal, self.last_a))[None])[0].numpy()
        self.infer_ms.append((time.perf_counter() - t) * 1e3)
        a = np.clip(a, -1.0, 1.0)
        self.target, self.last_a = C.integrate_target(self.target, a, self.lo, self.hi), a
        cmd = JointState(name=C.ARM_JOINTS + ["panda_finger_joint1"], position=[*map(float, self.target), C.GRIPPER_CLOSED])
        cmd.header.stamp = m.header.stamp                  # echo → the simulator measures the full round trip
        self.pub.publish(cmd)

rclpy.init(); rclpy.spin(PolicyNode(sys.argv[1], sys.argv[2]))
```

Run the policy node in a ROS 2-sourced shell: `python3 bridge/policy_node.py <run>/exported/policy.pt <run>/contract_runtime.json`. It needs a venv created with `--system-site-packages` (to see `rclpy`) and CPU PyTorch. Then run the simulator with Isaac Sim's Python (`python.sh` in a workstation install, which auto-selects the bundled Jazzy libraries, or `python` in the pip environment): `sim_bridge.py --contract <run>/contract_runtime.json --headless`, first with `--lockstep`, then without. The two modes answer different questions:

* **Lockstep** waits for the command that answers each observation, which removes timing from the experiment. Its success rate against the Lab nominal eval measures **contract and physics differences only**. If its CI does not overlap Lab's, stop and debug the contract (joint names, frames, quaternions, gains, friction) before measuring anything else.
* **Free-running** (the default) paces physics to the wall clock and applies whatever command has arrived. This is how a real robot behaves. Comparing it to lockstep measures the **cost of real latency**, which you trained for with the action-delay randomization. An achieved RTF below 1, or a non-zero `late_frac`, means the simulator cannot keep up with real time at this load.

Two known simplifications. The goal, cube and joint topics are not ordered relative to each other; a production bridge would carry one observation message per tick. And `/clock` is not published. Add Lecture 08's `ROS2PublishClock` graph if other ROS tools need `use_sim_time`. The latency here is measured on the system clock on purpose: it is wall-clock round trip.

**Optional camera variant.** Replace the cube position in the observation with a small RGB or depth camera (`CameraCfg`; in 3.0 `TiledCamera` is a deprecated alias because `Camera` includes the tiled path). Budget it with Lecture 07's per-render-product cost before training, and expect to need a CNN encoder and many more samples.

---

## 8. Use it in the real stack

* **Real robot:** the bridge scene is where `franka_ros`/`ros2_control` would go ([Lecture 08](Lecture-08.md), [Advanced ROS](../Advanced%20Robot%20Operating%20System/Lecture-01.md)). The contract file, latency budget and delay randomization carry over unchanged. Friction, mass and actuator dynamics are what sysid ([Lecture 12](Lecture-12.md) §6.2) should pin down first.
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
| **Bridge success: lockstep / free-running** | `sim_bridge.py` | Contract gap / latency cost |
| **Achieved RTF, `late_frac` (free-running)** | `n_steps · dt / wall`; overrun steps | ≈ 1 and ≈ 0: keeps up with real time |
| **E2E latency p50 / p99; inference ms** | Echoed stamps; node timer | p99 well below the 16.7 ms policy period |

---

## 10. Ship it

The capstone deliverables are the README's *What you ship*, completed:

* **`isaac_bench/`** with results for every lab, including `eval_policy.py` and the Lecture 12 breakdown
* **USD assets** from Lectures 02 and 06 (your imported, validated robot and work cell). Stretch goal: swap them into the capstone scene and requalify the policy.
* **`capstone/capstone_tasks/push/`**: task, contract, agent config, trained checkpoints and exported `policy.pt` for 3 seeds, learning curves (`results/curves.png`), eval table with CIs, and a video
* **`capstone/bridge/`**: both scripts and `results/bridge.jsonl` with lockstep and free-running runs
* **`PERFORMANCE.md`** and `RUN_MANIFEST.json`:

```markdown
# PERFORMANCE — Capstone push (GPU, driver, Isaac Sim 6.1.0, Isaac Lab <SHA>, PhysX, dt 1/120, decimation 2)
## VRAM budget (8 GB): fixed cost | per env | PyTorch | chosen num_envs | headroom
## Step-time breakdown: physics | env Python | policy | PPO update | other   (% of iteration)
## Training: wall-clock to target success, seeds 1-3
## Evaluation: nominal / shifted, k/n and Wilson 95% CI, per seed; robustness drop
## Bridge: lockstep vs free-running success (CI), RTF, late_frac, latency p50/p99, inference ms
## On 16 GB / 48 GB: predicted num_envs, camera variant, bridge-while-training, VLA server — and why
```

---

## Exit criteria

You are done when you can:

* hand the repo to another engineer who reproduces one training curve and one eval number from the manifest
* explain each step function of `PushEnv` and justify each reward term and randomization range
* state the policy contract from memory and show where each item is checked
* show that lockstep bridge success matches Lab evaluation within confidence intervals, or explain the gap
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
* Isaac Lab 3.0 — [Run benchmarks](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/developer-tools/benchmarking/run_benchmarks.html) · [Migration guide 2.x → 3.0](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/migration/migrating_to_isaaclab_3-0.html)
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
