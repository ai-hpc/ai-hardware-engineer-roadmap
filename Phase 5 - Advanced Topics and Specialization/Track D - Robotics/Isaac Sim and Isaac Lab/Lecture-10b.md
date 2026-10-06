# Lecture 10b: Worked Example: Spot Waypoint Navigation

## Overview

This lecture builds one complete robot behavior end to end: **a Boston Dynamics Spot quadruped that walks through a list of waypoints**, using a locomotion policy trained with RL in Isaac Lab and deployed in Isaac Sim 6.1. It is a worked example, so it uses nearly every earlier lecture: USD assets ([02](Lecture-02.md)), the experimental API and physics callbacks ([03](Lecture-03.md)), articulations and actuators ([04](Lecture-04.md)), PhysX vs Newton ([05](Lecture-05.md)), ROS 2 ([08](Lecture-08.md)), and Isaac Lab training ([09](Lecture-09.md), [10](Lecture-10.md)). The locomotion policy is trained by the RSL-RL PPO dissected in [Deep RL Lecture 06b](../Deep%20RL%20for%20Robot%20Learning/Lecture-06b.md).

The design pattern is the one real legged robots use: **hierarchical control**. A slow, simple navigation layer decides *where* to go and outputs a velocity command. A fast learned policy decides *how* to move the legs to follow that command. A PD actuator layer turns joint targets into torques, and physics does the rest. Each layer runs at its own rate and can be tested on its own.

By the end you should be able to:

* draw Spot's control stack with the rate of every layer, and say which layer owns which failure
* read a locomotion policy's contract: its 48-dimensional observation, 12-dimensional action, action scale, and training command ranges
* deploy a trained policy in Isaac Sim with `RobotPolicyRunner`, on PhysX or Newton
* write a waypoint-following controller that keeps its commands inside the policy's training distribution
* measure tracking error, completion time, falls, and real-time factor, and attribute each failure to a layer
* train your own Spot policy in Isaac Lab and deploy it with the same navigator

---

## 1. Why it matters: the canonical sim-to-real success

Legged locomotion is where simulation-trained RL has most clearly paid off on real hardware. Policies trained in massively parallel simulation now walk real quadrupeds over terrain that model-based controllers took years to handle. Spot is a useful case study for three reasons:

* **The full path exists in public.** Isaac Sim 6.1 ships pretrained Spot flat-terrain policies, one per physics engine. Isaac Lab 3.0 contains the task that trains them (`IsaacContrib-Velocity-Flat-Spot`). Boston Dynamics, NVIDIA and the RAI Institute offer an **RL Researcher Kit**: a Spot variant with a joint-level control API, mounting for a Jetson AGX Orin payload that runs the policy, and the Isaac Lab environment. That closes the loop from training to a real robot.
* **It forces you to respect a policy's contract.** A learned controller works only inside the conditions it was trained on: its command ranges, observation noise, control rate, and actuator model. Waypoint navigation tests that contract continuously, because the navigator generates every command the policy sees.
* **It separates concerns cleanly.** When Spot misses a waypoint, the cause is in one of three places: the navigator asked for something unreasonable, the policy failed to track a reasonable command, or the physics differs from training. The labs teach you to tell which.

---

## 2. Mental model: four layers, four rates

```text
 waypoints [(x, y), ...]
        │
 ┌──────▼──────────────────┐   50 Hz (once per rendered frame here; a real robot could run it slower)
 │ Navigation layer        │   pose (x, y, yaw) → velocity command [vx, vy, wz], body frame
 │ waypoint follower       │   clamped to the policy's training ranges, rate-limited
 └──────┬──────────────────┘
        │ [vx, vy, wz]
 ┌──────▼──────────────────┐   50 Hz (decimation 10)
 │ Locomotion policy π(o)  │   48-dim observation → 12 joint-position offsets
 │ MLP, trained with PPO   │   target q* = q_default + 0.2 · action
 └──────┬──────────────────┘
        │ q* held for 10 physics steps
 ┌──────▼──────────────────┐   500 Hz
 │ Actuators (PD)          │   hips: delayed PD (kp 60, kd 1.5, 45 N·m, 0–8 ms delay)
 │                         │   knees: remotized PD, torque limit from a joint-angle lookup table
 └──────┬──────────────────┘
        │ τ
 ┌──────▼──────────────────┐   500 Hz (dt = 0.002 s)
 │ Physics (PhysX/Newton)  │   M(q)q̈ + C q̇ + g = τ + Jᵀ F_contact   (Lecture 04 §2.7)
 └─────────────────────────┘
```

The rates come from the Isaac Lab task config, `sim.dt = 0.002` (500 Hz physics) and `decimation = 10` (a 50 Hz policy), and the deployed policy keeps them. `RobotPolicyRunner` reads them from the exported `env.yaml`. It runs inference once every `decimation` physics ticks and **holds** the joint targets in between, so the actuators see the same 50 Hz target updates as in training.

Spot is a **floating-base** robot. Its base has six unactuated degrees of freedom, so it can only move by pushing on the ground through contacts. That is why locomotion is learned rather than scripted: the policy has to discover a gait that produces the right contact forces at the right times.

---

## 3. The locomotion policy's contract

A deployed policy is a function with a strict interface. Isaac Sim's policy runtime reads that interface from two files exported by the Isaac Lab training run. The **IO descriptor** gives the ordered observation and action terms, shapes, joint order, scales and offsets. The **env config** gives physics timing, joint gains and limits, actuator models, and the initial state. For the bundled Spot policy, the observation is 48 numbers:

| Term | Dim | Training noise (uniform ±) | Notes |
|---|---|---|---|
| Base linear velocity (body frame) | 3 | 0.1 | On a real robot this comes from a state estimator, not ground truth |
| Base angular velocity (body frame) | 3 | 0.1 | From the IMU |
| Projected gravity | 3 | 0.05 | Gravity direction in the body frame: encodes roll and pitch |
| Velocity command \([v_x, v_y, \omega_z]\) | 3 | — | **The navigator's output** |
| Joint positions relative to default | 12 | 0.05 | |
| Joint velocities | 12 | 0.5 | |
| Previous action | 12 | — | Lets the policy smooth its own output |

The action is 12 joint-position offsets. The target is \(q^* = q_\text{default} + 0.2\,a\) (`JointPositionActionCfg(scale=0.2, use_default_offset=True)`).

**The command ranges are part of the contract.** During training, Isaac Lab samples velocity commands uniformly from

$$
v_x \in [-2.0,\ 3.0]\ \text{m/s}, \qquad v_y \in [-1.5,\ 1.5]\ \text{m/s}, \qquad \omega_z \in [-2.0,\ 2.0]\ \text{rad/s}
$$

(the `IsaacContrib-Velocity-Flat-Spot` config in Isaac Lab 3.0). A command outside these ranges is **out of distribution**, and the policy's behavior there is undefined. That is [Deep RL Lecture 02](../Deep%20RL%20for%20Robot%20Learning/Lecture-02.md)'s distribution shift showing up as a controller limit. The bundled policy's own training ranges are recorded in its hosted `physx_env.yaml`, so read them before trusting the numbers above. A good navigator stays well inside the ranges, not at their edges, because the policy tracks extreme commands less accurately.

**Quaternion trap.** `articulation.get_world_poses()` returns orientations as **wxyz** (Isaac Sim's experimental API). The hosted Spot env configs mix conventions, which is why the bundled spec pins the spawn orientation to identity. Isaac Lab 3.0 itself uses **xyzw**. Any code that moves orientations between the two must convert explicitly.

---

## 4. How the policy was trained

The Isaac Lab 3.0 task `IsaacContrib-Velocity-Flat-Spot` (a contrib task; its physics preset defaults to PhysX) is a manager-based environment ([Lecture 09](Lecture-09.md)). Its reward is mostly about **gait quality**, not only speed:

| Reward term | Weight | What it shapes |
|---|---|---|
| Gait (diagonal feet in sync, the two pairs out of sync) | +10.0 | A trotting gait |
| Foot air time | +5.0 | Lifting feet instead of shuffling |
| Linear-velocity tracking | +5.0 | Following \(v_x, v_y\) |
| Angular-velocity tracking | +5.0 | Following \(\omega_z\) |
| Foot clearance | +0.5 | Swing height |
| Base orientation / base motion | −3.0 / −2.0 | A level, steady body |
| Action smoothness, air-time variance | −1.0 each | Smooth, regular stepping |
| Joint position (with a stand-still scale) | −0.7 | Staying near the nominal pose, especially at zero command |
| Foot slip, joint velocity, torque, acceleration | −0.5, −1e-2, −5e-4, −1e-4 | Efficiency and wear |

The other pieces of the task:
* **Terminations:** a time-out, and leaving the terrain bounds, are both flagged `time_out=True`, so they are bootstrapped ([Deep RL 06b §4.1](../Deep%20RL%20for%20Robot%20Learning/Lecture-06b.md)). Body or leg contact with the ground is a real failure, so it terminates without bootstrapping.
* **Randomization ([Lecture 12](Lecture-12.md)):** friction and base mass at startup, random pushes during episodes, and a cobblestone-road terrain generator, even in the "flat" task.
* **PPO (RSL-RL):** 24 steps per environment, 5 epochs × 4 minibatches, an adaptive learning rate with KL target 0.01, clipped value loss with coefficient 0.5, entropy 0.0025, [512, 256, 128] ELU networks, and up to 20,000 iterations.

Notice how much of the sim-to-real work is in the **actuator model**, not the policy. The hip drives are `DelayedPDActuatorCfg` with a random 0-4 physics-step delay (0-8 ms), so the policy learns to tolerate actuation latency. The knees are `RemotizedPDActuatorCfg`. Spot's knee motor drives the joint through a linkage, so the available torque depends on the knee angle, and the config encodes that with a measured lookup table. Training against an idealized knee would produce a policy that asks the real robot for torque it cannot deliver.

---

## 5. The navigation layer

Given Spot's pose \((x, y, \psi)\) and the current waypoint \(g = (g_x, g_y)\), express the error in the body frame:

$$
\begin{bmatrix} e_x \\ e_y \end{bmatrix} =
\begin{bmatrix} \cos\psi & \sin\psi \\ -\sin\psi & \cos\psi \end{bmatrix}
\begin{bmatrix} g_x - x \\ g_y - y \end{bmatrix},
\qquad
\alpha = \operatorname{wrap}\big(\operatorname{atan2}(g_y - y,\ g_x - x) - \psi\big)
$$

Then a proportional controller that turns toward the goal and walks forward, slowing down when it faces away:

$$
v_x = \operatorname{clip}\big(k_v e_x,\ 0,\ v_x^{\max}\big)\cdot \max(\cos\alpha, 0), \quad
v_y = \operatorname{clip}\big(k_y e_y,\ \pm v_y^{\max}\big), \quad
\omega_z = \operatorname{clip}\big(k_\psi \alpha,\ \pm \omega_z^{\max}\big)
$$

Four design rules turn this into something a learned policy can follow:

1. **Stay inside the training ranges, with margin.** Use, for example, \(v_x^{\max} = 1.2\), \(v_y^{\max} = 0.4\), \(\omega_z^{\max} = 1.0\): well inside \([-2, 3] \times [-1.5, 1.5] \times [-2, 2]\). Lab 10b-c measures what happens at and beyond the edges.
2. **Prefer walking forward to strafing.** Spot can walk sideways, but most of the reward weight and command coverage is in forward walking. Use \(v_y\) only for small corrections.
3. **Rate-limit the command.** Training resamples commands abruptly, so steps are not out of distribution, but a rate limit (say 2 m/s² and 4 rad/s²) removes jerky transitions at waypoint switches and lowers the fall risk.
4. **Switch waypoints at a radius, not at a point.** A 0.3 m reach radius avoids orbiting a waypoint the policy cannot hit exactly.

This is a reactive follower with no obstacle avoidance. For real environments you would replace it with a planner, such as Nav2 over ROS 2 in Lab 10b-e. The interface to the policy stays the same: a body-frame velocity command.

---

## 6. The hardware view

* **The policy is tiny.** The Isaac Lab task's actor, a 48 → 512 → 256 → 128 → 12 MLP, has about 190 k parameters, roughly 0.4 MFLOP per inference, at 50 Hz. It costs nothing on any GPU or on a Jetson. On a real robot, what matters is **latency and jitter**: the policy was trained with 0-8 ms of actuator delay, so a deployment that adds tens of milliseconds of jitter is out of distribution in time. This is why the RL Researcher Kit runs the policy onboard, on a Jetson AGX Orin payload, rather than over a network.
* **Physics runs at 500 Hz.** One Spot means 500 articulation steps per simulated second. For one robot that is cheap. For RL, Isaac Lab runs thousands of them in parallel, which is the whole point of [Lecture 10](Lecture-10.md).
* **Rendering is the cost in this lab.** With the GUI, each frame (every 10 physics steps) renders the viewport, the path visualizations, and the robot. Headless, the same navigation run is mostly physics and Python. Measure the real-time factor both ways (Lab 10b-a).
* **Engines are not interchangeable at the policy level.** Isaac Sim ships **separate** PhysX and Newton Spot policies, each trained against its own engine. A policy trained on one engine and run on the other sees contact dynamics it never trained on ([Lecture 05](Lecture-05.md)). Lab 10b-c measures that gap.

> **8 GB budget.** One Spot on the default grid environment runs comfortably on a 4060, headless or in the GUI. The navigation lab is not GPU-bound. Training your own policy (Lab 10b-d) is where 8 GB matters. The task's PhysX config raises GPU contact-patch buffers, and thousands of Spots on generated terrain take real memory. Start at `--num_envs 512`-`1024` with `--viz none`, watch `nvidia-smi`, and expect longer wall-clock than on a 16-48 GB card. If training is too slow, deploy the bundled policy and treat Lab 10b-d as a cloud exercise.

---

## 7. Build it

Everything goes in `spot_nav/`. The labs use Isaac Sim 6.1 (Labs a-c, e) and Isaac Lab 3.0 (Lab d).

### Lab 10b-a — Run the official Spot example

Isaac Sim ships `standalone_examples/api/isaacsim.robot.policy.examples/spot_standalone.py`. It spawns Spot through `RobotPolicyRunner(get_spot_spec(), ...)`, drives it with a scripted sequence of velocity commands, and draws the commanded path (green) and the path Spot actually walked (red).

```bash
python standalone_examples/api/isaacsim.robot.policy.examples/spot_standalone.py --engine physx
python standalone_examples/api/isaacsim.robot.policy.examples/spot_standalone.py --engine newton
python standalone_examples/api/isaacsim.robot.policy.examples/spot_standalone.py --engine physx --test   # prints the tracking gap after one lap
```

Read the script before running it. Note the lifecycle: spawn while stopped; then in the post-physics-step callback, call `restart_from_default_state(command)` on the first tick and `step(dt, command)` on every tick after. Note also the timing: `PolicyEnvConfig.from_file(...).timing` provides `physics_dt`, `decimation` and `render_interval`. Record the `--test` tracking gap for both engines. Time 1,000 frames with and without the GUI to get the real-time factor.

### Lab 10b-b — A waypoint navigator

This script keeps the official example's setup and replaces its scripted command phases with the controller from Section 5. It is written against the 6.1 example's API and has not been run on a GPU here, so expect small fixes.

```python
# spot_nav/waypoint_nav.py
import argparse, csv, math
from isaacsim import SimulationApp

p = argparse.ArgumentParser()
p.add_argument("--engine", choices=["physx", "newton"], default="physx")
p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
p.add_argument("--course", choices=["square", "slalom"], default="square")
p.add_argument("--vx_max", type=float, default=1.2)
p.add_argument("--vy_max", type=float, default=0.4)
p.add_argument("--wz_max", type=float, default=1.0)
p.add_argument("--max_time", type=float, default=120.0)       # simulated seconds
p.add_argument("--headless", action="store_true")
p.add_argument("--log", default="spot_nav/run.csv")
args, _ = p.parse_known_args()
extra = [f"--/exts/isaacsim.core.simulation_manager/default_engine={args.engine}"]
if args.engine == "newton":
    extra += ["--enable", "isaacsim.physics.newton", "--enable", "isaacsim.physics.newton.tensors"]
simulation_app = SimulationApp({"headless": args.headless, "extra_args": extra})

import numpy as np, omni.timeline
from isaacsim.core.experimental.utils.stage import define_prim, set_stage_units, set_stage_up_axis
from isaacsim.core.rendering_manager import RenderingManager
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.simulation_manager.impl.isaac_events import IsaacEvents   # as in the 6.1 example
from isaacsim.robot.policy.examples import PolicyEnvConfig, RobotPolicyRunner, get_spot_spec
from isaacsim.storage.native import get_assets_root_path

COURSES = {"square": [(3, 0), (3, 3), (0, 3), (0, 0)],
           "slalom": [(2, 1), (4, -1), (6, 1), (8, -1), (8, 1), (0, 0)]}
TRAIN_RANGES = {"vx": (-2.0, 3.0), "vy": (-1.5, 1.5), "wz": (-2.0, 2.0)}  # Isaac Lab 3.0 Spot task cfg
assert 0 < args.vx_max <= TRAIN_RANGES["vx"][1] and args.vy_max <= TRAIN_RANGES["vy"][1] \
    and args.wz_max <= TRAIN_RANGES["wz"][1], "command limits outside the policy's training ranges"

def wrap(a): return math.atan2(math.sin(a), math.cos(a))

def pose_from(articulation):
    pos, quat = articulation.get_world_poses()                    # Warp arrays; quaternion is wxyz
    x, y, z = pos.numpy().reshape(-1)[:3]
    w, qx, qy, qz = quat.numpy().reshape(-1)[:4]
    yaw = math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    roll = math.atan2(2 * (w * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * qy - qz * qx))))
    return float(x), float(y), float(z), yaw, roll, pitch

class WaypointFollower:
    def __init__(self, waypoints, dt, k_v=1.0, k_y=1.0, k_w=2.0, reach=0.3, acc=2.0, alpha_acc=4.0):
        self.wps, self.i, self.dt = waypoints, 0, dt
        self.k_v, self.k_y, self.k_w, self.reach = k_v, k_y, k_w, reach
        self.max_delta = np.array([acc * dt, acc * dt, alpha_acc * dt])
        self.cmd = np.zeros(3, dtype=np.float32)
        self.prev_wp = (0.0, 0.0)

    @property
    def done(self): return self.i >= len(self.wps)

    def cross_track(self, x, y):                                   # distance to the current segment
        (ax, ay), (bx, by) = self.prev_wp, self.wps[min(self.i, len(self.wps) - 1)]
        dx, dy = bx - ax, by - ay
        t = max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / max(dx * dx + dy * dy, 1e-9)))
        return math.hypot(x - (ax + t * dx), y - (ay + t * dy))

    def command(self, x, y, yaw):
        if self.done:
            target = np.zeros(3)
        else:
            gx, gy = self.wps[self.i]
            if math.hypot(gx - x, gy - y) < self.reach:              # switch at a radius
                self.prev_wp, self.i = (gx, gy), self.i + 1
                return self.command(x, y, yaw)
            ex = math.cos(yaw) * (gx - x) + math.sin(yaw) * (gy - y)
            ey = -math.sin(yaw) * (gx - x) + math.cos(yaw) * (gy - y)
            alpha = wrap(math.atan2(gy - y, gx - x) - yaw)
            vx = min(max(self.k_v * ex, 0.0), args.vx_max) * max(math.cos(alpha), 0.0)
            vy = max(-args.vy_max, min(args.vy_max, self.k_y * ey))
            wz = max(-args.wz_max, min(args.wz_max, self.k_w * alpha))
            target = np.array([vx, vy, wz])
        self.cmd = (self.cmd + np.clip(target - self.cmd, -self.max_delta, self.max_delta)).astype(np.float32)
        return self.cmd

# --- scene and robot, as in spot_standalone.py ---
set_stage_up_axis("Z"); set_stage_units(meters_per_unit=1.0)
define_prim("/World/Ground", "Xform").GetReferences().AddReference(
    get_assets_root_path() + "/Isaac/Environments/Grid/default_environment.usd")
define_prim("/World/PhysicsScene", "PhysicsScene")
SimulationManager.set_physics_sim_device(args.device)
spec = get_spot_spec()
spot = RobotPolicyRunner(spec, prim_path="/World/Spot")
spot.spawn()
timing = PolicyEnvConfig.from_file(spec.engines[args.engine].env_config_path).timing
frame_dt = timing.render_interval * timing.physics_dt              # one navigator update per frame
RenderingManager.set_dt(frame_dt)
SimulationManager.set_physics_dt(spot.physics_dt)

base_command = np.zeros(3, dtype=np.float32)
state = {"first": True}
def on_physics_step(step_size, context):
    if state["first"]:
        spot.restart_from_default_state(base_command); state["first"] = False
    else:
        spot.step(step_size, base_command)
cb = SimulationManager.register_callback(on_physics_step, IsaacEvents.POST_PHYSICS_STEP)

nav = WaypointFollower(COURSES[args.course], frame_dt)
omni.timeline.get_timeline_interface().play()
simulation_app.update()
rows, t, fell = [], 0.0, False
while simulation_app.is_running() and t < args.max_time and not nav.done:
    simulation_app.update()
    if not SimulationManager.is_simulating():
        continue
    x, y, z, yaw, roll, pitch = pose_from(spot.articulation)
    if z < 0.25 or abs(roll) > 1.0 or abs(pitch) > 1.0:            # spawn height is 0.5 m
        fell = True; break
    base_command[:] = nav.command(x, y, yaw)
    sat = (abs(base_command[0]) >= args.vx_max - 1e-3) or (abs(base_command[2]) >= args.wz_max - 1e-3)
    rows.append([round(t, 3), x, y, yaw, *base_command.tolist(), nav.cross_track(x, y), int(sat), nav.i])
    t += frame_dt

with open(args.log, "w", newline="") as f:
    csv.writer(f).writerows([["t", "x", "y", "yaw", "vx", "vy", "wz", "cross_track", "saturated", "wp"], *rows])
ct = np.array([r[7] for r in rows]) if rows else np.zeros(1)
print(f"RESULT engine={args.engine} course={args.course} done={nav.done} fell={fell} time={t:.1f}s "
      f"cross_track_mean={ct.mean():.3f} max={ct.max():.3f} saturated={np.mean([r[8] for r in rows]):.2f}")
omni.timeline.get_timeline_interface().stop()
SimulationManager.deregister_callback(cb); spot.close(); simulation_app.close()
```

**Navigator baseline, before Isaac Sim.** The `WaypointFollower` class needs nothing from Isaac Sim, so test it first on an ideal robot that applies each command perfectly (integrate \(v_x, v_y, \omega_z\) in the body frame at `dt = 0.02`). Run that way, the class above completes both courses at every limit setting in Lab 10b-c without exceeding its limits, and its peak cross-track error is about **0.3 m**: the corner-cutting set by the 0.3 m reach radius (about 0.5 m on the slalom at the slowest turn rate). That is your floor. In Isaac Sim, any cross-track error above it comes from the policy or the physics, not the navigator.

Run both courses on both engines, headless. Plot the path (`x`, `y`) over the waypoints, the three commands over time, and the cross-track error. Before tuning anything, check two things. Does the robot track the *commands* (compare commanded \(v_x\) with the finite-difference velocity from the log)? Does the navigator produce sensible commands? Those are different layers' failures.

### Lab 10b-c — Find the edges of the contract

Run each of these on the square course, 3 times each (the start state is deterministic, but GPU physics may not be bit-exact; [Lecture 12](Lecture-12.md)):

| Experiment | Change | Question |
|---|---|---|
| Speed sweep | `--vx_max` ∈ {0.5, 1.2, 2.0, 2.8} | Where does tracking error grow? Does completion time keep falling? |
| Turn-rate sweep | `--wz_max` ∈ {0.5, 1.0, 1.8} | When do waypoint switches cause overshoot or stumbles? |
| Beyond training | Remove the assert, set `--vx_max 3.5 --wz_max 2.5` | What does an out-of-distribution command do to the gait? |
| Engine swap | Run with `--engine newton`, but construct the runner with `RobotPolicyRunner(spec, prim_path=..., training_engine="physx")` | How large is the cross-engine gap compared with the matched policy? |

Write down which layer each failure belongs to. If tracking error grows with speed while the commands stay smooth, the policy is the limit. If the commands oscillate at waypoint switches, the navigator's gains or rate limits are wrong. If the engine swap fails but matched policies don't, the contact model is the limit.

### Lab 10b-d — Train your own Spot policy and deploy it

Train with Isaac Lab 3.0 ([Lecture 10](Lecture-10.md)). On 8 GB, start small:

```bash
uv run --extra isaacsim isaaclab train --rl_library rsl_rl --task IsaacContrib-Velocity-Flat-Spot \
    --viz none --num_envs 1024 --max_iterations 3000
uv run --extra isaacsim isaaclab play  --rl_library rsl_rl --task IsaacContrib-Velocity-Flat-Spot \
    --checkpoint latest --num_envs 16                  # also writes exported/policy.pt and policy.onnx
```

Watch `Train/mean_reward`, the velocity-tracking reward terms, and `Loss/learning_rate` ([Deep RL 06b §12](../Deep%20RL%20for%20Robot%20Learning/Lecture-06b.md)). Then deploy your run in place of the bundled policy:

```python
import hashlib
from isaacsim.robot.policy.examples import PolicyArtifact, get_spot_spec
run_dir = "logs/rsl_rl/spot_flat/<timestamp>"                       # your training run
sha = hashlib.sha256(open(f"{run_dir}/exported/policy.pt", "rb").read()).hexdigest()
artifact = PolicyArtifact.from_training_run(run_dir, model_sha256=sha)
spec = get_spot_spec(engines={"physx": artifact})                   # keeps the bundled Spot USD
```

`from_training_run` prefers the exported policy and fails loudly on missing or ambiguous files. If it reports a missing IO descriptor, follow the Isaac Sim 6.1 robot-policy migration guide's section on exporting `IO_descriptors.yaml` from an Isaac Lab run. The runner verifies the SHA-256 before every load, so in production pin the hash from a trusted manifest, not from the file you are about to load. Drive your policy with the Lab 10b-b navigator and compare its tracking with the bundled policy's.

### Lab 10b-e (optional) — Drive Spot from ROS 2

Replace the navigator with a ROS 2 subscriber: copy `/cmd_vel` (`geometry_msgs/Twist`: `linear.x`, `linear.y`, `angular.z`) into `base_command`, after clamping it to the training ranges. Publish `/clock`, odometry and TF with the OmniGraph nodes from [Lecture 08](Lecture-08.md). Any ROS 2 planner, including Nav2, can then drive Spot, and the policy's contract becomes a **safety filter** in your bridge: never forward a command outside the training ranges.

---

## 8. Use it in the real stack

* **Isaac Sim 6.1** — `isaacsim.robot.policy.examples` deploys Isaac Lab policies for Spot, ANYmal-C, Go2, H1, Franka and Cartpole through one `RobotPolicyRunner`. Extension 7.0 replaced the older per-robot classes; constructing `SpotFlatTerrainPolicy` now raises an error that names its replacement. **You will see this in older code**: tutorials from 4.x-5.x construct `SpotFlatTerrainPolicy(prim_path=..., ...)` directly and call `forward(dt, command)`.
* **Isaac Lab 3.0** — `IsaacContrib-Velocity-Flat-Spot` (contrib) with its own reward module (`contrib/velocity/config/spot/mdp/rewards.py`) and the `SPOT_CFG` asset with delayed and remotized PD actuators.
* **MobilityGen**, Isaac Sim's mobility data-generation workflow, uses the same Spot and H1 specs to drive robots while recording sensor data.
* **The real robot** — the Boston Dynamics **RL Researcher Kit** pairs a Spot variant that exposes joint-level control with an onboard Jetson AGX Orin and the Isaac Lab environment, so a policy trained as in Lab 10b-d can run on hardware. NVIDIA's technical blog walks through that sim-to-real workflow. Real deployment adds what simulation gives you for free: a state estimator for base velocity, sensor noise, and real actuator latency. These are exactly the terms the training noise and actuator delays were designed to cover.

---

## 9. Measure it

| Metric | How | Layer it diagnoses |
|---|---|---|
| Completion time per course | Lab 10b-b `RESULT` | Whole stack |
| Cross-track error (mean, max) | Distance to the active segment, logged per frame | Navigator + policy |
| Velocity-tracking error | Commanded vs finite-difference body velocity | Policy |
| Command saturation fraction | Frames at \(v_x^{\max}\) or \(\omega_z^{\max}\) | Navigator limits |
| Falls | Base height < 0.25 m or tilt > 1 rad | Policy, physics |
| Tracking gap, PhysX vs Newton | Lab 10b-a `--test`, Lab 10b-c | Engine / contact model |
| Real-time factor, GUI vs headless | Simulated time / wall-clock | Rendering cost |
| Training wall-clock and peak VRAM | Lab 10b-d, `nvidia-smi` | Hardware budget |

---

## 10. Ship it

Commit `spot_nav/` with:

* `waypoint_nav.py`, `run_sweeps.sh`, and the CSV logs
* `paths.png` (walked paths over waypoints for both courses and engines), `commands.png`, `tracking.png`
* `sweeps.csv` and `CONTRACT.md` — the speed and turn-rate limits you would ship, justified by your measurements, with the out-of-distribution results as evidence
* (optional) your Lab 10b-d training curves, and a comparison of your policy against the bundled one on the same course
* `LAYERS.md` — for every failure you saw, the layer responsible and the evidence

---

## Exit criteria

You can move on when you can:

* draw the four-layer stack with its rates (navigator, 50 Hz policy, 500 Hz PD actuators, 500 Hz physics) and say what each layer owns
* list the 48 observation terms and the action mapping, and explain why the command ranges are part of the policy's contract
* run the bundled Spot policy on PhysX and Newton and explain why each engine ships its own policy
* write a waypoint follower that keeps commands inside the training distribution, and show its tracking error and completion times
* attribute a navigation failure to the navigator, the policy, or the physics, with evidence from your logs
* train a Spot policy in Isaac Lab and deploy it with `PolicyArtifact.from_training_run`

---

## Self-check

1. Spot completes the square course at `--vx_max 1.2` with small cross-track error. At `--vx_max 2.8`, the commands in your log are smooth, but cross-track error triples and Spot stumbles at corners. Which layer is the limit, and which two measurements from Section 9 confirm it?
2. A teammate wants Spot to cross the course faster and sets `wz_max = 3.0` so it turns on the spot at each waypoint. The script's assert fires. Explain, in terms of the policy's training, why removing the assert is the wrong fix. What would you change instead?
3. Why does the runner hold the joint targets for 10 physics steps instead of running the policy at 500 Hz? What would change if you ran inference every physics step with the same network?
4. Your own policy from Lab 10b-d walks well in Isaac Lab but drifts sideways in Isaac Sim with the same commands. Name three contract items to compare between the two setups, and one way the quaternion conventions could cause exactly this symptom.
5. The bundled PhysX policy, run under Newton, falls within a few steps, while the Newton policy walks fine. What does this tell you about what the policy learned, and what does it imply for deploying on a real Spot?
6. On a real Spot, the policy runs on a Jetson AGX Orin even though a laptop GPU could run it faster. Using Section 6, explain why onboard inference matters more than raw speed for this policy.

---

## References

* Isaac Sim 6.1 — robot policy example extension: [docs](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/robot_simulation/ext_isaacsim_robot_policy_example.html), [migration guide (extension 7.0, exporting IO descriptors)](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/migration_guides/isaac_sim_6_1/robot_policy_examples.html)
* Isaac Sim 6.1 — `spot_standalone.py` source: [GitHub](https://github.com/isaac-sim/IsaacSim/blob/v6.1.0/source/standalone_examples/api/isaacsim.robot.policy.examples/spot_standalone.py)
* Isaac Lab 3.0 — Spot task config, `source/isaaclab_tasks/isaaclab_tasks/contrib/velocity/config/spot/`: [GitHub](https://github.com/isaac-sim/IsaacLab/tree/release/3.0.0/source/isaaclab_tasks/isaaclab_tasks/contrib/velocity/config/spot)
* NVIDIA Technical Blog, "Closing the Sim-to-Real Gap: Training Spot Quadruped Locomotion with NVIDIA Isaac Lab" — [blog](https://developer.nvidia.com/blog/closing-the-sim-to-real-gap-training-spot-quadruped-locomotion-with-nvidia-isaac-lab/)
* Boston Dynamics, RL Researcher Kit — [product page](https://bostondynamics.com/reinforcement-learning-researcher-kit/) · Spot SDK — [docs](https://dev.bostondynamics.com/)
* Rudin, Hoeller, Reist, Hutter, "Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning," 2021 — [arXiv:2109.11978](https://arxiv.org/abs/2109.11978)
* Hwangbo et al., "Learning Agile and Dynamic Motor Skills for Legged Robots," 2019 — [arXiv:1901.08652](https://arxiv.org/abs/1901.08652)
* Schwarke et al., "RSL-RL: A Learning Library for Robotics Research," 2025 — [arXiv:2509.10771](https://arxiv.org/abs/2509.10771)

---

## Next in this special course

* Next: [Lecture 11 — Imitation, Teleop, and Policy Evaluation](Lecture-11.md)
* Previous: [Lecture 10 — Training Policies in Isaac Lab](Lecture-10.md)
* Back: [Isaac Sim and Isaac Lab — Overview](README.md)
