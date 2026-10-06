# Lecture 04: PhysX for Robots

## Overview

Most "the robot exploded" or "the cube slips out of the gripper" bugs are not mysteries. They come from a handful of physics parameters that someone left at the default or set without knowing what they trade: collision approximations, contact and rest offsets, friction combine modes, the solver type, iteration counts, the timestep, joint drive gains, and drive force limits. Each one also has a cost in milliseconds per step and, on the GPU pipeline, in VRAM.

This lecture covers what PhysX 5 does in one step, which knobs matter for robots, and how to set the knobs from Python when the Core Experimental wrappers don't expose them. Then you tune two things by measurement: a box stack that has to stay standing, and a Franka gripper that has to hold a cube through a shake.

By the end you should be able to:

* explain mass, inertia, colliders, contact/rest offsets and friction combine modes, and say which symptom each one causes when wrong
* choose between TGS and PGS, and trade iterations against timestep with numbers rather than guesses
* say why CCD and GPU dynamics don't mix in Isaac Sim, and what that means for an RL pipeline
* configure articulation joint drives as PD controllers (stiffness, damping, max force, armature) and pick position, velocity or effort control
* tune a two-finger grasp until it holds, and justify every parameter you changed
* measure the CPU vs GPU dynamics crossover and the VRAM cost of PhysX's GPU buffers on your card

---

## 1. Why it matters: physics is where sim-to-real gaps start

Physics parameters decide whether a policy trained in simulation learns the task or learns an artifact of the simulator. Three examples you will meet in this course:

| Symptom | Usual cause | Cost of the naive fix |
|---|---|---|
| Stacked boxes jitter or slowly slide apart | Too few solver iterations, a timestep too large for the contact stiffness, offsets too small | Raising the physics rate multiplies the cost of *everything*, collision detection included |
| A gripper "holds" the cube but the fingers sink into it | Drive stiffness and solver order fight the contact constraints; the drive targets sit deep inside the object | Very stiff drives with large forces can make the whole arm unstable |
| An RL policy learns to vibrate a joint to slide objects | Low damping plus bang-bang actions exploit how the solver integrates drives | The policy then fails on the real robot, or on another physics engine ([Lecture 05](Lecture-05.md)) |

The rule for this lecture: **change one parameter at a time, against a fixed reproduction, and record both the behavior and the ms/step.** Isaac Lab's own PhysX tuning guide prescribes the same diagnose-first loop.

---

## 2. Mental model: one PhysX step

```text
 per physics step (dt = 1 / timeStepsPerSecond)
 ┌────────────┐   ┌─────────────┐   ┌──────────────────────────────┐   ┌───────────┐
 │ broadphase │──►│ narrowphase │──►│ solver: N position iters     │──►│ integrate │
 │ AABB pairs │   │ contacts    │   │ (+ M velocity iters)         │   │ poses     │
 │ (MBP/GPU)  │   │ within      │   │ contacts, joints, drives,    │   │ + sleep   │
 │            │   │ contactOff. │   │ limits — islands solved      │   │ checks    │
 └────────────┘   └─────────────┘   └──────────────────────────────┘   └───────────┘
```

### 2.1 Rigid bodies: mass, inertia, center of mass

A prim becomes dynamic with `UsdPhysics.RigidBodyAPI`. Its mass properties come from `UsdPhysics.MassAPI` if authored; otherwise PhysX derives them from the collision shapes and a density. Wrong inertia is the most common hidden bug in imported robots: a link with near-zero inertia next to a heavy one gives a large **mass ratio**, and impulses from the heavy body produce huge velocities in the light one. The Omni Physics stability guide says to avoid large mass and inertia ratios and to make sure every value is non-zero.

The experimental `RigidPrim` exposes `set_masses`, `set_inertias` (shape `(N, 9)`), `set_coms` (position plus a **wxyz** orientation), `set_densities`, and `set_sleep_thresholds`. Check the docstring for each method's backends before you use it: `set_inertias` and `set_coms` are tensor-only, so they work only while the simulation is playing, while `set_sleep_thresholds` is USD-only, so you set it before play.

### 2.2 Colliders: what actually collides

Collision geometry has nothing to do with what you see. `GeomPrim.set_collision_approximations()` offers these approximations for meshes:

| Approximation | What PhysX uses | Robot use | Cost |
|---|---|---|---|
| Primitives (`Cube`, `Sphere`, `Capsule`) | Analytic shape | Boxes on a table, fingertip pads | Cheapest |
| `boundingCube` / `boundingSphere` | One fitted primitive | Placeholders, coarse obstacles | Cheapest |
| `convexHull` | One convex mesh (the default for meshes) | Most links | Low |
| `convexDecomposition` | A set of convex hulls | Concave parts: hooks, mugs, gripper jaws | Grows with hull count |
| `sphereFill` | A set of spheres | Rounded parts; an alternative to convex decomposition | Grows with sphere count |
| `sdf` | Signed-distance field of a triangle mesh | Tight-tolerance insertion, nuts and bolts | High (memory and compute) |
| `none` / `meshSimplification` | Triangle mesh | **Static** geometry only | Not supported on dynamic bodies, where it falls back to a convex hull |

The Isaac Sim docs make two points worth memorizing. A torus with a convex-hull collider has no hole, so things rest on top of it. A triangle mesh on a rigid body silently becomes a convex hull unless you choose SDF.

### 2.3 Contact offset and rest offset

Two shapes generate contacts once their distance drops below the **sum of their contact offsets**. They come to rest at the **sum of their rest offsets**. The contact offset must be positive and larger than the rest offset (`GeomPrim.set_offsets(contact_offsets=..., rest_offsets=...)`). The trade-off:

* **Contact offset too small:** contacts appear too late. You get jitter, missed contacts, or tunneling for thin or fast objects.
* **Contact offset too large:** many speculative contacts are generated, which costs solver time and GPU contact-buffer space.
* **Rest offset:** lets the collision shape sit slightly inside (negative) or outside (positive) its geometry. Use it to make the visual mesh and the contact location agree.

### 2.4 Physics materials and combine modes

A rigid-body material sets static friction, dynamic friction and restitution (`RigidBodyMaterial(path, static_frictions=..., dynamic_frictions=..., restitutions=...)`), and you bind it with `GeomPrim.apply_physics_materials(material)`. A contact involves **two** materials, so each one also has a **combine mode** (`set_combine_modes(frictions=..., restitutions=...)`) with the values `average`, `min`, `multiply` and `max`. When the two sides disagree, the mode with higher priority wins, in the order **average < min < multiply < max**. In practice, setting `max` on the object you grasp lets its friction dominate whatever the finger material is, which is useful when you don't control the robot asset.

PhysX computes friction **per contact patch**, applying friction at up to two anchor points per patch. TGS applies friction in every position and velocity iteration; PGS by default applies it only in the last three position iterations. That is one reason grasps often behave better under TGS.

### 2.5 The solver: TGS vs PGS, iterations, timestep

| | PGS (Projected Gauss-Seidel) | TGS (Temporal Gauss-Seidel) |
|---|---|---|
| Idea | Solve constraint velocities for the full `dt`, then integrate | Split `dt` into equal substeps; **each position iteration solves one substep** |
| Strengths | Slightly cheaper per iteration; some stiff legacy assets prefer it | Better convergence, high mass ratios, joint-drive accuracy, high-frequency effects |
| Velocity iterations | PhysX SDK default is 4 position / 1 velocity | Often 0-1 suffice (velocity iterations only clean up the last substep) |
| Default | — | The Isaac Lab default (`solver_type=1`), and the one to start with for articulations |

Because TGS position iterations act like substeps of the solver only, raising them is often cheaper than raising the physics rate: collision detection still runs once per step. Iteration counts are set per actor (`physxRigidBody:solverPositionIterationCount`, or `Articulation.set_solver_iteration_counts()`). PhysX treats them as **lower bounds**: the final count depends on the solver island the actor ends up in.

The timestep is `1 / timeStepsPerSecond` on the physics scene (`PhysxScene.set_steps_per_second()` or `set_dt()`). The stability guide's key quantity is the product of a drive's (or contact's) natural frequency and the timestep:

$$
\omega_n = \sqrt{\frac{k_p}{I_{\text{eff}}}}, \qquad \zeta = \frac{k_d}{2\sqrt{k_p\, I_{\text{eff}}}}, \qquad \text{keep } \omega_n\,\Delta t \lesssim 1
$$

Here \( I_{\text{eff}} \) is the smallest relevant mass or inertia of the body pair the joint couples. If \( \omega_n \Delta t \gg 1 \), raise TGS position iterations or shrink `dt`. Both cost time.

### 2.6 CCD, stabilization, sleeping, determinism

* **CCD** (continuous collision detection) sweeps fast bodies between poses. It must be enabled on the **scene** (`PhysxScene.set_enabled_ccd(True)`) **and** on the **body** (`PhysxSchema.PhysxRigidBodyAPI` → `CreateEnableCCDAttr`). **CCD is not supported with GPU dynamics.** `PhysxScene.set_enabled_gpu_dynamics(True)` switches CCD off, and Isaac Lab forces it off with a warning. Choosing CCD therefore means choosing the CPU pipeline.
* **Stabilization** (`set_enabled_stabilization`) is an extra pass that helps at large `dt`. Isaac Lab recommends it only when `dt` is larger than about 1/30 s, and warns that it makes reported contact forces inaccurate. It also damps free spinning. Check its value with `get_enabled_stabilization()` instead of assuming a default.
* **Sleeping:** bodies that stay still stop being simulated. PhysX does *not* wake sleeping actors when you change gravity. Lab 4b toggles gravity on a cube, so it sets that cube's sleep threshold to 0.
* **Determinism:** PhysX gives identical results for the same scene, inserted in the same order, with the same timestep, release and platform. *Enhanced determinism* (`physxScene:enableEnhancedDeterminism`, which Isaac Lab sets for `deterministic=True`) also isolates islands from each other, at a performance cost. [Lecture 12](Lecture-12.md) covers this.

### 2.7 Articulations and joint drives

An **articulation** is a reduced-coordinate tree of links rooted at a prim with `UsdPhysics.ArticulationRootAPI`. It is simulated by joint coordinates rather than as free bodies held together by constraints, so the joints never drift apart. On every joint, `physics:body0` must be the **parent** and `physics:body1` the **child**. USD stores revolute angles and limits in **degrees**.

**What the solver is integrating.** In joint coordinates \(q\), the equations of motion of an articulated robot are

$$
M(q)\,\ddot q + C(q, \dot q)\,\dot q + g(q) = \tau + J(q)^\top F_{\text{ext}}
$$

with \(M(q)\) the joint-space mass matrix (link masses and inertias, plus armature on the diagonal), \(C(q,\dot q)\dot q\) the Coriolis and centrifugal terms, \(g(q)\) gravity, \(\tau\) the joint torques from the drives below, and \(J^\top F_{\text{ext}}\) the contact and external forces mapped through the contact Jacobian. You never write this down in Isaac Sim, but it explains most tuning behavior. The mass matrix is why a light wrist with a stiff drive oscillates and why armature fixes it. Contacts enter as forces, so contact solver iterations matter as much as drive gains. Every term depends on the USD masses, inertias and joint axes you imported, which is why [Lecture 06](Lecture-06.md) validates them before anything else. A floating-base robot, such as the quadruped in [Lecture 10b](Lecture-10b.md), adds six unactuated base coordinates to \(q\). It moves only by pushing on the ground through \(J^\top F_{\text{ext}}\).

A joint drive is an **implicit PD controller** whose force is clamped:

$$
\tau = \operatorname{clamp}\!\big(k_p\,(q^{*} - q) + k_d\,(\dot q^{*} - \dot q),\; -\tau_{\max},\; \tau_{\max}\big)
$$

| Control mode | Stiffness \( k_p \) | Damping \( k_d \) | Experimental API |
|---|---|---|---|
| Position | > 0 | ≥ 0 (low, to stop oscillation) | `set_dof_position_targets` |
| Velocity | 0 | > 0 | `set_dof_velocity_targets` |
| Effort (torque) | 0 | 0 | `set_dof_efforts` |

`Articulation.switch_dof_control_mode("position"|"velocity"|"effort")` applies exactly that table to the gains. The other drive knobs on `Articulation`: `set_dof_gains(stiffnesses, dampings)` for \( k_p, k_d \); `set_dof_max_efforts` for \( \tau_{\max} \) (use the actuator datasheet value, not "large"); `set_dof_max_velocities` (the stability guide suggests a generous clamp for early RL training); `set_dof_friction_properties`; `set_enabled_self_collisions`; and `set_dof_armatures`. **Armature** adds joint-space inertia, which for a geared motor is the reflected rotor inertia \( J G^2 \). It raises \( I_{\text{eff}} \), which lowers \( \omega_n \) and stabilizes light links with stiff drives.

An **acceleration drive** normalizes by joint inertia, which makes it easier to tune, but contacts aren't included in that inertia, so it can look weak during grasping.

> **You will see this in older code:** `SingleArticulation.get_articulation_controller().apply_action(ArticulationAction(...))` and `set_gains(kps, kds)` on 5.x `isaacsim.core.api` and `isaacsim.core.prims`. In 6.1 these are the batched `Articulation.set_dof_position_targets` / `set_dof_gains` shown above.

**Solving articulation contacts last (added in 5.1).** By default the articulation solver resolves constraints in a fixed order, and it favors whichever constraint type comes last. For grippers with persistent finger penetration, raise the scene attribute `physxScene:solveArticulationContactLast`. Dynamic contact is then solved near the end and the max joint velocity is enforced last, which produces stronger normal and friction forces. It costs measurable performance. Isaac Lab exposes it as `PhysxCfg(solve_articulation_contact_last=True)` and advises keeping it only if it measurably helps.

**Gain Tuner** (`isaacsim.robot_setup.gain_tuner`, under *Tools > Robotics > Asset Editors > Gain Tuner*) edits per-joint gains and runs Snap-to-Limits, Sinusoidal, Step, Stress and **dt Sweep** tests. Under PhysX it displays angular gains in **degree** units. [Lecture 06](Lecture-06.md) uses it on an imported robot.

**Surface grippers** (suction, `isaacsim.robot.surface_gripper`) work by creating D6 joints at contact points. The Isaac Lab 2.3 release notes say they are **CPU-only**, so they do not fit a GPU-pipeline RL setup.

---

## 3. Setting what the wrappers don't expose

The Core Experimental API covers most robot-facing parameters. Below that, you set USD schema attributes directly, which is the pattern Lecture 01 §2.2 introduced:

| Parameter | Wrapper (6.1) | Raw schema |
|---|---|---|
| Timestep, solver, GPU dynamics, CCD, broadphase, stabilization | `PhysxScene` (`set_steps_per_second`, `set_solver_type`, `set_enabled_gpu_dynamics`, `set_enabled_ccd`, `set_broadphase_type`, `set_enabled_stabilization`) | `PhysxSchema.PhysxSceneAPI` |
| GPU buffer capacities | `PhysxScene.set_gpu_configuration(PhysxGpuCfg(...))` | `physxScene:gpu*` attributes |
| Contact-last ordering | — | `physxScene:solveArticulationContactLast` (Bool) |
| Body iteration counts, body CCD | — | `PhysxSchema.PhysxRigidBodyAPI` (`CreateSolverPositionIterationCountAttr`, `CreateEnableCCDAttr`) |
| Articulation iterations | `Articulation.set_solver_iteration_counts` | `physxArticulation:solverPositionIterationCount` |
| Offsets, approximation, materials | `GeomPrim.set_offsets`, `set_collision_approximations`, `apply_physics_materials` | `PhysxSchema.PhysxCollisionAPI`, `UsdPhysics.MaterialAPI` |
| Mass, CoM, inertia | `RigidPrim.set_masses` / `set_coms` / `set_inertias` | `UsdPhysics.MassAPI` |

`SimulationManager.set_physics_dt`, `enable_ccd`, `enable_gpu_dynamics` and `set_solver_type` still exist in 6.1 but are **deprecated** in favor of the `PhysxScene` methods. Most USD-backed setters only take effect if you call them **before** `play()`.

---

## 4. The hardware view: what each knob costs

**CPU vs GPU pipeline.** `SimulationManager.set_device("cuda:0")` switches the PhysX scenes to the GPU broadphase and GPU dynamics, enables Fabric, and suppresses readback, so state stays on the GPU and the tensor API hands it to torch without a copy. `set_device("cpu")` uses MBP broadphase and CPU dynamics on PhysX worker threads. The GPU pipeline has a fixed cost per step (kernel launches, synchronization) that small scenes can't amortize, while the CPU pipeline scales with cores and bodies. So there is a **crossover body count**, specific to your CPU, GPU and scene, below which the CPU is faster. Lab 4c measures it.

**Iterations vs timestep.** Halving `dt` doubles the cost of every stage per simulated second, collision detection included. Doubling TGS position iterations roughly doubles only the solver stage, and velocity iterations are cheaper still and often unnecessary with TGS. Plot ms/step against each knob, together with the stability metric, and choose on the Pareto front.

**Contacts drive cost.** Solver work scales with contact constraints, not bodies. A dense stack, a resting grasp, or an over-generous contact offset multiplies the constraint count. Hull count from convex decomposition and SDF resolution multiply narrowphase work.

**GPU buffers are fixed VRAM.** PhysX allocates the GPU contact, patch, pair and heap buffers at scene creation; read them with `PhysxScene.get_gpu_configuration()`. Most capacities do **not** grow (the heap is the exception). If you exceed one, PhysX drops contacts with a `[PhysX]` warning, so treat that warning as a hard failure. Raising a capacity raises the **fixed** VRAM cost even when the scene is small, so size the buffers to the busiest reset state, plus headroom.

> **8 GB budget.** Physics is rarely what fills a 4060. The Kit/RTX baseline and render products are ([Lecture 07](Lecture-07.md)). The physics risks on 8 GB are over-provisioned GPU buffers "to be safe" (pure fixed cost), SDF colliders on many objects, and convex decompositions with dozens of hulls per link. Keep primitives on objects you manipulate, size buffers from a measured busiest state, and run tuning sweeps headless. CCD forces the CPU pipeline, which frees VRAM but rules out GPU RL. If a workload needs SDF contact for many parts, *and* cameras, *and* thousands of environments at once, move it to a cloud L40S or RTX PRO 6000 instead of starving the renderer.

---

## 5. Build it

Labs go in `physics_lab/`. Every script follows Lecture 01's standalone pattern and prints one CSV row with Lecture 01's metric names.

### Lab 4a — Box-stack parameter sweep

```python
# physics_lab/box_stack.py — S stacks of H boxes; prints stability + cost as one CSV row
import argparse, time
p = argparse.ArgumentParser()
p.add_argument("--stacks", type=int, default=1)
p.add_argument("--height", type=int, default=10)
p.add_argument("--hz", type=int, default=60)                  # physics steps per second
p.add_argument("--solver", choices=["TGS", "PGS"], default="TGS")
p.add_argument("--pos-iters", type=int, default=4)
p.add_argument("--vel-iters", type=int, default=1)
p.add_argument("--contact-offset", type=float, default=None)  # meters; None keeps the default
p.add_argument("--rest-offset", type=float, default=None)
p.add_argument("--device", default="cuda:0")                  # "cpu" or "cuda:0"
p.add_argument("--buf-mult", type=int, default=1)             # scale GPU contact buffers
p.add_argument("--steps", type=int, default=600)
p.add_argument("--headless", action="store_true")
args = p.parse_args()

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": args.headless})

import numpy as np, omni.timeline
from pxr import PhysxSchema
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.experimental.objects import Cube, GroundPlane
from isaacsim.core.experimental.prims import GeomPrim, RigidPrim
from isaacsim.core.simulation_manager import PhysxGpuCfg, PhysxScene, SimulationManager

stage_utils.create_new_stage()
scene = PhysxScene("/World/PhysicsScene")
scene.set_steps_per_second(args.hz)
scene.set_solver_type(args.solver)
GroundPlane("/World/ground")

H, GAP = 0.05, 0.001                                   # 10 cm boxes, spawned 1 mm apart (no initial overlap)
side = int(np.ceil(np.sqrt(args.stacks)))
s, k = [a.ravel() for a in np.meshgrid(np.arange(args.stacks), np.arange(args.height), indexing="ij")]
pos = np.stack([(s % side) * 0.5, (s // side) * 0.5, H + k * (2 * H + GAP)], axis=1)
n = len(pos)

shape = Cube(paths=[f"/World/box_{i}" for i in range(n)], positions=pos, sizes=[2 * H] * n)
geoms = GeomPrim(paths=shape.paths, apply_collision_apis=True)
if args.contact_offset is not None or args.rest_offset is not None:
    geoms.set_offsets(contact_offsets=args.contact_offset, rest_offsets=args.rest_offset)
boxes = RigidPrim(paths="/World/box_.*", masses=0.5)
for prim in boxes.prims:                               # not exposed by RigidPrim: set the PhysX schema directly
    api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    api.CreateSolverPositionIterationCountAttr().Set(args.pos_iters)
    api.CreateSolverVelocityIterationCountAttr().Set(args.vel_iters)

gpu = args.device.startswith("cuda")
SimulationManager.set_device(args.device)              # carb settings (CUDA device, readback, Fabric)
scene.set_enabled_gpu_dynamics(gpu)                     # and make sure *this* scene follows
scene.set_broadphase_type("GPU" if gpu else "MBP")
if gpu and args.buf_mult > 1:
    c = scene.get_gpu_configuration()
    scene.set_gpu_configuration(PhysxGpuCfg(
        gpu_max_rigid_contact_count=c.gpu_max_rigid_contact_count * args.buf_mult,
        gpu_max_rigid_patch_count=c.gpu_max_rigid_patch_count * args.buf_mult))

omni.timeline.get_timeline_interface().play()
simulation_app.update()
p0 = boxes.get_world_poses()[0].numpy()

t = time.perf_counter()
SimulationManager.step(steps=args.steps)
p1 = boxes.get_world_poses()[0].numpy()                 # the read also forces a sync with the GPU
ms = (time.perf_counter() - t) / args.steps * 1e3

lin, _ = boxes.get_velocities()
drift = np.linalg.norm(p1[:, :2] - p0[:, :2], axis=1).max()
fallen = int((p0[:, 2] - p1[:, 2] > H).sum())
sink = H - p1[k == 0, 2].min()                          # > 0: bottom boxes sit inside the ground
rtf = (1.0 / args.hz) / (ms / 1e3)
print(f"{n},{args.hz},{args.solver},{args.pos_iters},{args.vel_iters},{args.contact_offset},{args.device},"
      f"{ms:.3f},{rtf:.1f},{drift:.4f},{fallen},{sink:.4f},{np.abs(lin.numpy()).max():.4f}")
simulation_app.close()
```

Each row reads: bodies, hz, solver, iterations, offset, device, physics ms/step, real-time factor, max drift (m), fallen boxes, ground sink (m), and max residual speed (m/s).

Sweep one variable at a time around a baseline (`--height 10`, TGS, 60 Hz, 4/1 iterations) with a shell loop in `run_sweep.sh`: `--hz` ∈ {30, 60, 120, 240}, `--pos-iters` ∈ {1, 2, 4, 8, 16, 32}, `--solver PGS` vs `TGS`, `--vel-iters` ∈ {0, 1, 4}, and `--contact-offset` ∈ {default, 0.002, 0.02}.

What to look for: the iteration count at which drift and fallen boxes go to zero, compared with the physics rate that achieves the same, and which of the two costs fewer ms/step; whether PGS needs more velocity iterations than TGS to stop the residual speed; and whether a tiny contact offset brings back jitter.

A 10-box stack at 60 Hz with few iterations is a deliberately hard case. Some settings should fail visibly. Run one failing configuration without `--headless` and watch it.

### Lab 4b — A Franka grasp that survives a shake

The fixture parks a cube between the open fingers with gravity disabled, closes the gripper, re-enables gravity, holds, and then shakes the arm by oscillating joint 1. This avoids motion planning, which comes in [Lecture 06](Lecture-06.md), and isolates the contact problem. The physics of the hold, with two finger contacts:

$$
2\,\mu\,F_n \;\ge\; m\,\lVert \mathbf{g} + \mathbf{a} \rVert, \qquad F_n \approx \min\!\big(k_p\,(q_{\text{contact}} - q^{*}),\; F_{\max}\big)
$$

With a position drive, the squeeze force comes from putting the finger target \( q^{*} \) *inside* the object, and it saturates at the drive's max force. The levers are therefore friction \( \mu \), stiffness \( k_p \), how far inside the target sits, and \( F_{\max} \). The penetration you see depends on iterations, timestep, and solver order.

```python
# physics_lab/franka_hold.py — tune until the cube survives the shake
import argparse
p = argparse.ArgumentParser()
p.add_argument("--friction", type=float, default=0.5)
p.add_argument("--combine", default="average", choices=["average", "min", "multiply", "max"])
p.add_argument("--finger-kp", type=float, default=None)       # None keeps the asset's gains
p.add_argument("--finger-kd", type=float, default=None)
p.add_argument("--finger-max-effort", type=float, default=None)
p.add_argument("--squeeze", type=float, default=0.0)          # finger target (m); cube contact is ~0.025
p.add_argument("--pos-iters", type=int, default=4)
p.add_argument("--hz", type=int, default=120)
p.add_argument("--contact-last", action="store_true")
p.add_argument("--mass", type=float, default=0.2)
p.add_argument("--shake-amp", type=float, default=0.5)        # rad on panda_joint1
p.add_argument("--shake-hz", type=float, default=1.0)
p.add_argument("--headless", action="store_true")
args = p.parse_args()

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": args.headless})

import numpy as np, omni.timeline
from pxr import PhysxSchema, Sdf
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.experimental.materials import RigidBodyMaterial
from isaacsim.core.experimental.objects import Cube, GroundPlane
from isaacsim.core.experimental.prims import Articulation, GeomPrim, RigidPrim
from isaacsim.core.rendering_manager import RenderingManager
from isaacsim.core.simulation_manager import PhysxScene, SimulationManager
from isaacsim.storage.native import get_assets_root_path

stage_utils.create_new_stage()
scene = PhysxScene("/World/PhysicsScene")
scene.set_steps_per_second(args.hz)
scene.set_solver_type("TGS")
scene.prim.CreateAttribute("physxScene:solveArticulationContactLast", Sdf.ValueTypeNames.Bool).Set(args.contact_last)
GroundPlane("/World/ground")
stage_utils.add_reference_to_stage(
    usd_path=get_assets_root_path() + "/Isaac/Robots_Multiphysics/FrankaRobotics/FrankaPanda/franka/franka.usda",
    path="/World/Franka")

S = 0.05                                                       # 5 cm cube, parked away from the robot
Cube("/World/cube", sizes=S, positions=np.array([[0.6, 0.4, S / 2]]))
GeomPrim("/World/cube", apply_collision_apis=True)
cube = RigidPrim("/World/cube", masses=args.mass)
cube.set_sleep_thresholds(0.0)                                 # gravity changes don't wake sleeping bodies
PhysxSchema.PhysxRigidBodyAPI.Apply(cube.prims[0]).CreateSolverPositionIterationCountAttr().Set(args.pos_iters)
mat = RigidBodyMaterial("/World/cube_material", static_frictions=args.friction, dynamic_frictions=args.friction)
mat.set_combine_modes(frictions=args.combine)
GeomPrim("/World/cube").apply_physics_materials(mat)

robot = Articulation("/World/Franka")                          # finds the ArticulationRootAPI prim below it
robot.set_solver_iteration_counts(position_counts=args.pos_iters, velocity_counts=1)
hand = RigidPrim(robot.link_paths[0][robot.link_names.index("panda_hand")])

def rotate(q, v):                                              # rotate v by quaternion q = (w, x, y, z)
    w, u = q[0], q[1:]
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)

def run(steps, per_step=None):
    for i in range(steps):
        if per_step: per_step(i)
        SimulationManager.step()
        if not args.headless and i % 2 == 0: RenderingManager.render()

omni.timeline.get_timeline_interface().play()
simulation_app.update()

HOME = {"panda_joint1": 0.0, "panda_joint2": -0.785, "panda_joint3": 0.0, "panda_joint4": -2.356,
        "panda_joint5": 0.0, "panda_joint6": 1.571, "panda_joint7": 0.785}   # a common Franka ready pose
fingers = [i for i, n in enumerate(robot.dof_names) if "finger" in n]
q = np.array([[HOME.get(n, 0.04) for n in robot.dof_names]])                 # fingers open (0.04 m each)
robot.set_dof_positions(q); robot.set_dof_position_targets(q)
if args.finger_kp is not None:
    robot.set_dof_gains(stiffnesses=args.finger_kp, dampings=args.finger_kd, dof_indices=fingers)
if args.finger_max_effort is not None:
    robot.set_dof_max_efforts(args.finger_max_effort, dof_indices=fingers)
run(args.hz)                                                                   # settle 1 s

TCP = 0.10                                     # fingertip center along the hand's z axis (m); check in the GUI
def tcp_and_cube():
    hp, hq = (a.numpy()[0] for a in hand.get_world_poses())
    return hp + rotate(hq, np.array([0.0, 0.0, TCP])), hq, cube.get_world_poses()[0].numpy()[0]

tcp, hq, _ = tcp_and_cube()
cube.set_enabled_gravities(False)
cube.set_world_poses(positions=tcp[None], orientations=hq[None])               # align the cube with the fingers
cube.set_velocities(np.zeros((1, 3)), np.zeros((1, 3)))
q[0, fingers] = args.squeeze
robot.set_dof_position_targets(q)
run(args.hz // 2)                                                              # close the gripper
cube.set_enabled_gravities(True)

slip = []
def log(_):
    tcp, _, c = tcp_and_cube(); slip.append(np.linalg.norm(c - tcp))
run(args.hz, log)                                                              # hold 1 s under gravity
j1 = robot.dof_names.index("panda_joint1")
def shake(i):
    q[0, j1] = args.shake_amp * np.sin(2 * np.pi * args.shake_hz * i / args.hz)
    robot.set_dof_position_targets(q); log(i)
run(3 * args.hz, shake)                                                        # shake 3 s

gap = robot.get_dof_positions().numpy()[0, fingers].sum()
print(f"friction={args.friction},{args.combine},kp={args.finger_kp},fmax={args.finger_max_effort},"
      f"squeeze={args.squeeze},iters={args.pos_iters},hz={args.hz},last={args.contact_last},"
      f"max_slip={max(slip):.4f},held={slip[-1] < 0.01},penetration={S - gap:.4f}")
simulation_app.close()
```

How to use it:

1. Run once with the GUI and confirm the cube really sits between the fingers. Adjust `TCP` if it doesn't. Print `robot.dof_names` and check that the finger DOFs exist under the names you expect.
2. Run the baseline headless and record it. Then change **one** lever per run, in this order: friction and combine mode (`--combine max`); finger max effort and squeeze target; stiffness and damping; position iterations, then `--hz`; `--contact-last`.
3. Stop when the cube is `held=True` with small penetration across **five** consecutive runs, and with a `--mass` 1.5× larger. That is your margin.

Mimic-joint caveat: Isaac Lab's PhysX↔Newton transfer guide notes that when the Franka's second finger is authored as a **mimic** of the first, PhysX still treats the follower as a driven DOF. Commanding both fingers then applies the drive through both joints, roughly doubling the squeeze compared with a single driven finger. Know which case your asset is in before you compare gains across engines in [Lecture 05](Lecture-05.md).

### Lab 4c — CPU vs GPU dynamics crossover

1. Reuse `box_stack.py` with short, stable stacks (`--height 4 --pos-iters 8`). Sweep `--stacks` ∈ {1, 4, 16, 64, 256, 1024}, which gives 4 to 4,096 bodies, on `--device cpu` and on `--device cuda:0`, logging `nvidia-smi` throughout.
2. Plot physics ms/step against bodies for both devices on log-log axes, and mark the crossover.
3. Repeat the GPU run at 1,024 stacks with `--buf-mult 1, 2, 4`, and record the VRAM change at a *fixed* scene. That is the fixed cost of buffer capacity.

If the largest GPU runs print `[PhysX]` buffer-overflow warnings, the stability columns are invalid until you raise `--buf-mult`.

---

## 6. Use it in the real stack

* **Isaac Lab 3.0** sets the same scene parameters through `isaaclab_physx.physics.PhysxCfg`, passed as `SimulationCfg(dt=..., physics=PhysxCfg(...))`. Its fields include `solver_type`, the min/max position and velocity iteration ranges, `enable_ccd`, `enable_stabilization`, `bounce_threshold_velocity`, `friction_offset_threshold`, `friction_correlation_distance`, `solve_articulation_contact_last`, and the `gpu_*` capacities. Per-actor properties stay in the asset's schema configs. [Lecture 09](Lecture-09.md) reads a full config.
* **Isaac Lab's PhysX tuning guide** prescribes a fixed reproduction, then the solver choice, then position iterations before velocity iterations, then contacts, then buffers. That is the same order as Lab 4b.
* **The Omni Physics Articulation Stability Guide** is the reference for drive natural frequency, max force, armature, mass ratios, self-collision filtering, and solver order. Read it before tuning an imported robot in [Lecture 06](Lecture-06.md).
* **For RL**, drive gains and friction are domain-randomization targets. See [Deep RL Lecture 13](../Deep%20RL%20for%20Robot%20Learning/Lecture-13.md). A grasp that holds at one friction value is not a grasp.

---

## 7. Measure it

| Metric | How | Why it matters |
|---|---|---|
| **Physics ms/step** | `SimulationManager.step(steps=N)` timed, then a pose read to sync | Cost of each parameter choice |
| **Real-time factor** | `(1/hz) / wall-clock per step` | Must stay ≥ 1 for sim-in-the-loop when you raise `hz` |
| **Stack drift / fallen / sink / residual speed** | Lab 4a outputs | Stability, measured instead of eyeballed |
| **Max slip, held, finger penetration** | Lab 4b outputs | Grasp quality, and the margin under mass increase |
| **CPU/GPU crossover bodies** | Lab 4c | Which pipeline to use for a given scene size |
| **VRAM vs buffer capacity** | Lab 4c `--buf-mult`, `nvidia-smi` | The fixed physics VRAM cost you chose |
| **Iterations-vs-hz Pareto** | Lab 4a | The cheapest stable configuration |

---

## 8. Ship it

Commit `physics_lab/` with:

* `box_stack.py`, `franka_hold.py`, `run_sweep.sh`, and the raw CSVs (`stack_sweep.csv`, `grasp_tuning.csv`, `crossover.csv`) plus `vram_log.csv`
* `stack_pareto.png` (ms/step vs stability for the iteration and hz sweeps) and `crossover.png`
* `PHYSICS_NOTES.md`: your chosen stack configuration and why; the grasp parameters that hold, as a baseline → change → result table; the crossover body count; the VRAM cost per `--buf-mult` step; and one paragraph on which parameters you would randomize for RL

---

## Exit criteria

You can move on when you can:

* explain contact vs rest offset, combine-mode priority, and TGS vs PGS without notes
* show a measured case where raising TGS position iterations beats raising the physics rate on ms/step, or explain why it didn't on your scene
* state why CCD and GPU dynamics are mutually exclusive in Isaac Sim, and what you would do about tunneling in a GPU RL task
* tune a Franka grasp to hold through a shake with margin, and defend each change with a before/after row
* give your card's CPU/GPU crossover and the VRAM cost of doubling the contact buffers

---

## Self-check

1. A 20 cm-thick wall is hit by a 0.5 kg ball at 30 m/s in a GPU-pipeline RL environment, and the ball sometimes passes through. A teammate wants to enable CCD. What happens, and what are two alternatives that keep the GPU pipeline?
2. Your 10-box stack is stable at 240 Hz with 4 position iterations, and also at 60 Hz with 16 iterations (TGS). Which do you expect to be cheaper per simulated second, and why? What one measurement settles it?
3. The Franka holds the cube, but the fingers visibly sink 4 mm into it. The finger drive has very high stiffness and max force, and its target is fully closed. List three changes, in the order you would try them, and say what each one changes physically.
4. An imported gripper link has an authored mass of 1e-4 kg and is attached to a 2 kg wrist through a stiff drive. The gripper jitters even with no contact. Use \( \omega_n \Delta t \) to explain why, and name two fixes besides "lower the stiffness".
5. You set `static_frictions=1.0` with `max` combine on a cube, but the gripper pads' material uses `min`. Which friction does the contact use, and why?
6. Lab 4c shows the GPU pipeline slower than the CPU for 64 bodies but 10× faster for 4,096. Your ROS 2 sim-in-the-loop scene has 40 bodies and one robot; your RL task has 4,096 environments. Which pipeline do you pick for each, and what do you lose in each case?

---

## References

* Isaac Sim 6.1 — [Physics Simulation Fundamentals](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/physics/simulation_fundamentals.html) (colliders, offsets, materials, combine modes, CCD, joints)
* Isaac Sim 6.1 — [Gain Tuner extension](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/robot_setup/ext_isaacsim_robot_setup_gain_tuner.html)
* Isaac Sim 5.1 — [release notes](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/overview/release_notes.html) (solve articulation contacts last)
* Isaac Sim v6.1.0 source — [`PhysxScene` / `PhysxGpuCfg`](https://github.com/isaac-sim/IsaacSim/blob/v6.1.0/source/extensions/isaacsim.core.simulation_manager/python/impl/physx_scene.py), [`Articulation`](https://github.com/isaac-sim/IsaacSim/blob/v6.1.0/source/extensions/isaacsim.core.experimental.prims/python/impl/articulation.py)
* PhysX 5.6 SDK — [Rigid Body Dynamics](https://nvidia-omniverse.github.io/PhysX/physx/5.6.0/docs/RigidBodyDynamics.html) (TGS vs PGS, iterations, sleeping, determinism), [Articulations](https://nvidia-omniverse.github.io/PhysX/physx/5.6.0/docs/Articulations.html), [Advanced Collision Detection](https://nvidia-omniverse.github.io/PhysX/physx/5.6.0/docs/AdvancedCollisionDetection.html)
* Omni Physics — [Articulation and Robot Simulation Stability Guide](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/guides/articulation_stability_guide.html)
* Isaac Lab 3.0 — [Tune the PhysX Solver](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/concepts/solver-tuning/tune_physx.html)
* OpenUSD — [UsdPhysics overview](https://openusd.org/release/api/usd_physics_page_front.html)

---

## Next in this special course

* Next: [Lecture 05 — Newton and Multi-Physics](Lecture-05.md)
* Previous: [Lecture 03 — The Core Experimental API](Lecture-03.md)
* Back: [Isaac Sim and Isaac Lab — Overview](README.md)
