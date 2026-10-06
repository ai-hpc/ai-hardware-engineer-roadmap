# Lecture 06: Robots: Import, Tune, and Move

## Overview

Every robot you simulate starts as someone else's description file: a URDF from a ROS package, an MJCF from a MuJoCo model zoo, or a USD from NVIDIA's asset library. Of everything in your scene, that file is the part you should trust least. Inertias get copied from CAD with the wrong units. Collision meshes are missing, or they are the 200k-triangle visual meshes. Joint limits use the wrong sign. Drive gains are zero because URDF has no notion of a position controller. The simulator will run all of it without complaint, and then the arm sags, the gripper slips, or a policy learns to exploit a phantom collision.

This lecture takes a robot from a description file to a moving, validated asset in Isaac Sim 6.1. You import with the rewritten 6.x importers, run a validation checklist that catches the usual problems, attach a gripper with the Robot Assembler, tune drives, and plan a pick-and-place with the new cuMotion integration while measuring planning latency.

By the end you should be able to:

* import a URDF or MJCF with `URDFImporter` / `MJCFImporter` and explain each importer option that changes physics
* run a validation checklist (units, mass/inertia, collision coverage, joint limits and axes, drive gains, base type) and fix what it finds
* attach a gripper with the Robot Assembler and explain why the result is a variant on the robot asset
* reason about drive gains with the PD model, and know when to reach for the Gain Tuner or System Identification
* plan collision-free motions with cuMotion (`CumotionRobot`, `CumotionWorldInterface`, `GraphBasedMotionPlanner`) and execute them through joint position targets
* say what collision geometry and motion planning cost on an 8 GB card

---

## 1. Why it matters: the robot is the asset you trust least

Robot bugs show up later and somewhere else, which is why they are expensive:

| Symptom later | Usual root cause in the asset | Where it bites |
|---|---|---|
| Arm sags or oscillates at a "hold" target | Zero or wrong-unit drive gains, max force too low | Scripted control, ROS 2 control (Lecture 08) |
| Cube passes through the fingers | Missing finger colliders, or collisions only on visuals | Grasping (Lecture 04), pick-and-place |
| Robot explodes on Play | Overlapping self-collision pairs; bad inertia (zero, or violating the triangle inequality) | Everything |
| Fixed arm falls over / mobile base is welded to the world | Wrong base type at import | Scene setup |
| Policy trains in sim, fails on hardware | Gains, friction, armature, delays not identified | Sim-to-real (Lecture 12) |
| Simulation slow for no visible reason | Convex decomposition on every link, unmerged meshes | Physics ms/step, VRAM |

A validated robot USD is also the artifact every later lecture reuses. Lectures 08, 10 and 13 assume the robot holds a pose, has sensible collision geometry, and follows targets.

---

## 2. Mental model: description file → validated, movable asset

```text
 URDF / MJCF ──► importer (6.x) ─────────────────────────────► robot package (USD)
                 │ urdf-usd-converter → flattened stage            <name>/<name>.usda
                 │ fix base · density · drive overrides            instanceable meshes
                 │ merge meshes · collision from visuals           ArticulationRootAPI
                 │ self-collision · robot schema                   + Newton schemas
                 │ multi-physics conversion (PhysX ⇄ MuJoCo attrs) mimic → NewtonMimicAPI
                 └ asset transformer (layered package)
                                                     │
                       validate (checklist, SimReady validator, self-collision detector)
                                                     │
                       assemble (Robot Assembler → gripper variant)
                                                     │
                       tune (Gain Tuner; System Identification against real data)
                                                     │
                       move (cuMotion: RMPflow · RRT · trajectory gen/opt · Pink IK)
```

### 2.1 The 6.x importers

Both importers (`isaacsim.asset.importer.urdf`, `isaacsim.asset.importer.mjcf`) were rewritten in 6.0. The Python API is a config dataclass plus an importer object:

```python
from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig

cfg = URDFImporterConfig(urdf_path="/path/robot.urdf", usd_path="/path/out", fix_base=True)
usd_path = URDFImporter(cfg).import_urdf()        # returns <usd_path>/<robot_name>/<robot_name>.usda
```

The options that change physics, as they appear on `URDFImporterConfig` in v6.1.0 (MJCF has the same core set on `MJCFImporterConfig`, plus `import_scene` and MuJoCo gain/bias overrides):

| Field | Default | What it does | When to change it |
|---|---|---|---|
| `fix_base` | `None` | **Tri-state.** `None` keeps the source authoring; `True` adds a world-to-root fixed joint and moves `ArticulationRootAPI` to the right ancestor; `False` strips an existing world joint (floating base) | Always set it explicitly per robot class: `True` for bolted arms, `False` for mobile robots |
| `collision_from_visuals` / `collision_type` | `False` / `"Convex Hull"` | Generate colliders from visual meshes: `"Convex Hull"`, `"Convex Decomposition"`, `"Bounding Sphere"`, `"Bounding Cube"` | Only when the URDF has no collision geometry. Decomposition is accurate and expensive (§6) |
| `allow_self_collision` | `False` | Enables self-collision on the articulation | Needed for robots that can hit themselves; check overlapping pairs first (§3) |
| `merge_fixed_joints` | `False` | Merges links joined by fixed joints before conversion | Fewer bodies and shapes; keep frames you need (tool frames, sensor mounts) |
| `merge_mesh` | `False` | Merges meshes under a rigid body | Fewer prims, faster load and render |
| `link_density` | `None` | Density for links with no explicit mass | URDFs exported without inertials |
| `joint_drive_type`, `joint_target_type` | `None` | `"force"`/`"acceleration"`; `"none"`/`"position"`/`"velocity"`. A dict of joint-name regexes sets values per joint | Torque-controlled legs → target `"none"`; wheels → `"velocity"` |
| `override_joint_stiffness`, `override_joint_damping` | `None` | Gains in Nm/rad (revolute) or N/m (prismatic), scalar or per-regex dict | A first guess; tune afterwards (§4) |
| `robot_type` | `"Default"` | Sets `isaac:robotType` (Manipulator, Quadruped, Wheeled, …) on the robot schema | Tools such as the Gain Tuner discover robots through the robot schema |
| `run_multi_physics_conversion` | `True` | Writes PhysX and MuJoCo (`mjc:*`) equivalents of URDF joint data (effort → `maxForce`, friction, armature) | Leave on if you might switch to Newton (Lecture 05) |

What else changed in 6.x, which you need to know when reading an imported asset:

* **Output is instanceable** ("the meshes are already instantiable"), and in the Isaac Sim asset-structure layout produced by the asset transformer. Isaac Lab 3.0 also notes nested rigid bodies. A plain `stage.Traverse()` does not enter instance proxies, so it will miss colliders. Lab 6a shows the fix.
* **Articulation and mimic authoring are engine-neutral.** The root link gets `UsdPhysics.ArticulationRootAPI` plus `NewtonArticulationRootAPI`, and self-collision is authored as `newton:selfCollisionEnabled`. `PhysxArticulationAPI` is no longer authored. URDF `<mimic>` joints become `NewtonMimicAPI` (`newton:mimicJoint`, `newton:mimicCoef0/1`), not `PhysxMimicJointAPI`.
* **Gains are your job.** NVIDIA's own UR10e tutorial says "the importer does not set the gains for the UR robot automatically" and moves straight to the Gain Tuner.
* **Importing from a running ROS 2 system** goes through `isaacsim.ros2.urdf` (`RobotDefinitionReader` reads `robot_description`, then you call `URDFImporter`). This is also how you import xacro without converting it by hand. Launch Isaac Sim from a terminal with the robot's workspace sourced so `package://` URLs resolve; see [Advanced ROS Lecture 01](../Advanced%20Robot%20Operating%20System/Lecture-01.md).

> **You will see this in older code.** 5.x scripts call `omni.kit.commands.execute("URDFParseAndImportFile", ...)` with a config from `URDFCreateImportConfig`, set a boolean `import_config.fix_base = True`, or pass URDF text to `URDFParseText`. `URDFParseAndImportFile` was **removed in 6.0 with no shim**. It fails immediately. The others are deprecated transition aids. The new importer only takes a file path, so write a string to a temp file first.

### 2.2 Units and conventions that bite

| Quantity | URDF | USD as authored by the importer | Core Experimental `Articulation` API |
|---|---|---|---|
| Length | metres | stage units (`metersPerUnit`, normally 1.0) | stage units |
| Revolute limits / positions | radians | **degrees** (`physics:lowerLimit`) | **radians** (the wrapper converts) |
| Revolute drive gains | n/a | per degree under PhysX (the Gain Tuner labels angular joints in degrees; radians under Newton) | per radian via the wrapper |
| Orientation | RPY | quaternion | **wxyz** (Isaac Lab 3.0 uses xyzw, see Lecture 09) |

The `Articulation` wrapper in v6.1.0 does the deg↔rad conversion itself (`np.deg2rad` on USD limits, `np.rad2deg` when it writes targets to USD). So the classic mistake is mixing layers: reading a limit from USD with `pxr` and comparing it with a position from `get_dof_positions()`.

---

## 3. Validate before you trust: the checklist

Run this on every imported robot, and again after every assembly. Lab 6a automates most of it.

| # | Check | How | Failure looks like | Fix |
|---|---|---|---|---|
| 1 | Units | `UsdGeom.GetStageMetersPerUnit`, `GetStageUpAxis` | Robot 100× too big or small; lying on its side | Fix the source, or set stage units before referencing |
| 2 | One articulation root, on the right prim | Count prims with `ArticulationRootAPI` | 0 roots: links fall apart; 2 roots after assembly | Re-import with explicit `fix_base`; disable the attached robot's root |
| 3 | Base type | Hold test: play with no targets changed, watch the root pose | Fixed arm topples, or mobile base cannot move | `fix_base=True/False` |
| 4 | Mass and inertia | `get_link_masses()`, `get_link_inertias()` → eigenvalues > 0 and \( I_1 + I_2 \ge I_3 \) | Jitter, explosions, "massless" links | Fix inertials in the source; `link_density` as a fallback |
| 5 | Collision coverage | Every rigid body has ≥ 1 collider (traverse instance proxies); viewport: *Show by type → Physics → Colliders* | Fingers pass through objects | Add collision geometry, or `collision_from_visuals` with a deliberate `collision_type` |
| 6 | Self-collision at rest | Robot Self-Collision Detector (`isaacsim.robot_setup.collision_detector`) | Robot twitches or explodes at t = 0 | Filter adjacent overlapping pairs (`UsdPhysics.FilteredPairsAPI`) |
| 7 | Joint limits and axes | `get_dof_limits()` vs datasheet; jog each joint in the GUI | Joint moves the wrong way, or past its stops | Fix the URDF axis/sign; never patch it downstream |
| 8 | Drives | `get_dof_gains()`, `get_dof_max_efforts()`; hold test drift | Sag under gravity, oscillation, saturation | §4 |
| 9 | Asset rules | *Window → SimReady Asset Validation*, profile **Robot Assets** | Failed requirements (e.g. "joint has drive or mimic API") | Follow each finding's guidance |

The SimReady validator (`isaacsim.asset.validation`) checks structure and authoring rules. Checks 3, 4 and 8 need a running simulation and a physics view, so you still need the script.

---

## 4. Assemble and tune

### 4.1 Robot Assembler

`isaacsim.robot_setup.assembler` joins two robot assets (both with the robot schema applied) with a **physically simulated fixed joint**: arm + gripper, or arm + moving base. Each side has an *attach point* (a link or site). For the UR10e + Robotiq 2F-140 tutorial these are `wrist_3_link` and `robotiq_arg2f_base_link`. Two consequences follow:

* The joint only acts while the timeline plays. Do not use the assembler to bolt a robot to a static table. Just place both.
* If you assemble on the base robot's own stage, the attachment is saved next to the robot asset as `payloads/<namespace>/<variant_name>.usd` (that is what the v6.1.0 `robot_assembler.py` writes; the 6.1 docs page still describes an older `configuration/<robot>_<namespace>_<attach>.usd` name) and exposed as a **variant set named after the namespace** (default `Gripper`) on the robot. Every scene that references the robot can then choose its gripper, or `None`. This is the USD variant mechanism from Lecture 02 in practice. The 6.1 hosted Franka asset ships variant sets `Gripper`, `Mesh` (with a `performance` option) and `Physics`, and the UR10 asset ships a `Gripper` variant set with a suction option. Assembling a stage that *references* the base robot instead edits only that stage.

The Python API mirrors the UI: `RobotAssembler().begin_assembly(stage, base, base_mount, attach, attach_mount, namespace, variant_name)` → `assemble()` → (simulate and check) → `finish_assemble()` or `cancel_assembly()` (the 6.1 docs page spells it `cancel_assemble()`, which does not exist in the v6.1.0 source). The assembler removes the attached robot's articulation root, but re-run checks 2 and 8 anyway.

### 4.2 Drive gains: the PD model

A USD joint drive is an implicit PD controller. Written in restoring form, the drive the Gain Tuner describes is

$$
\tau = k_p\,(q_{\text{target}} - q) + k_d\,(\dot q_{\text{target}} - \dot q), \qquad |\tau| \le \tau_{\max}
$$

Stiffness \( k_p > 0 \) gives position control. \( k_p = 0, k_d > 0 \) gives velocity control (wheels). A drive of type *acceleration* is normalized by the joint's mass, so it behaves like an ideal actuator that is independent of configuration. A *force* drive applies \( \tau \) directly. NVIDIA's manipulator tutorial tunes with natural frequency and damping ratio, where \( m \) is the joint's effective inertia:

$$
\omega_n = \sqrt{k_p / m}, \qquad \zeta = \frac{k_d}{2\,m\,\omega_n}
$$

\( \zeta = 1 \) is critically damped. Raise \( \omega_n \) for tighter tracking, until \( \tau_{\max} \) saturates or the physics timestep can no longer resolve the response. The Gain Tuner's *dt Sweep* test measures exactly that. In practice:

* **Gain Tuner** (`isaacsim.robot_setup.gain_tuner`, *Tools → Robotics → Asset Editors → Gain Tuner*). It edits gains per joint for whichever backend is active (PhysX `DriveAPI`, MuJoCo `mjc:*`, Newton actuators) and runs *Snap to Limits*, *Sinusoidal*, *Step*, *Stress* (random extreme commands, the RL failure mode) and *dt Sweep* tests.
* **System Identification** (`isaacsim.robot_setup.sysid` + `.sysid.ui`). It fits friction, stiffness, damping, armature, link mass/CoM/inertia, joint-limit scales and command delay to **recorded** robot data (CSV, ROS 2 bag, MCAP, LeRobot). It has train/validation chunks, an identifiability *Check* page, and optimizers (Levenberg-Marquardt, CMA-ES, Bayesian, Adam on a differentiable Newton Featherstone path). Use it when the asset is structurally right but its motion does not match the real robot. That is a Lecture 12 concern, but the tool belongs to this pipeline.

---

## 5. Moving: motion generation in 6.x

Isaac Sim 6.0 deprecated `isaacsim.robot_motion.lula` and `isaacsim.robot_motion.motion_generation`. The replacement has three parts:

| Package | Role |
|---|---|
| `isaacsim.robot_motion.experimental.motion_generation` | Common layer: `RobotState`/`JointState`/`SpatialState`, `BaseController`, `Path`, `Trajectory`, `TrajectoryFollower`, `SceneQuery`, `ObstacleStrategy`, `WorldBinding`, `WorldInterface` |
| `isaacsim.robot_motion.cumotion` | GPU-accelerated planning/control, "descended from the Lula library": `CumotionRobot`, `CumotionWorldInterface`, `RmpFlowController`, `GraphBasedMotionPlanner`, `TrajectoryGenerator`, `TrajectoryOptimizer` |
| `isaacsim.robot_motion.pink` | Reactive differential IK (Pink) |

The four cuMotion algorithms answer different questions:

| Algorithm | Class | Output | Collision-aware | Use for |
|---|---|---|---|---|
| RMPflow | `RmpFlowController` | Joint targets every step (reactive) | Yes | Following a moving target; teleop-like control |
| Graph planning (RRT variants) | `GraphBasedMotionPlanner` | `Path` (waypoints) → `to_minimal_time_joint_trajectory(...)` | Yes | Global collision-free point-to-point moves |
| Trajectory generation | `TrajectoryGenerator` | Time-optimal trajectory through waypoints / path specs | **No** | Smooth timing of a path you already trust |
| Trajectory optimization | `TrajectoryOptimizer` | `CumotionTrajectory` from `plan_to_goal(q0, target)` | Yes, with joint/velocity/acceleration limits | Smooth, constrained global trajectories (new vs Lula) |

The two objects every algorithm shares:

* **`CumotionRobot`** comes from `load_cumotion_supported_robot("franka" | "ur10")` or `load_cumotion_robot(directory=...)`. A custom robot needs `robot.urdf` plus **`robot.xrdf`**. The XRDF holds cuMotion's view of the robot: the active vs fixed joints that define c-space, the **collision spheres** used for planning, tool frames, and self-collision ignore lists. You generate it with the Robot Description Editor (`isaacsim.robot_setup.xrdf_editor`) from a URDF exported from your USD. If your robot's link names or collision geometry change, regenerate the XRDF.
* **`CumotionWorldInterface`** is the planner's collision world. A `WorldBinding` fills it from the stage: `SceneQuery.get_prims_in_aabb(...)` finds prims with collision APIs, an `ObstacleStrategy` picks a representation per shape type (`"cube"`, `"obb"`, `"triangulated_mesh"`, plus a safety margin), `initialize()` builds the world, and `synchronize_transforms()` pushes poses each frame. The binding does **not** track shape or scale changes, so rebuild it when obstacles are added or resized. The robot base pose is separate: `update_world_to_robot_root_transforms(...)`.

Frames: cuMotion works in the robot base frame. The graph planner's `plan_to_pose_target` takes world-frame position and a **wxyz** quaternion. The optimizer's task-space targets go through `isaac_sim_to_cumotion_pose(...)`. Trajectories are not tied to the articulation, so start every plan from the robot's *current* joint state.

> **You will see this in older code.** 5.x scripts import `RMPFlowController`, `ArticulationKinematicsSolver` or `LulaKinematicsSolver` from `isaacsim.robot_motion.motion_generation`, load "robot description" YAMLs, and step everything through `World`. These are deprecated in 6.0. The Isaac Sim docs' **cuRobo tutorials are "no longer maintained"**, and the cuMotion integration covers most of the same ground. cuRobo and cuMotion are not supported on DGX Spark.

---

## 6. The hardware view: collision geometry and planners share the card

**Collision geometry sets physics cost.** Narrow-phase work and contact buffers scale with the number of shape pairs that come close, not with the number of links. If a link is decomposed into \( k \) convex hulls and touches a link with \( k' \) hulls, the pair can cost up to \( k \cdot k' \) narrow-phase tests where a single hull per link would cost one. So the importer's `collision_type` is a performance knob. The cheapest is a bounding sphere or box, then one convex hull per link, and convex decomposition is the most expensive. Spend decomposition only where contact accuracy matters (fingers, the palm) and keep hulls elsewhere. `Articulation.num_shapes` gives the count. Track it next to physics ms/step.

**Visual meshes set VRAM and load time, not physics.** Unique meshes and textures cost memory once each. Instanced meshes, which is the importer's default output, share the memory. `merge_mesh` and the Franka asset's `Mesh: performance` variant cut prim counts. Re-measure frame ms when you switch variants.

**Planning shares the card with everything else.** cuMotion is GPU-accelerated, and its world interface lives on the Warp device you give it. NVIDIA's RMPflow example passes `device="cpu"`, and the trajectory-optimizer example uses the default. Planning latency is wall-clock time in your control loop. RRT is stochastic, so expect a distribution with a tail, not a single number. The first call includes warm-up and allocation, so discard it. Planner GPU work competes with physics and RTX for SMs and memory, so measure it with the scene running, not in isolation.

> **8 GB budget.** One imported arm plus a gripper, a few obstacles, and a cuMotion planner fit comfortably on an RTX 4060. Even so, the card is below NVIDIA's 16 GB minimum, so close other GPU apps and keep the Lecture 01 fixed-cost number in mind. Run validation and planning-latency sweeps headless. Use hull colliders except at contact surfaces, the `performance` mesh variant, and planner device `cpu` if VRAM is tight (measure both). What does not fit is many robots × decomposed colliders × cameras. That is a Lecture 07/10 problem, and a cloud L40S or RTX PRO 6000 is the fallback, not a bigger planner.

---

## 7. Build it

Everything goes in `robot_lab/`. Timing and VRAM reuse Lecture 01's conventions (`nvidia-smi` logging, ms per step).

### Lab 6a — Import, validate, fix one issue

The importer ships sample URDFs (UR10, Carter, Kaya) under its extension's `data/urdf/robots/`. Start with the UR10, then repeat with a public arm of your choice (for example the UR or Franka description packages from ROS 2 Jazzy).

```python
# robot_lab/import_and_validate.py
import argparse, json, os
p = argparse.ArgumentParser()
p.add_argument("--urdf", default=None)                      # default: the UR10 sample shipped with the importer
p.add_argument("--out", default="robot_lab/usd")
p.add_argument("--fix-base", choices=["source", "fixed", "mobile"], default="source")
p.add_argument("--collision-type", default=None)            # e.g. "Convex Decomposition" (implies collision_from_visuals)
args = p.parse_args()

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import numpy as np, omni.kit.app
ext = omni.kit.app.get_app().get_extension_manager()
for name in ("omni.scene.optimizer.core", "isaacsim.robot.schema"):   # the official urdf_import.py enables these first
    ext.set_extension_enabled_immediate(name, True)

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig
from isaacsim.core.experimental.prims import Articulation
from pxr import Usd, UsdGeom, UsdPhysics

urdf = args.urdf or os.path.join(ext.get_extension_path(ext.get_enabled_extension_id("isaacsim.asset.importer.urdf")),
                                 "data", "urdf", "robots", "ur10", "urdf", "ur10.urdf")
cfg = URDFImporterConfig(urdf_path=urdf, usd_path=os.path.abspath(args.out),
                         fix_base={"source": None, "fixed": True, "mobile": False}[args.fix_base],
                         collision_from_visuals=args.collision_type is not None,
                         collision_type=args.collision_type or "Convex Hull")
usd_path = URDFImporter(cfg).import_urdf()

# --- static checks on the composed USD ---------------------------------------
stage = Usd.Stage.Open(usd_path)
prims = list(stage.Traverse(Usd.TraverseInstanceProxies()))   # instanceable output hides colliders from Traverse()
roots = [str(p.GetPath()) for p in prims if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
bodies = {str(p.GetPath()) for p in prims if p.HasAPI(UsdPhysics.RigidBodyAPI)}
colliders_per_body = dict.fromkeys(bodies, 0)
for p in prims:
    if p.HasAPI(UsdPhysics.CollisionAPI):
        q = p
        while q and str(q.GetPath()) not in bodies:              # nearest rigid-body ancestor (bodies can nest)
            q = q.GetParent()
        if q:
            colliders_per_body[str(q.GetPath())] += 1
joints = []
for p in prims:
    if p.IsA(UsdPhysics.RevoluteJoint):
        j, d = UsdPhysics.RevoluteJoint(p), UsdPhysics.DriveAPI.Get(p, "angular")
        joints.append({"joint": p.GetName(), "axis": str(j.GetAxisAttr().Get()),
                       "limits_deg": [j.GetLowerLimitAttr().Get(), j.GetUpperLimitAttr().Get()],
                       "kp_usd": d.GetStiffnessAttr().Get(), "kd_usd": d.GetDampingAttr().Get(),
                       "max_force": d.GetMaxForceAttr().Get()})
report = {"usd": usd_path, "metersPerUnit": UsdGeom.GetStageMetersPerUnit(stage),
          "upAxis": str(UsdGeom.GetStageUpAxis(stage)), "articulation_roots": roots,
          "bodies_without_colliders": [b for b, n in colliders_per_body.items() if n == 0],
          "joints": joints}

# --- runtime checks: masses, inertias, limits, gains, hold test --------------
stage_utils.open_stage(usd_path)
robot = Articulation(roots[0])
app_utils.play(); app_utils.update_app(steps=5)                # a default physics scene is created on play
masses = robot.get_link_masses().numpy()[0]
eig = np.linalg.eigvalsh(robot.get_link_inertias().numpy()[0].reshape(-1, 3, 3))   # (L, 3), ascending
kp, kd = (a.numpy()[0] for a in robot.get_dof_gains())
q0, root0 = robot.get_dof_positions().numpy(), robot.get_world_poses()[0].numpy()
robot.set_dof_position_targets(q0)                             # "hold this pose" for 240 frames
[simulation_app.update() for _ in range(240)]
report |= {"num_shapes": robot.num_shapes, "dof_names": robot.dof_names,
           "massless_links": [n for n, m in zip(robot.link_names, masses) if m <= 0],
           "bad_inertia_links": [n for n, e in zip(robot.link_names, eig) if e[0] <= 0 or e[0] + e[1] < e[2]],
           "zero_gain_dofs": [n for n, k in zip(robot.dof_names, kp) if k == 0],
           "hold_drift_rad": float(np.abs(robot.get_dof_positions().numpy() - q0).max()),
           "root_drift_m": float(np.linalg.norm(robot.get_world_poses()[0].numpy() - root0))}
print(json.dumps(report, indent=1, default=str))
json.dump(report, open(os.path.join(args.out, "validation.json"), "w"), indent=1, default=str)
simulation_app.close()
```

```bash
python robot_lab/import_and_validate.py --fix-base source
python robot_lab/import_and_validate.py --fix-base fixed
```

Interpret the two runs. With `source`, a URDF that has no world link imports as a floating base, and the arm falls: `root_drift_m` is large. With `fixed`, root drift should be near zero and `articulation_roots` should point at the correct ancestor. Then read `zero_gain_dofs` and `hold_drift_rad`. Expect the importer to leave gains you have to set. Fix **one** issue properly: re-import with `override_joint_stiffness`/`override_joint_damping` as a first guess, then refine in the Gain Tuner until the hold drift is small and the *Step* test does not overshoot. Write down what was wrong, how the checklist caught it, and the before/after numbers. If `num_shapes` or `bad_inertia_links` look wrong for your second robot, that is a better issue to fix.

### Lab 6b — Attach a gripper

Use the UR10e and Robotiq 2F-140 assets from NVIDIA's *Setup a Manipulator* tutorial (Content Browser: `Isaac Sim/Samples/Rigging/Manipulator/import_manipulator/`), or your own arm plus a gripper from the asset library. First do it in the GUI (*Tools → Robotics → Asset Editors → Robot Assembler*), then script it:

```python
# robot_lab/assemble.py  (Script Editor, both robots on the stage). Mount prims are links; with nested rigid
# bodies they may sit deeper than shown, so copy the real paths from Window > Robot Inspector.
import omni.usd
from isaacsim.robot_setup.assembler import RobotAssembler

asm = RobotAssembler()
asm.begin_assembly(omni.usd.get_context().get_stage(), "/ur", "/ur/<path-to>/wrist_3_link",
                   "/ur/robotiq_2f_140", "/ur/robotiq_2f_140/<path-to>/robotiq_arg2f_base_link",
                   "Gripper", "robotiq_2f_140")   # then fix the gripper pose via USD (tutorial: +90° about Z)
asm.assemble()          # fixed joint created, attached articulation root disabled; play and check now
asm.finish_assemble()   # or asm.cancel_assembly(); writes payloads/Gripper/robotiq_2f_140.usd + Gripper variant. Save the stage.
```

Then re-run the Lab 6a checks on the saved asset. Expect one articulation root and the gripper's DOFs (mimic joints included) in `dof_names`. Switch the `Gripper` variant to `None` and back, and confirm the DOF count changes. Tune the finger drive in the Gain Tuner.

### Lab 6c — Scripted pick-and-place with cuMotion, timed

This follows the 6.1 cuMotion examples (`rmpflow_follow_target.py` and the graph-planner scenario). It uses the hosted Franka and NVIDIA's Franka grasp conventions from `isaacsim.robot_motion.examples`: tool frame `panda_hand`, grasp orientation wxyz `(0, 1, 0, 0)`, and a 0.1034 m hand-to-grasp offset.

```python
# robot_lab/pick_place_cumotion.py
import argparse, json, time
p = argparse.ArgumentParser()
p.add_argument("--headless", action="store_true")
p.add_argument("--device", default="cpu")          # Warp device for the planning world ("cpu" or "cuda:0")
p.add_argument("--repeats", type=int, default=20)  # repeat the first query for a latency distribution
args = p.parse_args()

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": args.headless})

import numpy as np
import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.experimental.objects import Cube, GroundPlane
from isaacsim.core.experimental.prims import Articulation, GeomPrim, RigidPrim
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.robot_motion.cumotion import CumotionWorldInterface, GraphBasedMotionPlanner, load_cumotion_supported_robot
from isaacsim.robot_motion.experimental.motion_generation import (ObstacleConfiguration, ObstacleStrategy,
                                                                   SceneQuery, TrackableApi, WorldBinding)
from isaacsim.storage.native import get_assets_root_path

DT, ROBOT, CUBE, WALL, GROUND = 1 / 60, "/World/Franka", "/World/cube", "/World/wall", "/World/ground"
PICK, PLACE = np.array([0.5, -0.25, 0.025]), np.array([0.5, 0.25, 0.025])
HAND_DOWN, UP = np.array([0.0, 1.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.1034])   # wxyz; hand origin above grasp point

stage_utils.create_new_stage(template="default stage")
stage_utils.set_stage_up_axis("Z"); stage_utils.set_stage_units(meters_per_unit=1.0)
stage_utils.add_reference_to_stage(
    usd_path=get_assets_root_path() + "/Isaac/Robots_Multiphysics/FrankaRobotics/FrankaPanda/franka/franka.usda", path=ROBOT)
robot = Articulation(ROBOT)
GroundPlane(GROUND, sizes=10.0)
Cube(CUBE, sizes=0.05, positions=[PICK]); GeomPrim(CUBE, apply_collision_apis=True); cube = RigidPrim(CUBE, masses=[0.05])
Cube(WALL, sizes=0.1, positions=[[0.5, 0.0, 0.15]]); GeomPrim(WALL, apply_collision_apis=True)   # static obstacle
SimulationManager.setup_simulation(dt=DT)
app_utils.play(); app_utils.update_app(steps=10)

cfg = load_cumotion_supported_robot("franka")
arm = cfg.controlled_joint_names                                       # the 7 c-space joints from the XRDF
arm_idx = robot.get_dof_indices(arm).numpy().flatten()
fingers = robot.get_dof_indices(["panda_finger_joint1", "panda_finger_joint2"]).numpy().flatten()

pos, ori = robot.get_world_poses()
obstacles = SceneQuery().get_prims_in_aabb(search_box_origin=pos.numpy()[0], search_box_minimum=[-2, -2, -2],
    search_box_maximum=[2, 2, 2], tracked_api=TrackableApi.PHYSICS_COLLISION,
    exclude_prim_paths=[ROBOT, CUBE, GROUND])                          # never plan around what you are about to grasp
strategy = ObstacleStrategy(); strategy.set_default_configuration(Cube, ObstacleConfiguration("cube", 0.02))
binding = WorldBinding(world_interface=CumotionWorldInterface(device=args.device), obstacle_strategy=strategy,
                       tracked_prims=obstacles, tracked_collision_api=TrackableApi.PHYSICS_COLLISION)
binding.initialize()
binding.get_world_interface().update_world_to_robot_root_transforms(poses=(pos, ori))
binding.synchronize_transforms()
planner = GraphBasedMotionPlanner(cumotion_robot=cfg, cumotion_world_interface=binding.get_world_interface(),
                                  tool_frame="panda_hand")

latency_ms, durations = {}, {}
q_arm = lambda: robot.get_dof_positions(dof_indices=arm_idx).numpy().flatten().astype(np.float64)

def plan(name, hand_pos, repeats=1):
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        path = planner.plan_to_pose_target(q_initial=q_arm(), position=hand_pos, orientation=HAND_DOWN)
        times.append((time.perf_counter() - t) * 1e3)
    latency_ms[name] = times
    if path is None:
        raise RuntimeError(f"planning failed: {name}")
    traj = path.to_minimal_time_joint_trajectory(max_velocities=np.full(7, 1.0), max_accelerations=np.full(7, 1.0),
                                                 robot_joint_space=robot.dof_names, active_joints=arm)
    durations[name] = traj.duration
    return traj

def run(traj):
    t = 0.0
    while t <= traj.duration:                       # assumes one physics step of DT per app update
        s = traj.get_target_state(t)
        if s is not None and s.joints.positions is not None:
            robot.set_dof_position_targets(s.joints.positions, dof_indices=s.joints.position_indices)
        simulation_app.update(); t += DT
    [simulation_app.update() for _ in range(30)]    # settle

def gripper(width):                                 # per-finger opening in metres (0.04 = open, 0.0 = closed)
    robot.set_dof_position_targets(np.full((1, 2), width), dof_indices=fingers)
    [simulation_app.update() for _ in range(45)]

gripper(0.04)
run(plan("pregrasp", PICK + UP + [0, 0, 0.15], repeats=args.repeats))
run(plan("grasp", PICK + UP)); gripper(0.0)
run(plan("lift", PICK + UP + [0, 0, 0.15]))
lifted_z = float(cube.get_world_poses()[0].numpy()[0, 2])
run(plan("over_place", PLACE + UP + [0, 0, 0.15]))
run(plan("place", PLACE + UP + [0, 0, 0.005])); gripper(0.04)
run(plan("retreat", PLACE + UP + [0, 0, 0.15]))
err = float(np.linalg.norm(cube.get_world_poses()[0].numpy()[0, :2] - PLACE[:2]))
lat = np.array(latency_ms["pregrasp"][1:])          # drop the warm-up call
print(json.dumps({"device": args.device, "lifted_z": lifted_z, "place_error_m": err,
                  "pregrasp_ms_p50": float(np.percentile(lat, 50)), "pregrasp_ms_p95": float(np.percentile(lat, 95)),
                  "first_call_ms": latency_ms["pregrasp"][0], "per_phase_ms": latency_ms, "traj_s": durations}, indent=1))
simulation_app.close()
```

```bash
python robot_lab/pick_place_cumotion.py --device cpu    | tee robot_lab/pick_cpu.json
python robot_lab/pick_place_cumotion.py --device cuda:0 | tee robot_lab/pick_gpu.json
```

Success means `lifted_z` is clearly above 0.025 (the cube came up with the hand) and `place_error_m` is a few centimetres or less. If the cube slips, the cause is a Lecture 04 problem (friction, finger drive force, solver iterations), not a planner problem. Fix it there. Three limitations are deliberate, and you should be able to explain each:

* **Approach.** The `grasp` move is an RRT path, not a straight line, so the fingers may sweep into the cube sideways. The trajectory generator's task-space path specifications are the tool for a straight Cartesian approach.
* **Floor.** The floor is excluded from the planning world. Add a thin box obstacle if paths dip below the table.
* **Attached object.** The planner does not know the cube is attached during the carry, so check clearance against the wall by eye.

Then swap the planner for `TrajectoryOptimizer` (`plan_to_goal` with a `TaskSpaceTarget`, following the official tutorial) and compare latency and trajectory duration.

---

## 8. Use it in the real stack

* **Official 6.1 pick-and-place.** `standalone_examples/api/isaacsim.robot_motion.examples/manipulation/pick_place.py` composes `RmpFlowController` with gripper controllers inside a `PickPlaceController` (Franka or UR10 with suction). Its `--robot-config-dir` flag takes your own URDF + XRDF + `rmp_flow.yaml`. Read it after Lab 6c. It is the reactive counterpart to your planned version.
* **Isaac Lab** consumes the USD you validated here. Its asset configs (Lecture 09) point at a USD, and every drive gain, collider and inertia you fixed becomes the starting point for thousands of cloned copies. Isaac Lab 3.0 installs the importers through its `importers` extra.
* **ROS 2.** `isaacsim.ros2.urdf` imports from a live `robot_description`, and Lecture 08 drives the same articulation through `ros2_control` and MoveIt 2. Keep joint names identical across URDF, USD and XRDF. Every tool in the chain matches by name.
* **Robot learning.** Gains, armature and friction are part of the policy's environment. Changing them after training is a domain shift, and [Deep RL Lecture 13](../Deep%20RL%20for%20Robot%20Learning/Lecture-13.md) covers randomizing them on purpose.

---

## 9. Measure it

| Metric | How | Why it matters |
|---|---|---|
| **Collision shape count** (`num_shapes`) per `collision_type` | Lab 6a with Hull vs Decomposition | Predicts narrow-phase cost |
| **Physics ms/step** vs shape count | Lecture 01 timing on the imported robot, both collider variants | The price of collider accuracy |
| **Hold drift** (rad) and **root drift** (m) | Lab 6a hold test | Base type and gains are right |
| **Planning latency** p50 / p95, first call | Lab 6c, CPU vs CUDA world device | Fits a control loop or not; warm-up cost |
| **Trajectory duration** | `traj.duration` per phase | Cycle time; sensitive to velocity/acceleration limits |
| **Pick success, place error** | Lab 6c over ≥ 10 runs with jittered cube poses | Whether the whole chain works, not just one demo |
| **Peak VRAM** with planner running | `nvidia-smi` log during Lab 6c | What motion generation adds to your budget |

---

## 10. Ship it

Commit `robot_lab/` with:

* `import_and_validate.py`, `validation.json` for each robot and each `fix_base`/`collision_type` variant you tried
* `assemble.py` and the assembled robot USD (or a note on where the variant lives)
* `pick_place_cumotion.py`, `pick_cpu.json`, `pick_gpu.json`, and a short screen recording of one successful pick
* `ROBOT_NOTES.md`: every checklist failure you found and how you fixed it, the final gains and how you chose them, a table of collider variant vs shape count vs physics ms/step, and planning latency p50/p95 on CPU vs GPU

---

## Exit criteria

You can move on when you can:

* import a URDF with explicit `fix_base`, collider and drive settings, and explain what each did to the USD
* run the checklist on an unfamiliar robot and find at least one real problem in it
* attach a gripper with the Robot Assembler and explain where the result lives (payload file + variant set)
* derive gains from \( \omega_n \) and \( \zeta \), and verify them with the Gain Tuner's tests
* plan and execute a collision-free pick-and-place with cuMotion, and report its planning-latency distribution

---

## Self-check

1. A teammate imports a URDF arm with default settings, presses Play, and the arm topples off its pedestal. The arm itself looks fine. What happened, which importer field fixes it, and why does NVIDIA recommend setting that field explicitly for every robot class?
2. Your validation script reports every rigid body as "without colliders", but the viewport's collider view clearly shows hulls on every link. What is wrong with the script?
3. A joint limit reads `-170` from USD and `-2.97` from `get_dof_limits()`. Is the asset broken? Which layer uses which unit, and where do people get burned?
4. Switching all links from Convex Hull to Convex Decomposition makes grasps more stable but physics ms/step rises sharply. Explain the cost mechanism, and propose a collider layout that keeps most of the benefit.
5. Lab 6c's planning p50 is fine, but p95 is several times larger and the first call is slowest of all. Why both? What would you do before putting the planner inside a 30 Hz control loop?
6. You assembled a Robotiq gripper onto an arm, and the scene now has two articulation roots and jitters at rest. What went wrong, and what two checks from §3 would have caught it?

---

## References

* URDF and MJCF importers (options, multi-physics conversion, mimic joints, articulation root) — [URDF](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/importer_exporter/ext_isaacsim_asset_importer_urdf.html) · [MJCF](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/importer_exporter/ext_isaacsim_asset_importer_mjcf.html) · [Tutorial: Import URDF](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/importer_exporter/import_urdf.html)
* 6.0 robot asset pipeline migration (removed commands, tri-state `fix_base`, validation checklist) — [docs](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/migration_guides/isaac_sim_6_0/urdf_mjcf_importer_exporter_pipeline.html)
* Asset validation (SimReady profiles) and self-collision detector — [validation](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/robot_setup/asset_validation.html) · [collision detector](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/robot_setup/ext_isaacsim_robot_setup_collision_detector.html)
* Robot Assembler and the UR10e + 2F-140 manipulator tutorial — [assembler](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/robot_setup/assemble_robots.html) · [Tutorial 6: Setup a Manipulator](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/robot_setup_tutorials/tutorial_import_assemble_manipulator.html)
* Gain Tuner and System Identification — [gain tuner](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/robot_setup/ext_isaacsim_robot_setup_gain_tuner.html) · [sysid](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/robot_setup/ext_isaacsim_robot_setup_sysid.html)
* cuMotion integration — [overview](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/cumotion/index.html) · [robot configuration / XRDF](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/cumotion/tutorial_robot_configuration.html) · [world interface](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/cumotion/tutorial_world_interface.html) · [graph planner](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/cumotion/tutorial_graph_planner.html) · [trajectory optimizer](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/cumotion/tutorial_trajectory_optimizer.html)
* Motion generation migration (Lula → cuMotion / Pink) — [docs](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/migration_guides/isaac_sim_6_0/robot_motion_to_experimental_motion_generation.html) · cuRobo status — [docs](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/manipulators/manipulators_curobo.html)
* v6.1.0 source used for the labs — [`urdf_import.py`](https://github.com/isaac-sim/IsaacSim/blob/v6.1.0/source/standalone_examples/api/isaacsim.asset.importer.urdf/urdf_import.py) · [`rmpflow_follow_target.py`](https://github.com/isaac-sim/IsaacSim/blob/v6.1.0/source/standalone_examples/api/isaacsim.robot_motion.cumotion/rmpflow_follow_target.py) · [`pick_place.py`](https://github.com/isaac-sim/IsaacSim/blob/v6.1.0/source/standalone_examples/api/isaacsim.robot_motion.examples/manipulation/pick_place.py)

---

## Next in this special course

* Next: [Lecture 07 — Sensors, Rendering, and Synthetic Data](Lecture-07.md)
* Previous: [Lecture 05 — Newton and Multi-Physics](Lecture-05.md)
* Back: [Isaac Sim and Isaac Lab — Overview](README.md)
