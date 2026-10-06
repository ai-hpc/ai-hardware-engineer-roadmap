# Lecture 03: The Core Experimental API

## Overview

Lecture 01 used the Core Experimental API to drop cubes, and Lecture 02 used its backends to read poses. This lecture covers the API itself. It is the base API for Isaac Sim 6.x. The classic `isaacsim.core.api` (`World`, `SimulationContext`, `DynamicCuboid`, `SingleArticulation`) is deprecated, and most code you will find online still uses it.

The experimental API is shaped around one hardware fact: **a Python call has a fixed cost, and that cost is roughly the same whether the call moves one number or ten thousand.** So every wrapper is batched, takes regex paths, broadcasts its inputs, and returns Warp arrays on the simulation device. Used that way, Python overhead stays flat as the scene grows. Used like the classic API (one object per prim, `.numpy()` on everything), it is just as slow as before. This lecture teaches the design, the lifecycle (`SimulationManager`, `RenderingManager`, callbacks), articulation control, and Warp ↔ PyTorch interop, ports a classic script line by line, and then measures per-call overhead and copy cost on your card.

By the end you should be able to:

* explain the API's design rules (batched, regex paths, broadcast `(N, ...)` inputs, Warp outputs, selectable backends) and why each exists
* navigate `prims`, `objects`, `materials`, `utils`, and 6.1's new `actuators` and `primdata` extensions
* drive the simulation with `SimulationManager` and `RenderingManager`, including physics callbacks
* read and command articulations by DOF name and index
* move data between Warp, PyTorch and NumPy without accidental copies
* port a classic `World` script to Isaac Sim 6.1, and measure the batch-vs-loop and copy overheads on your GPU

---

## 1. Why it matters: Python is the slowest part of your loop

A training or control loop does the same few things every step: read state, compute, write commands, step physics. With GPU physics, the physics step for thousands of bodies can take less wall-clock time than the Python around it. The usual mistakes all add Python overhead per prim or per step:

| Pattern | Cost that scales badly | Fix in this API |
|---|---|---|
| One wrapper per robot or object | Python call overhead × N, every step | One wrapper over a regex path: `RigidPrim("/World/env_.*/cube")` |
| Per-element Python loops over results | Interpreter time × N | Vectorized ops on the returned array (Warp kernel or torch) |
| `.numpy()` on every result | GPU→CPU copy and a sync, every step | `wp.to_torch(...)`: zero-copy, stays on the GPU |
| Reading poses through `usd` | Per-prim USD transform computation (Lecture 02) | `tensor` backend after play |

---

## 2. Mental model: one wrapper, N prims

### 2.1 Design rules

1. **Batched by default.** A wrapper covers one or many prims. The experimental API has no `Single*` classes like the classic `SingleRigidPrim`.
2. **Regex paths.** `RigidPrim("/World/c_.*")` matches every prim the pattern resolves to, in a stable order. `wrapper.paths` tells you that order, and every row of every result follows it.
3. **Broadcast `(N, ...)` inputs.** Setters take `list | np.ndarray | wp.array` of shape `(N, ...)`, and smaller shapes broadcast with NumPy rules. `set_dof_position_targets(-1.5, dof_indices=[1])` sets one joint on all N robots.
4. **`indices=` selects prims, `dof_indices=` selects joints.** Both are optional, both accept arrays, and both avoid Python loops.
5. **Warp outputs on the simulation device.** Getters return `wp.array`s on the device the physics runs on (the wrapper picks it up from `SimulationManager` when it is created and again when physics starts). Convert explicitly when you need torch or NumPy (§5).
6. **Selectable backends.** `use_backend("tensor" | "usd" | "usdrt" | "fabric")` (Lecture 02). Each method documents the backends it supports, and the first one listed is the default.
7. **Engine-agnostic.** The same wrappers sit on PhysX or Newton tensors, so code written against them survives an engine switch (Lecture 05).

Shapes you will see constantly:

| Quantity | Shape | Notes |
|---|---|---|
| World positions / translations | `(N, 3)` | meters |
| Orientations | `(N, 4)` | **quaternion wxyz** (Isaac Lab 3.0 is xyzw) |
| Linear + angular velocities | `(N, 3)` each | `get_velocities()` returns a tuple |
| DOF positions / velocities / efforts | `(N, D)` | `D = num_dofs`, ordered as `dof_names` |
| Masses | `(N, 1)` | |
| Jacobians | `(N, links − 1, 6, D)` fixed base; `(N, links, 6, D + 6)` floating base | the official IK example slices `[:, ee_index - 1, :, :7]` |

### 2.2 Module map (6.1)

| Module | Contents | Notes |
|---|---|---|
| `isaacsim.core.experimental.prims` | `Prim`, `XformPrim`, `GeomPrim`, `RigidPrim`, `Articulation`, `DeformablePrim` | Note `XformPrim` (lowercase "form"); the classic one was `XFormPrim` |
| `isaacsim.core.experimental.objects` | `Cube`, `Sphere`, `Capsule`, `Cone`, `Cylinder`, `Plane`, `Mesh`, `GroundPlane`, lights, `Camera` | Each one creates prims if the paths don't exist and wraps them if they do |
| `isaacsim.core.experimental.materials` | `PreviewSurfaceMaterial`, `OmniPbrMaterial`, rigid-body physics material, `SurfaceDeformableMaterial`, `VolumeDeformableMaterial` | Bind with `apply_visual_materials` / `apply_physics_materials` |
| `isaacsim.core.experimental.utils` | `stage`, `prim`, `transform`, `backend`, `app`, `ops`, `semantics`, ... | `transform` Euler helpers take **[roll, pitch, yaw] (XYZ)** since 6.0, which silently breaks old ZYX code |
| `isaacsim.core.experimental.actuators` (new in 6.1) | `ArticulationActuators`, `ActuatorConfig`, USD authoring helpers | Newton actuator models (delay → controller → clamping); experimental |
| `isaacsim.core.experimental.primdata` (new in 6.1) | C++ provider of read-only prim, rigid-body and articulation data views | Not a Python API you call; it backs sensors and nodes |
| `isaacsim.core.simulation_manager` | `SimulationManager`, `SimulationEvent`, `PhysxScene`, `PhysicsScene` | Stepping, device, callbacks, engine switch |
| `isaacsim.core.rendering_manager` | `RenderingManager`, `RenderingEvent` | Rendering without stepping physics |

**Actuators (6.1).** `ArticulationActuators` wraps an `Articulation` and runs one or more Newton actuator pipelines per step: an optional delay, a controller (PD, PID, or neural), and clamping (max effort, DC motor, position-based). It registers a pre-physics callback, writes the resulting efforts to the articulation, and zeroes the USD drive gains on the DOFs it actuates, so the implicit PD doesn't fight it. Actuators can be authored in USD (`NewtonActuator` prims, so they travel with the asset) or built in Python through `ArticulationActuators.from_actuators(...)` with `ActuatorConfig` objects. This is how you model a real motor's delay and torque limits, which matters for sim-to-real (Lecture 12). The extension is marked experimental, so check its docs before relying on exact names.

**primdata (6.1).** This extension provides the C++ implementation of the `IPrimDataReader` interface: lazy, cached, read-only views of transforms, rigid bodies and articulations. On PhysX it fills buffers by calling the tensor API directly from C++, with no Python involved. You won't call it from a script. It explains why sensors and OmniGraph nodes in 6.1 can read state cheaply.

### 2.3 Lifecycle: authoring, play, tensor-valid

```text
 create stage / prims ──► wrap (USD backend works) ──► timeline play ──► physics ready ──► tensor backend valid
   stage_utils, objects      XformPrim, RigidPrim,      app_utils.play()   SIMULATION_SETUP   get_dof_positions,
                             Articulation(...)          + one update                          efforts, Jacobians...
```

Some methods work at authoring time through USD (poses, masses, gains, default state, all `usd` backend). Others exist **only** on the tensor backend: `get_dof_positions`, `get_dof_velocities`, `set_dof_positions`, `set_dof_efforts`, `get_dof_efforts`, `get_jacobian_matrices`, `get_mass_matrices`. Calling those before play fails with an assertion that the physics tensor entity is not valid. That message nearly always means you forgot `play()` and one update. One more trap: setters on the tensor backend write to the **engine**, not to USD. A mass you set after play is simulated but not saved with the stage.

---

## 3. SimulationManager and RenderingManager

`World` used to own stepping, rendering, resets and callbacks. In 6.x, `SimulationManager` (all class methods) and `RenderingManager` split those jobs between them:

| Job | Call | Notes |
|---|---|---|
| Set timestep and device | `SimulationManager.setup_simulation(dt=1/60, device="cuda:0")` | Creates a physics scene if none exists. `set_device("cuda:0")` switches to GPU broadphase + GPU dynamics and enables Fabric output. `get_device()` tells you what you got |
| Per-scene physics settings | `PhysxScene("/PhysicsScene").set_solver_type("TGS")`, `.set_dt(...)`, `.set_enabled_ccd(...)` (`/PhysicsScene` is where `setup_simulation` creates one) | `SimulationManager.set_physics_dt` and friends still work but are deprecated in 6.1 |
| Step physics only | `SimulationManager.step(steps=1, callback=None, update_fabric=False)` | No rendering. Pass `update_fabric=True` if something will read Fabric (renderer, `fabric` backend) |
| Render only | `RenderingManager.render()` | An app update with physics stepping suppressed |
| Both, plus UI | `simulation_app.update()` | What Lecture 01 timed as "frame ms" |
| Physics callbacks | `SimulationManager.register_callback(fn, SimulationEvent.PHYSICS_PRE_STEP)` | `fn(dt, context)`; also `PHYSICS_POST_STEP`; returns an id for `deregister_callback` |
| Lifecycle callbacks | `SimulationEvent.SIMULATION_SETUP`, `SIMULATION_STARTED`, `SIMULATION_PAUSED`, `SIMULATION_STOPPED`, `PRIM_DELETED` | Lifecycle hooks. There is no `POST_RESET` equivalent; resets are your code's job |
| Render callbacks | `RenderingManager.register_callback(RenderingEvent.NEW_FRAME, callback=fn)` | Fires once per rendered frame |
| Time | `SimulationManager.get_simulation_time()`, `get_num_physics_steps()` | Use these, not wall-clock, for control timing |
| Engine | `SimulationManager.switch_physics_engine("newton")` | One engine active at a time. Switch **before** starting the simulation (Lecture 05) |

Two details matter:

* **`IsaacEvents` is deprecated** in favor of `SimulationEvent`. You will still see `IsaacEvents.POST_PHYSICS_STEP` in 6.1's own examples, and it still works. Write new code with `SimulationEvent`.
* **Multi-tick only.** 6.0 removed the non-multi-tick stepping path. The packaged `SimulationApp` sets `/rtx/hydra/supportMultiTickRate` to true. If you ship a custom `.kit` file, don't override it to false.

Physics callbacks registered through `SimulationManager` fire only once the simulation view exists, that is, after play. Code that runs every physics substep (applying efforts, reading contact sensors) belongs in a `PHYSICS_PRE_STEP` or `PHYSICS_POST_STEP` callback. Code that runs once per policy step can stay in your main loop.

---

## 4. Articulations through the API

```python
from isaacsim.core.experimental.prims import Articulation
robots = Articulation("/World/env_.*/franka")       # resolves to prims with ArticulationRootAPI under each match
print(len(robots), robots.num_dofs, robots.dof_names)   # N, D, ['panda_joint1', ..., 'panda_finger_joint2']
arm = list(range(7))
grip = robots.get_dof_indices(["panda_finger_joint1", "panda_finger_joint2"])   # wp.array of indices

robots.set_default_state(dof_positions=[0.0, -0.57, 0.0, -2.81, 0.0, 3.04, 0.74, 0.04, 0.04])
# ... play + one update, then:
robots.reset_to_default_state()                     # tensor backend: teleport to the default state
q  = robots.get_dof_positions()                     # (N, D) wp.array, tensor backend only
dq = robots.get_dof_velocities(dof_indices=arm)     # (N, 7)
robots.set_dof_position_targets(q.numpy()[:, :7] + 0.01, dof_indices=arm)   # PD target, not a teleport
robots.set_dof_position_targets(0.0, dof_indices=grip)                       # broadcast: close all grippers
J  = robots.get_jacobian_matrices()                 # (N, links - 1, 6, D) for a fixed base
```

The control vocabulary:

| Intent | Call | What actually happens |
|---|---|---|
| Position control | `set_dof_position_targets(...)` | Sets the implicit PD drive's target. The joint gets there over several steps, as fast as stiffness, damping and max effort allow |
| Velocity control | `set_dof_velocity_targets(...)` | The same drive, tracking velocity |
| Effort control | `set_dof_efforts(...)` | Writes actuation forces. **Must be called every step.** Pair it with `switch_dof_control_mode("effort")`, which zeroes stiffness and damping |
| Teleport | `set_dof_positions(...)`, `set_dof_velocities(...)` | Overwrites state directly. Use it for resets, never as a controller |
| Gains | `set_dof_gains(stiffnesses, dampings)`, `get_dof_gains()` | The PD drive (Lecture 04). `switch_dof_control_mode(mode)` sets them by rule for `"position"`, `"velocity"` or `"effort"` |
| Limits | `set_dof_limits`, `set_dof_max_efforts`, `set_dof_max_velocities`, `set_dof_armatures` | Actuator realism (Lecture 04, Lecture 12) |
| Measured forces | `get_dof_projected_joint_forces()`, `get_link_incoming_joint_force()` | `get_dof_efforts()` reads back the actuation forces **you commanded**. It is not the drive's output |

In this example, `q.numpy()[:, :7] + 0.01` is fine for a demo, but it is the copy pattern §7 measures. In a hot loop, keep the math on the device (§5).

---

## 5. Warp, PyTorch, NumPy: where the copies are

| Conversion | Copy? | Notes |
|---|---|---|
| `wp.to_torch(a)` | **No** (zero-copy) | The tensor aliases the Warp array's memory on the same device. Clone it if you need the values after the array may change |
| `wp.from_torch(t)` | **No** | Use it to pass torch results back into setters |
| `a.numpy()` on a CPU array | No | Returns a view |
| `a.numpy()` on a CUDA array | **Yes** | Device → temporary buffer → host, and the host waits for it |
| Passing NumPy or a list to a setter on GPU | **Yes** (host → device) | Fine at reset time, wasteful every step |

The official 6.1 Franka IK example keeps the whole control law in torch:

```python
q   = wp.to_torch(robot.get_dof_positions())                       # zero-copy
J   = wp.to_torch(robot.get_jacobian_matrices())[:, ee_idx - 1, :, :7]
dq  = damped_least_squares(J, ...)                                  # torch on the sim device
robot.set_dof_position_targets(wp.from_torch(q[:, :7] + dq), dof_indices=list(range(7)))
```

This only stays on the GPU if physics runs there (`setup_simulation(device="cuda:0")` before you create the wrappers). On a CPU simulation, the wrapper outputs live on the CPU, `wp.to_torch` gives you CPU tensors, and a CUDA policy forces a copy every step. The copy is still there, just hidden in your own code.

**Quaternions across the boundary.** The experimental wrappers return **wxyz**. Underneath, the PhysX tensor view stores xyzw, and `RigidPrim.get_world_poses` reorders it for you. Isaac Lab 3.0 uses **xyzw** everywhere. When you move orientations between Isaac Sim scripts and Isaac Lab code, reorder explicitly (`q_xyzw = q_wxyz[:, [1, 2, 3, 0]]`) and test with a known 90° rotation.

---

## 6. Migrating from the classic API

| Classic (5.x, deprecated in 6.0) | Isaac Sim 6.1 |
|---|---|
| `from isaacsim.core.api import World` | `stage_utils.create_new_stage()` + `SimulationManager` + `RenderingManager` |
| `World(physics_dt=..., device=...)` | `SimulationManager.setup_simulation(dt=..., device=...)` |
| `world.scene.add_default_ground_plane()` | `GroundPlane("/World/ground")` |
| `world.scene.add(DynamicCuboid(prim_path=..., size=..., mass=...))` | `Cube(path, sizes=...)` + `GeomPrim(path, apply_collision_apis=True)` + `RigidPrim(path, masses=[...])` |
| `isaacsim.core.utils.stage.add_reference_to_stage(usd_path, prim_path)` | `isaacsim.core.experimental.utils.stage.add_reference_to_stage(usd_path=..., path=..., variants=[...])` |
| `SingleArticulation(prim_path)` + `world.scene.add(...)` | `Articulation(path)`; there is no scene registry, so keep your own references |
| `world.reset()` (initializes handles) | `app_utils.play()` + `simulation_app.update()`; `reset_to_default_state()` for resets |
| `world.add_physics_callback(name, fn(step_size))` (runs before each step) | `SimulationManager.register_callback(fn(dt, ctx), SimulationEvent.PHYSICS_PRE_STEP)` |
| `world.step(render=True)` | `simulation_app.update()` (or `SimulationManager.step()` + `RenderingManager.render()`) |
| `art.get_joint_positions()` → NumPy `(D,)` | `art.get_dof_positions()` → Warp `(N, D)` |
| `art.apply_action(ArticulationAction(joint_positions=q))` | `art.set_dof_position_targets(q)` |
| `prim.get_world_pose()` → NumPy, one prim | `prims.get_world_poses()` → Warp, batched |
| `isaacsim.core.utils.rotations.euler_angles_to_quat(...)` | `transform_utils.euler_angles_to_quaternion(...)`, with input order **[roll, pitch, yaw]** |
| `isaacsim.core.utils.nucleus` | `isaacsim.storage.native` |
| Residual reporting (`get_position_residual`, ...) | Removed, with no replacement |

**Read-only: the classic script you will find online.** This is 5.x code, shown so you can read it. Don't write new code like this:

```python
# classic_franka_cube.py — Isaac Sim 5.x classic Core API (deprecated in 6.0). READ-ONLY.
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})
import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction

world = World(physics_dt=1 / 60, rendering_dt=1 / 60)
world.scene.add_default_ground_plane()
cube = world.scene.add(DynamicCuboid(prim_path="/World/Cube", name="cube",
                                     position=np.array([0.5, 0.0, 0.1]), size=0.05, mass=0.1))
add_reference_to_stage(usd_path=FRANKA_USD_5X, prim_path="/World/Franka")   # a 5.x asset path
franka = world.scene.add(SingleArticulation(prim_path="/World/Franka", name="franka"))
world.reset()                                                  # creates physics handles
target = np.array([0.0, -0.57, 0.0, -2.81, 0.0, 3.04, 0.74, 0.04, 0.04])
world.add_physics_callback("hold", lambda step_size: franka.apply_action(ArticulationAction(joint_positions=target)))
for _ in range(600):
    world.step(render=False)
print(franka.dof_names, franka.get_joint_positions(), cube.get_world_pose())
simulation_app.close()
```

**The 6.1 port:**

```python
# isaac_bench/port_franka_cube.py — Isaac Sim 6.1, Core Experimental API
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})
import numpy as np
import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.experimental.objects import Cube, GroundPlane
from isaacsim.core.experimental.prims import Articulation, GeomPrim, RigidPrim
from isaacsim.core.simulation_manager import SimulationEvent, SimulationManager
from isaacsim.storage.native import get_assets_root_path

stage_utils.create_new_stage()
SimulationManager.setup_simulation(dt=1 / 60, device="cpu")    # set the device BEFORE creating wrappers
GroundPlane("/World/ground")
Cube("/World/Cube", sizes=0.05, positions=[[0.5, 0.0, 0.1]])
GeomPrim("/World/Cube", apply_collision_apis=True)
cube = RigidPrim("/World/Cube", masses=[0.1])
stage_utils.add_reference_to_stage(
    usd_path=get_assets_root_path() + "/Isaac/Robots_Multiphysics/FrankaRobotics/FrankaPanda/franka/franka.usda",
    path="/World/Franka")
franka = Articulation("/World/Franka")
target = np.array([[0.0, -0.57, 0.0, -2.81, 0.0, 3.04, 0.74, 0.04, 0.04]])  # (N=1, D=9)

def hold(dt, context):                                          # runs before every physics step
    franka.set_dof_position_targets(target)
cb = SimulationManager.register_callback(hold, SimulationEvent.PHYSICS_PRE_STEP)

app_utils.play(); simulation_app.update()                      # physics ready: tensor backend valid
SimulationManager.step(steps=600)
pos, quat = cube.get_world_poses()                              # Warp (1, 3), (1, 4) wxyz
print(franka.dof_names, franka.get_dof_positions().numpy(), pos.numpy(), quat.numpy())
SimulationManager.deregister_callback(cb); app_utils.stop()
simulation_app.close()
```

Line by line, the meaning didn't change. What changed is who owns each job (no `World`), the shapes (always a batch dimension), the return type (Warp, not NumPy), and the asset path (`Robots_Multiphysics`). The callback setting a constant target every step is deliberately naive. For a fixed target, setting it once is enough, and that is the first optimization in Lab 3a.

---

## 7. The hardware view: per-call overhead, batch size, copies

Model one wrapper call that touches \( N \) prims as a fixed cost \( t_0 \) (Python dispatch, argument broadcasting, index resolution, the tensor-view call, any kernel launch) plus a per-prim cost \( t_1 \):

$$
t_{\text{batch}}(N) \approx t_0 + N\,t_1, \qquad t_{\text{loop}}(N) \approx N\,(t_0 + t_1)
$$

On the GPU, \( t_1 \) is tiny because the per-prim work is parallel, so \( t_0 \) dominates until \( N \) is large. The loop version pays \( t_0 \) N times. That is the whole case for batching, and Lab 3b measures \( t_0 \) and \( t_1 \) for your card.

The second cost is **crossing the PCIe bus**. With GPU physics, `.numpy()` on a result does two things: it copies device → host, and it makes the CPU wait for the GPU to finish everything queued before it. The copy for a few thousand floats is small. The wait removes the overlap between Python and GPU work. Keeping data on the device (`wp.to_torch`, Warp kernels) avoids both. Copying once every k steps for logging is fine. Copying every step to feed a CPU policy is a design choice you should make on purpose, with its cost measured (Lab 3c).

| Choice | Moves | Scales with |
|---|---|---|
| N wrappers instead of 1 | CPU ms/step; one physics view + subscriptions per wrapper | N × \( t_0 \) |
| `.numpy()` every step on GPU | CPU ms/step, GPU idle time | Steps × (sync + copy) |
| NumPy inputs to setters on GPU | Host → device copy per call | Calls per step |
| `usd` backend at runtime | CPU ms/step | N × per-prim USD cost (Lecture 02) |
| Physics on CPU, policy on GPU | Two copies per step | Steps × state size |

> **8 GB budget.** The API choices in this lecture barely move VRAM. They move **CPU time**, and on a laptop-class CPU paired with a 4060, Python overhead is often the bottleneck before VRAM is. Use one wrapper per prim *type* (not per prim), physics on `cuda:0` when your policy is on the GPU, and `wp.to_torch` in the loop. Keep `.numpy()` for logging. Lab 3b's crossover \( N \) tells you when batching starts to matter. You won't need a bigger GPU for this lecture. If your measured \( t_0 \) is high, a faster CPU helps more than a 48 GB card does.

---

## 8. Build it

Extend `isaac_bench/` from Lecture 01. Every script prints a single CSV-ready line.

### Lab 3a — Port and verify

1. Run `port_franka_cube.py` above on `cpu` and on `cuda:0`. Confirm that the cube rests near z = 0.025 (half of 0.05). Record the arm joints' error against `target`: it should be small and steady, and any sag under gravity shows how stiff the asset's drives are (Lecture 04).
2. Remove the callback and set the target once after play. Results should match, and physics ms/step should drop slightly. Time both with `SimulationManager.step` in a timed loop, as in Lecture 01.
3. Port one more classic script of your choice from the 5.x `standalone_examples/` (or a forum snippet) using the §6 table. Keep the classic version in `isaac_bench/classic/` as read-only reference.

### Lab 3b — Batch vs loop sweep

```python
# isaac_bench/batch_vs_loop.py   usage: batch_vs_loop.py DEVICE(cpu|cuda:0)
import sys, time
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})
import numpy as np, warp as wp
import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.experimental.objects import Cube, GroundPlane
from isaacsim.core.experimental.prims import GeomPrim, RigidPrim
from isaacsim.core.simulation_manager import SimulationManager
device, reps, NMAX = sys.argv[1], 20, 1024

stage_utils.create_new_stage()
SimulationManager.setup_simulation(dt=1 / 60, device=device)
GroundPlane("/World/ground")
i = np.arange(NMAX)
shape = Cube([f"/World/c_{k}" for k in range(NMAX)], sizes=0.05,
             positions=np.stack([(i % 32) * 0.1, (i // 32) * 0.1, np.full(NMAX, 0.03)], 1))
GeomPrim(shape.paths, apply_collision_apis=True)
batch = RigidPrim("/World/c_.*")
singles = [RigidPrim(p) for p in shape.paths]                   # the anti-pattern, for measurement
app_utils.play(); simulation_app.update()

def timeit(fn):
    fn(); wp.synchronize_device(); t = time.perf_counter()
    for _ in range(reps):
        fn()
    wp.synchronize_device(); return (time.perf_counter() - t) / reps * 1e3

for n in (1, 4, 16, 64, 256, 1024):
    idx = wp.array(np.arange(n, dtype=np.int32), device=batch.get_world_poses()[0].device)
    ms_batch = timeit(lambda: batch.get_world_poses(indices=idx))
    ms_loop = timeit(lambda: [s.get_world_poses() for s in singles[:n]])
    print(f"{device},{n},{ms_batch:.4f},{ms_loop:.4f}")
simulation_app.close()
```

Run it on `cpu` and `cuda:0`. Plot ms/call against N (log-log) for both strategies, fit \( t_0 \) and \( t_1 \) from §7, and mark the N where the loop costs 10× the batch. Repeat with a setter (`set_velocities` with zeros) to see whether writes behave like reads. Expect the loop to grow linearly from the start and the batch to stay nearly flat at small N.

### Lab 3c — Zero-copy vs `.numpy()` in a stepping loop

Add to the Lab 3b scene (GPU only), with one `batch` wrapper over all 1,024 cubes:

```python
import torch                                                    # wp.to_torch needs torch installed
def step_zero_copy():                                           # stays on the GPU
    SimulationManager.step()
    p, _ = batch.get_world_poses()
    return wp.to_torch(p)[:, 2].mean()                          # a tiny "policy input" computation
def step_numpy():                                               # copy + sync every step
    SimulationManager.step()
    p, _ = batch.get_world_poses()
    return p.numpy()[:, 2].mean()
def step_only():
    SimulationManager.step()
for name, fn in (("step_only", step_only), ("zero_copy", step_zero_copy), ("numpy", step_numpy)):
    print(f"{name},{timeit(fn):.4f}")                          # timeit from Lab 3b (raise its reps from 20 to ~200 for stable numbers here)
```

Report ms/step for each variant at N = 1,024 and N = 64. The difference between `numpy` and `zero_copy` is the per-step price of crossing the bus. Then add a fourth variant that calls `.numpy()` only every 100 steps, and confirm its cost converges to `zero_copy`.

---

## 9. Use it in the real stack

* **Isaac Sim's official examples** for this API live in `standalone_examples/api/isaacsim.core.experimental.api/` (for instance `control_robot_torch.py`, `control_robot_warp.py`, `control_frankas.py`, `simulation_callbacks.py`), and the actuator examples are in `.../isaacsim.core.experimental.actuators/`. Read `control_robot_torch.py` next to §5.
* **Isaac Lab 3.0** (Lecture 09) follows the same principles one level up. Asset `.data.*` properties return a `ProxyArray` with `.torch` and `.warp` views, everything is batched over environments, and quaternions are **xyzw**. If you can write Lab 3c's zero-copy loop, you already know the Isaac Lab data model.
* **Deep RL for Robot Learning** ([Lecture 01](../Deep%20RL%20for%20Robot%20Learning/Lecture-01.md)) defines env-steps/s and the rollout regimes. Per-call overhead from this lecture is part of what caps env-steps/s when environments are cheap.

---

## 10. Measure it

| Metric | How | Why it matters |
|---|---|---|
| **\( t_0 \), \( t_1 \) per backend and device** | Lab 3b fit | Predicts the Python cost of any batch size |
| **Loop/batch crossover N** | Lab 3b | Where the anti-pattern becomes visible |
| **Copy cost per step** | Lab 3c: `numpy` − `zero_copy` | Price of a CPU-side policy or per-step logging |
| **Physics ms/step, callback vs set-once** | Lab 3a | Cost of per-step Python callbacks |
| **Port correctness** | Lab 3a final cube z and joint error | A port that runs is not a port that's right |

---

## 11. Ship it

Commit to `isaac_bench/`:

* `port_franka_cube.py`, `classic/classic_franka_cube.py` (read-only), and one more ported script, each with a two-line note on what changed
* `batch_vs_loop.py`, `batch_vs_loop.csv`, `batch_vs_loop.png` with fitted \( t_0 \) and \( t_1 \) for CPU and GPU physics
* `copies.csv`: ms/step for `step_only`, `zero_copy`, `numpy` and `numpy_every_100` at two values of N
* `API_NOTES.md`: your project's rules (one wrapper per prim type, device set before wrappers, no `.numpy()` in the hot loop, and the wxyz ↔ xyzw conversion helper you will reuse in Lectures 09-13)

---

## Exit criteria

You can move on when you can:

* write a standalone 6.1 script from scratch that creates, wraps, plays, steps and reads back a batch of rigid bodies and an articulation
* port a classic `World` / `DynamicCuboid` / `SingleArticulation` script and explain each change
* explain which methods need the tensor backend, and why they fail before play
* state your card's \( t_0 \) and the loop/batch crossover N, and the per-step cost of `.numpy()` at N = 1,024
* move orientations between Isaac Sim (wxyz) and Isaac Lab (xyzw) code without a sign or order bug

---

## Self-check

1. A script builds 256 `RigidPrim` wrappers, one per box, and reads each pose every step. It runs at a fraction of the expected rate even though physics ms/step is small. Rewrite the read in one line, and predict how the Python time changes with box count before and after.
2. `robot.get_dof_positions()` raises an assertion about the physics tensor entity, but `robot.get_world_poses()` works. Why do the two calls behave differently, and what is missing from the script?
3. Your torch policy runs on `cuda:0`, but the simulation was set up with `device="cpu"`. You use `wp.to_torch` everywhere and still see a copy every step. Where is it, and what is the fix?
4. You call `set_dof_efforts` once after reset and the arm collapses under gravity a few steps later. Explain it from the method's contract, and say what else you should configure for effort control.
5. A classic script built spawn orientations with Euler helpers that relied on the old ZYX `extrinsic=True` ordering. After porting to `transform_utils.euler_angles_to_quaternion` with the same numbers, robots spawn facing the wrong way and nothing raises an error. What changed in 6.0, and how do you fix the call?
6. You pass a gripper orientation read with `RigidPrim.get_world_poses()` straight into an Isaac Lab 3.0 command term, and the gripper rotates 180° about an odd axis. Diagnose it.

---

## References

* Core API overview (experimental API status) — [docs](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/python_scripting/core_api_overview.html)
* Core API → Core Experimental migration guide — [docs](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/migration_guides/isaac_sim_6_0/core_api_to_core_experimental.html)
* Core API tutorials — [Hello World](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/core_api_tutorials/tutorial_core_hello_world.html), [Hello Robot](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/core_api_tutorials/tutorial_core_hello_robot.html)
* API reference — [`isaacsim.core.experimental.prims`](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/py/source/extensions/isaacsim.core.experimental.prims/docs/index.html), [`isaacsim.core.experimental.utils`](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/py/source/extensions/isaacsim.core.experimental.utils/docs/index.html), [`isaacsim.core.simulation_manager`](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/py/source/extensions/isaacsim.core.simulation_manager/docs/index.html), [`isaacsim.core.rendering_manager`](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/py/source/extensions/isaacsim.core.rendering_manager/docs/index.html)
* Newton actuators tutorials (6.1) — [docs](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/newton_actuators_tutorials/index.html)
* Physics data flow (wrappers over USD / Fabric / tensors) — [docs](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/physics/new_physics_engine.html)
* Isaac Sim conventions (quaternion order per API) — [docs](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/reference_material/reference_conventions.html)
* Source at v6.1.0 — [`isaacsim.core.experimental.prims`](https://github.com/isaac-sim/IsaacSim/tree/v6.1.0/source/extensions/isaacsim.core.experimental.prims), [`isaacsim.core.simulation_manager`](https://github.com/isaac-sim/IsaacSim/tree/v6.1.0/source/extensions/isaacsim.core.simulation_manager), [experimental API standalone examples](https://github.com/isaac-sim/IsaacSim/tree/v6.1.0/source/standalone_examples/api/isaacsim.core.experimental.api)
* NVIDIA Warp interoperability (NumPy / PyTorch, zero-copy) — [docs](https://nvidia.github.io/warp/stable/user_guide/interoperability.html)

---

## Next in this special course

* Next: [Lecture 04 — PhysX for Robots](Lecture-04.md)
* Previous: [Lecture 02 — USD for Roboticists](Lecture-02.md)
* Back: [Isaac Sim and Isaac Lab — Overview](README.md)
