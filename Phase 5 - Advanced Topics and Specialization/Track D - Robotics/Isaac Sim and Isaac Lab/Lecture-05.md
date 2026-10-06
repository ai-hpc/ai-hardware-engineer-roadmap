# Lecture 05: Newton and Multi-Physics

## Overview

Until Isaac Sim 6.0, "physics" in Isaac Sim meant PhysX. Now there is a second engine: **Newton**, an open-source GPU physics engine built on NVIDIA Warp, with Google DeepMind's MuJoCo-Warp as its primary backend. It is experimental in Isaac Sim 6.1 and a selectable backend in Isaac Lab 3.0. It also runs on its own, without Kit and without an RTX renderer, which makes it the lightest way to do serious physics on an 8 GB card.

A second engine is not just a curiosity. It gives you different contact models, solvers for cloth and soft bodies, kit-less training, and a hard test of your policies: a policy that only works on one solver has learned that solver. This lecture explains what Newton is, how it plugs into Isaac Sim and Isaac Lab, which solvers it has, how its contacts differ from PhysX, what does and doesn't transfer between engines, and what each path costs on your GPU.

By the end you should be able to:

* explain Newton's architecture (Model / State / Control / Contacts + solver), its governance, and the versions each product pins
* switch an Isaac Sim 6.1 scene between PhysX and Newton, and list the asset constraints that make Newton refuse to start
* pick a Newton solver (MuJoCo-Warp, Featherstone, XPBD, VBD, MPM, Kamino) for a job, at the level the docs actually support
* contrast PhysX contacts, MuJoCo soft contacts, and hydroelastic contacts, and say what each costs
* run a PhysX-trained policy against Newton with the right checks (the PN/NP transfer matrix)
* run Newton kit-less, measure step time and memory against world count, and size a cloth demo for 8 GB

---

## 1. Why it matters: one simulator is one opinion

Every physics engine makes modeling choices: how contacts are generated, how friction cones are approximated, how stiff constraints are regularized, how state is represented. A policy trained on one engine can exploit those choices. Isaac Lab 3.0's transfer guide says it plainly: expect *similar* behavior across backends, not identical trajectories.

| Reason to use Newton | Where it applies | Caveat |
|---|---|---|
| **Kit-less simulation** (no Omniverse Kit, no RTX baseline) | Isaac Lab 3.0 `physics=newton_mjwarp`; standalone `newton` | No RTX sensors in that process; Newton has its own viewers |
| **MuJoCo models and semantics** on the GPU | MJCF assets, MuJoCo-style contacts | Assets tuned for PhysX may need retuning |
| **Deformables and particles**: cloth (VBD, XPBD, Style3D), granular materials (MPM) | Standalone Newton; Isaac Lab (experimental); Isaac Sim 6.1 adds VBD/XPBD | The Isaac Sim Newton tensor API does not cover deformables |
| **Area contacts** (hydroelastic) for grasping and insertion | Newton collision pipeline, added in Isaac Sim 6.1 | Opt-in per shape; needs SDF data; costs memory |
| **Cross-engine validation** of policies | Isaac Lab PP/PN/NN/NP matrix (§6) | Both backends must expose the same MDP |
| **Differentiability** | Standalone Newton (`wp.Tape`, `finalize(requires_grad=True)`) | Solver-dependent; not an Isaac Sim workflow |

---

## 2. Mental model: what Newton is

```text
 ┌──────────────── newton (Python package, Apache-2.0) ────────────────┐
 │  ModelBuilder ──finalize()──► Model   (static: bodies, shapes, joints, │
 │   add_body / add_shape_* /             materials, per-world layout)   │
 │   add_usd / replicate                                                 │
 │                                State ×2 (body_q, body_qd, joint_q...) │
 │  CollisionPipeline.collide(state) ─► Contacts                         │
 │  solver.step(state_in, state_out, control, contacts, dt)              │
 │     SolverMuJoCo │ SolverFeatherstone │ SolverXPBD │ SolverVBD │ ...   │
 ├──────────────────────────────────────────────────────────────────────┤
 │  NVIDIA Warp: Python-authored kernels, JIT-compiled to CUDA (or CPU)  │
 └──────────────────────────────────────────────────────────────────────┘
```

* **Governance.** Newton is a Linux Foundation project, started by Disney Research, Google DeepMind and NVIDIA. Its code is Apache-2.0. It extends and generalizes Warp's deprecated `warp.sim` and uses MuJoCo-Warp as its primary backend. It needs Python 3.10+ and runs on Linux, Windows, and macOS (CPU only). On the GPU it needs an NVIDIA card (Maxwell or newer) with a driver of 545 or newer; no local CUDA toolkit is needed.
* **Backend vs solver.** Isaac Lab's terminology is worth adopting. A *backend* (PhysX, Newton, OvPhysX) owns the simulation lifecycle and data exchange. A *solver* is the numerical method inside a backend. Newton is not the same thing as MJWarp: it hosts MJWarp, Kamino, VBD, MPM and others, and **solver settings are not numerically portable between solver families.**
* **Quaternions.** Warp's `wp.quat` stores **(x, y, z, w)**, the same order as Isaac Lab 3.0. The Isaac Sim experimental prim API uses **wxyz**. Convert when you move poses between a standalone Newton script and Isaac Sim wrappers.

**Versions in play (October 2026):**

| Component | Version | Source |
|---|---|---|
| Newton bundled with Isaac Sim 6.1 | 1.5.0 (with MuJoCo / mujoco-warp 3.11.0) | Isaac Sim 6.1 release notes |
| Newton in Isaac Lab 3.0.0-EA | 1.5.2 | v3.0.0-EA release notes |
| Newton on PyPI (latest) | 1.6.0 | newton releases / PyPI |
| MuJoCo-Warp (latest upstream) | 3.14.0 | google-deepmind/mujoco_warp |

The Isaac Sim docs warn that installing a different Newton, or an editable Newton checkout, into Isaac Sim's Python environment can break it. **Use a separate virtual environment for standalone Newton**, pinned to the version you want to compare against (`newton[examples]==1.5.2` to match Isaac Lab 3.0).

---

## 3. Solvers, at the level the docs support

Newton's solver docs organize the choice around coordinates. **Generalized (reduced) coordinates** (`SolverMuJoCo`, `SolverFeatherstone`) parameterize an articulation by its joint values, like PhysX articulations. **Maximal coordinates** (`SolverXPBD`, `SolverSemiImplicit`, `SolverKamino`) give every body a full pose and enforce joints as constraints.

| Solver (`newton.solvers.*`) | Integration | What it covers | Status / notes |
|---|---|---|---|
| `SolverMuJoCo` (MuJoCo-Warp) | Explicit, semi-implicit, implicit-in-velocity | Rigid bodies, articulations; equality and mimic constraints; joint friction and effort limits | Primary, validated path. Uses MuJoCo's built-in collision by default (`use_mujoco_contacts=False` switches to Newton's pipeline) |
| `SolverFeatherstone` | Semi-implicit | Rigid bodies, articulations (generalized), particles, soft bodies | Basic differentiability |
| `SolverXPBD` | Implicit (position-based) | Rigid bodies, articulations (maximal), particles, cloth (no self-collision), experimental soft bodies | Added to Isaac Sim in 6.1 |
| `SolverVBD` (Vertex Block Descent) | Implicit | Cloth, soft bodies, particles; rigid bodies with limited joint support (AVBD) | **Experimental**; added to Isaac Sim in 6.1 |
| `SolverImplicitMPM` | Implicit | Particles only (granular, snow, viscous) | Isaac Lab lists MPM as experimental |
| `SolverKamino` | Semi-implicit (Euler, Moreau-Jean) | Maximal-coordinate rigid mechanisms **with kinematic loops** and hard frictional contacts | **Experimental** in Newton; the Isaac Lab EA notes call it experimental, the backend docs a beta path |
| `SolverSemiImplicit`, `SolverStyle3D` | Semi-implicit / implicit | Basic rigid and particle work; Style3D for garment cloth | Specialist |

Two consequences for robotics:

* **Closed loops.** Isaac Sim's Newton integration rejects closed kinematic chains ("Joint graph contains a cycle"), and so do PhysX articulations. Kamino is the Newton solver aimed at loops. Use it only once you have read its guide.
* **Coupling.** Newton can partition one model between solvers, for example an MJWarp robot interacting with VBD cloth or an MPM material, using **proxy** or **ADMM** coupling. Isaac Lab exposes this as experimental, through `isaaclab_contrib.coupling`.

---

## 4. Contacts: three models

| | PhysX | MuJoCo-Warp | Hydroelastic (Newton pipeline) |
|---|---|---|---|
| Contact generation | Points within contact offset; friction per **patch** (≤ 2 anchors) | Points from MuJoCo collision (or Newton's pipeline) | A contact **patch** extracted from overlapping SDF "compliant layers" |
| Constraint | Rigid, with rest offset; position and velocity iterations | **Soft** constraints solved as an optimization: small, controlled penetration is part of the model; elliptic or pyramidal friction cone; `impratio` | Per-triangle force contributions; a stiffness feeds MuJoCo/penalty solvers (XPBD uses only the geometry) |
| CCD | Scene + body flag, **CPU only** | None. `ccd_iterations` is a GJK/EPA convergence cap, *not* a CCD switch | — |
| Capacity | Fixed GPU buffers (`PhysxGpuCfg`) | Per-world `nconmax` (contacts) and `njmax` (constraint rows), preallocated | Collision-pipeline buffers with resize multipliers |
| Best at | Broadly validated reference | Locomotion, MuJoCo-style articulated control | Flat-on-flat, conforming, large-area contacts: grasping, insertion |

**Hydroelastic contact** (added in Isaac Sim 6.1, opt-in). Each participating shape stores a signed distance field \( \phi \) and a stiffness \( k \). In the overlap region, Newton defines a pressure-balance field whose zero set is the contact patch: \( k_a \phi_A = k_b \phi_B \), so the softer shape deforms further. Newton extracts the patch as triangles with marching cubes, gives each triangle a weight \( w = A(-\delta) \) along its normal, and hands the solver a pair stiffness in series:

$$
k_{\text{eff}} = \frac{k_a\,k_b}{k_a + k_b}, \qquad c = A\,k_{\text{eff}}, \qquad f_n \approx c\,d
$$

Requirements:

* **Both** shapes must opt in, through `NewtonSDFCollisionAPI` in USD.
* Both must carry valid SDF data. Planes, heightfields, and non-watertight meshes are not supported.
* The runtime pipeline must enable the hydroelastic path (`HydroelasticConfig` on the Newton config).

The SDF is stored sparsely: a narrow high-resolution band around the surface, so memory scales with surface area, not volume. Use hydroelastic contact for grasping, pushing and insertion. Keep default contacts for locomotion, planes, and speed.

**Porting implication** (from Isaac Lab's solver-differences and transfer guides): equal friction coefficients do not give equal slip or grasp stability, and Isaac Lab's Newton friction randomization currently uses **one** coefficient where PhysX has static and dynamic. Revalidate contacts before you tune friction.

---

## 5. Newton inside Isaac Sim 6.1

**Turning it on.**

* The workstation build ships `./isaac-sim.newton.sh`, which enables Newton and disables PhysX. You can also use the engine selector at the top left of the viewport.
* In a standalone script, enable `isaacsim.physics.newton` and `isaacsim.physics.newton.tensors` yourself, then call `SimulationManager.switch_physics_engine("newton")` **before** play.
* `SimulationManager.get_available_physics_engines()` lists what is registered. Only one engine is active at a time.
* A switch deactivates the old engine, invalidates simulation views, and reloads physics from USD.

**Scene configuration** lives on the physics scene prim through schemas:

| Class (`isaacsim.core.simulation_manager`) | Applies | Configures |
|---|---|---|
| `PhysicsScene` | `NewtonSceneAPI` (applied to every physics scene) | Gravity, `newton:timeStepsPerSecond`, `newton:maxSolverIterations` |
| `NewtonMjcScene` | + `MjcSceneAPI` | `set_dt` (`mjc:option:timestep`), `set_integrator` (euler / rk4 / implicit / implicitfast), `set_solver` (pgs / cg / newton), `set_iterations`, `set_tolerance`, `set_cone`, `set_impratio` |
| `NewtonXpbdScene` | + `NewtonXpbdSceneAPI` | Relaxation and compliance for joints and contacts, angular damping, restitution |

A `NewtonVbdScene` class exists in the 6.1 source but is commented out of the public exports "until VBD schema release". The Newton runtime config (`isaacsim.physics.newton.acquire_stage().cfg`) holds the solver config. Its MuJoCo defaults in 6.1 are `nconmax=200`, `njmax=1200`, and `use_mujoco_contacts=False` (Newton's collision pipeline), alongside `num_substeps` and `collision_cfg` / `HydroelasticConfig`. If stdout prints *"Number of Newton contacts (N) exceeded MJWarp limit"*, contacts are being dropped: raise `nconmax` before play.

**What breaks** when a PhysX-tuned asset meets Newton (from the 6.1 docs):

* `physics:body0` / `body1` swapped on a joint ("Reversed joints are not supported")
* closed kinematic loops
* bodies with zero or tiny mass or inertia
* zero-size collision shapes
* joints outside any `ArticulationRootAPI`
* USD composition errors that PhysX ignores
* negative scales on colliders
* `metersPerUnit` ≠ 1 (Newton does not convert units)
* a stage with no joints at all, for MuJoCo

**Behavioral gaps:**

* Newton ignores USD edits made **while playing** (no change listener). Write through the tensor API or the experimental wrappers instead.
* The `primdata` reader (Lecture 03) leaves measured `dof_efforts` unavailable on Newton, because Newton does not expose projected joint forces.
* The Gain Tuner shows Newton gains in **radians**, where PhysX shows degrees.
* The first run of an asset JIT-compiles Warp kernels, which can take up to about a minute. The kernels are cached after that.
* The backend is tested only on a short list of robots (G1, H1, T1, UR5e, Allegro, Shadow Hand). The Franka from Lecture 04 is not on it.

> **You will see this in older code:** 6.0-era examples import `NewtonMjcScene` from `isaacsim.core.simulation_manager.impl.mjc_scene` and still use the deprecated `isaacsim.core.utils.stage.add_reference_to_stage`. In 6.1, import the class from `isaacsim.core.simulation_manager` and use `isaacsim.core.experimental.utils.stage`.

**Deformables in 6.x.** Isaac Sim 6.0 replaced the old PhysX deformables with separate **surface** (cloth-like) and **volume** (solid) deformables. The experimental API exposes them as `DeformablePrim(paths, deformable_type="surface"|"volume")` with `SurfaceDeformableMaterial` and `VolumeDeformableMaterial`. Legacy particle cloth and the old deformable wrappers are error stubs. These deformables run on **PhysX**: the Newton tensor API in Isaac Sim covers rigid bodies, articulations, and rigid contacts only. Newton's own cloth and soft-body solvers live in standalone Newton and in Isaac Lab's experimental VBD and coupled paths.

---

## 6. Newton in Isaac Lab 3.0 and what transfers

Isaac Lab 3.0 selects backends with a preset: `physics=isaacsim_physx` (the established reference), `physics=newton_mjwarp` (Newton, **beta** in the backend docs, Warp-native, runs **without Isaac Sim**), or `physics=ovphysx` (experimental kit-less PhysX). Backend support is task-specific: `uv run isaaclab train --rl_library rsl_rl --task <T> --help` lists the presets a task accepts. Configuration classes are `isaaclab_physx.physics.PhysxCfg` and `isaaclab_newton.physics.NewtonCfg` with `MJWarpSolverCfg`. Under Newton, each physics tick runs `num_substeps` solver substeps.

**Transferring a checkpoint between engines** (Isaac Lab's how-to). The checkpoint maps ordered observations to actions and contains no physics, so transfer works only if both backends present the **same MDP**. The guide's checklist:

| Contract | Must match exactly |
|---|---|
| Actions | Term order, ordered joint names, width, scale, offset, clipping |
| Observations | Term order, widths, history, units, frames, noise |
| Timing | Physics `dt`, decimation, policy period (Newton substeps may differ inside it) |
| Mechanism | Body/joint **ordering**, active DOFs, mimic coupling |
| Episode | Resets, commands, rewards, terminations, horizon |

Then evaluate the **full matrix**: PP (PhysX→PhysX), PN (PhysX-trained on Newton), NN, and NP.

* **Ordering.** Branched robots can order joints and bodies differently on each backend; set `env.scene.robot.joint_ordering=physx` and `env.scene.robot.body_ordering=physx` for PN, or `mjwarp` for NP. The signature of a scrambled axis is a locomotion policy that falls within a few dozen steps in every environment.
* **Actuators.** Raise damping to suppress bang-bang control. Armature matters in MJWarp. Drive only the leader finger of a mimic pair.
* **Domain randomization.** Randomize gains, friction, armature and mass to stop the policy overfitting one solver.
* **Validated tasks.** The guide lists `Isaac-Lift-Franka` (no ordering override needed), `Isaac-Velocity-Rough-G1` and `Isaac-Velocity-Rough-AnymalD` (ordering overrides needed).

See [Deep RL Lecture 13](../Deep%20RL%20for%20Robot%20Learning/Lecture-13.md) for evaluation with seeds and confidence intervals.

---

## 7. The hardware view: Warp kernels, graphs, and preallocated worlds

**Warp JIT.** Newton's kernels are Python functions that Warp compiles to CUDA the first time they run. Recompilation is triggered by things like the asset's DOF count. The first run of a new scene pays CPU compile time, and later runs hit the cache. Always exclude a warm-up from timings, and run each configuration twice.

**CUDA graphs.** One simulated frame is `substeps × (collide + solver kernels)`, which is many small launches. Newton's examples record one frame with `wp.ScopedCapture()` and replay it with `wp.capture_launch(graph)`, which removes the Python and launch overhead per kernel. At small world counts this overhead can dominate, so time both ways (Lab 5b).

**No RT cores needed for physics.** Newton's requirements are an NVIDIA GPU (Maxwell+) or a CPU. Kit-less Newton has no Kit or RTX baseline, so the whole card minus the desktop is yours. That is the opposite of Lecture 01's fixed cost. Data-center GPUs that cannot render Isaac Sim (A100, H100) are not excluded by Newton's stated requirements for physics-only work.

**Memory is preallocated per world.**

* MuJoCo-Warp reserves `nconmax` contacts and `njmax` constraint rows *per environment*. VRAM scales with **worlds × capacity**, not with the contacts actually present.
* Hydroelastic pipelines add buffers with explicit resize multipliers. Sparse SDFs scale with surface area.
* Size every capacity from the measured busiest reset state, as with PhysX buffers in [Lecture 04](Lecture-04.md). An undersized buffer drops contacts silently, apart from a warning. An oversized one is pure fixed cost.

**Inside Isaac Sim**, switching engines changes the physics column only. Kit, Fabric sync and RTX still cost what Lecture 01 measured.

> **8 GB budget.** Kit-less Newton is the friendliest path on a 4060. State-only RL and physics experiments get nearly the whole card, so prefer `physics=newton_mjwarp` without the `isaacsim` extra for physics-only work, and `--viewer null` / `--viz none` for timing runs.
>
> * Keep `nconmax`/`njmax` at measured need times headroom, not "large".
> * Cloth cost grows with particle count, roughly width × height of the grid, so a 256×128 sheet is far heavier than 64×32.
> * Don't run the Isaac Sim GUI and a standalone Newton benchmark at the same time.
>
> Coupled rigid–MPM scenes with large particle counts, or hydroelastic contact on many high-resolution SDFs alongside RTX cameras, are the cases to move to a cloud L40S / RTX PRO 6000.

---

## 8. Build it

### Lab 5a — The same scene on PhysX and Newton (Isaac Sim 6.1)

Extend Lecture 04's `physics_lab/box_stack.py` with an `--engine` flag. Only the scene setup changes, and the PhysX-only block gets a guard:

```python
# physics_lab/box_stack.py — additions (Lecture 05)
p.add_argument("--engine", choices=["physx", "newton"], default="physx")
p.add_argument("--mjc-iters", type=int, default=None)     # MuJoCo constraint-solver iterations
p.add_argument("--nconmax", type=int, default=None)       # MJWarp contact capacity
# ... after SimulationApp(...):
import isaacsim.core.experimental.utils.app as app_utils
if args.engine == "newton":                                # not auto-loaded outside isaac-sim.newton.sh
    app_utils.enable_extension("isaacsim.physics.newton")
    app_utils.enable_extension("isaacsim.physics.newton.tensors")
# ... replace the scene block:
stage_utils.create_new_stage()
if args.engine == "newton":
    from isaacsim.core.simulation_manager import NewtonMjcScene
    assert SimulationManager.switch_physics_engine("newton")   # before building, before play
    scene = NewtonMjcScene("/World/PhysicsScene")
    scene.set_dt(1.0 / args.hz)
    if args.mjc_iters: scene.set_iterations(args.mjc_iters)
    if args.nconmax:
        import isaacsim.physics.newton as newton_ext
        ns = newton_ext.acquire_stage()                    # Newton runtime config (see the 6.1 Newton page)
        if ns is not None: ns.cfg.solver_cfg.nconmax = args.nconmax
        # Newton initialization may replace solver_cfg on play: print ns.cfg.solver_cfg.nconmax after
        # play to confirm the value stuck (or set it through isaacsim.physics.newton.configure_newton).
else:
    scene = PhysxScene("/World/PhysicsScene")
    scene.set_steps_per_second(args.hz)
    scene.set_solver_type(args.solver)
# ... wrap the per-body PhysxSchema iterations, set_device and GPU-buffer code in:
#     if args.engine == "physx":
# ... and before t = time.perf_counter(), warm up (Newton JIT-compiles on first steps):
SimulationManager.step(steps=10)
```

Run the Lecture 04 baseline (`--height 10 --hz 60`) and a stable variant on both engines, each twice, keeping the second run. Then repeat with `--stacks 16 --height 4` (64 boxes in one world). On Newton, watch stdout for the MJWarp contact-limit message, and rerun with a larger `--nconmax` until it stops. Compare these columns:

* **ms/step:** which engine is faster for this scene, and how it changes with box count
* **sink:** MuJoCo's soft contacts settle with a small steady penetration, so a non-zero `sink` is expected and is not a bug
* **drift, fallen, and residual speed:** does the stack stand on both engines?

Write down which knobs had no equivalent. PhysX position iterations are not MuJoCo iterations, and Isaac Lab's guide says not to translate them numerically.

Optional: run Lecture 04's `franka_hold.py` under Newton with the same pattern. Expect to retune, since the Franka is not on Newton's tested list. Record what fails first: asset load, drive behavior, or grasp.

### Lab 5b — Kit-less Newton: worlds sweep

Use a separate environment so you don't disturb Isaac Sim's pinned packages:

```bash
python3.12 -m venv ~/env_newton && source ~/env_newton/bin/activate
pip install "newton[examples]==1.5.2"           # match Isaac Lab 3.0-EA's Newton
python -m newton.examples --list
python -m newton.examples basic_shapes           # GL viewer: XPBD by default, --solver vbd also
python -m newton.examples pyramid --benchmark --num-pyramids 4 --pyramid-size 10
python -m newton.examples basic_shapes --viewer usd --output-path shapes.usd   # open it in Isaac Sim
```

Then the benchmark you own:

```python
# newton_lab/stack_bench.py — W independent worlds, each an H-box stack; XPBD or MuJoCo-Warp
import argparse, time
import numpy as np, warp as wp, newton

p = argparse.ArgumentParser()
p.add_argument("--solver", choices=["xpbd", "mujoco"], default="mujoco")
p.add_argument("--worlds", type=int, default=1)
p.add_argument("--height", type=int, default=4)
p.add_argument("--fps", type=int, default=60)            # control rate: one "env step" per frame
p.add_argument("--substeps", type=int, default=4)        # keep even: states ping-pong back to s0
p.add_argument("--frames", type=int, default=300)
p.add_argument("--nconmax", type=int, default=None)      # MuJoCo-Warp contacts per world
p.add_argument("--no-graph", action="store_true")
p.add_argument("--device", default=None)                 # cpu, cuda:0
args = p.parse_args()
if args.device: wp.set_device(args.device)

H = 0.05
stack = newton.ModelBuilder()
stack.default_shape_cfg.mu = 0.8
stack.add_ground_plane()
for i in range(args.height):                             # add_body = body + free joint + articulation
    b = stack.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, H + i * 2.02 * H), q=wp.quat_identity()))
    stack.add_shape_box(b, hx=H, hy=H, hz=H)
builder = newton.ModelBuilder()
builder.replicate(stack, world_count=args.worlds)        # each copy is its own world
model = builder.finalize()

if args.solver == "xpbd":
    solver = newton.solvers.SolverXPBD(model, iterations=10)
    pipeline = newton.CollisionPipeline(model); contacts = pipeline.contacts()
else:
    solver = newton.solvers.SolverMuJoCo(model, nconmax=args.nconmax)  # MuJoCo's own collision
    pipeline, contacts = None, None
s0, s1, control = model.state(), model.state(), model.control()
newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
dt = 1.0 / (args.fps * args.substeps)

def frame():
    global s0, s1
    for _ in range(args.substeps):
        s0.clear_forces()
        if pipeline is not None: pipeline.collide(s0, contacts)
        solver.step(s0, s1, control, contacts, dt)
        s0, s1 = s1, s0

graph = None
if wp.get_device().is_cuda and not args.no_graph:
    with wp.ScopedCapture() as cap:                      # record one frame of launches
        frame()
    graph = cap.graph
def run(n):
    for _ in range(n):
        if graph: wp.capture_launch(graph)
        else: frame()

run(10); wp.synchronize()                                # warm-up (and JIT on first run)
t = time.perf_counter(); run(args.frames); wp.synchronize()
ms = (time.perf_counter() - t) / args.frames * 1e3
q = s0.body_q.numpy().reshape(args.worlds, args.height, 7)   # px py pz qx qy qz qw
drift = np.linalg.norm(q[:, :, :2], axis=2).max()
top_err = np.abs(q[:, -1, 2] - (H + (args.height - 1) * 2.02 * H)).max()
print(f"{args.solver},{args.worlds},{args.height},{wp.get_device()},graph={graph is not None},"
      f"ms/frame={ms:.3f},env_steps/s={args.worlds / (ms / 1e3):.0f},"
      f"rtf={(1 / args.fps) / (ms / 1e3):.1f},drift={drift:.4f},top_err={top_err:.4f}")
```

Steps:

1. Sweep `--worlds` ∈ {1, 16, 256, 1024, 4096} for both solvers, logging `nvidia-smi` throughout.
2. Repeat the small counts with `--no-graph` and with `--device cpu`.
3. Plot env-steps/s and VRAM against worlds.

What to expect, qualitatively:

* At small world counts, fixed launch overhead dominates, so the graph matters most.
* Throughput grows with worlds until the GPU saturates.
* VRAM grows linearly with worlds, with a slope set by capacities such as `nconmax` more than by the four boxes.

Rerun one point with `--nconmax` doubled and see the slope change.

### Lab 5c — Cloth on 8 GB

```bash
for res in "32 16" "64 32" "128 64" "256 128"; do set -- $res
  for s in vbd xpbd; do
    python -m newton.examples cloth_hanging --solver $s --width $1 --height $2 --benchmark --num-frames 600
  done
done
```

Record the FPS the benchmark reports and the peak VRAM from `nvidia-smi`, then plot both against particle count. Watch the largest stable run once with the GL viewer, without `--benchmark`, and check it for stretching, jitter, and interpenetration. Then try `python -m newton.examples cloth_franka` and `softbody_hanging` and note their peak VRAM. That is the cost of rigid–deformable interaction relative to cloth alone.

---

## 9. Use it in the real stack

* **Kit-less training (Isaac Lab 3.0):** `uv run isaaclab train --rl_library rsl_rl --task Isaac-Cartpole-Direct physics=newton_mjwarp --viz none`. Add `--extra isaacsim` only when you need the full Isaac Sim stack. [Lecture 09](Lecture-09.md) and [Lecture 10](Lecture-10.md) measure kit vs kit-less throughput properly.
* **Sim-to-sim check before sim-to-real:** train on PhysX, play the checkpoint with `physics=newton_mjwarp` (plus the ordering overrides for branched robots), and report all four cells of PP/PN/NN/NP.
* **Hydroelastic grasping:** Newton's `robot_panda_hydro` and `nut_bolt_hydro` examples, and Isaac Sim's *Configure Hydroelastic Contact for a Nut-and-Bolt Assembly* walkthrough, are the reference setups for area contacts.
* **Preparing assets:** Isaac Lab's *Prepare an Asset for Newton with MJWarp* separates common USD Physics properties from PhysX-only and MuJoCo-only ones. The 6.1 importers generate multiphysics assets that carry both ([Lecture 06](Lecture-06.md)).

---

## 10. Measure it

| Metric | How | Why it matters |
|---|---|---|
| **Physics ms/step, PhysX vs Newton** | Lab 5a, warm second run | Engine cost on the same scene inside Isaac Sim |
| **Behavior deltas** (sink, drift, fallen) | Lab 5a | Contact-model differences you must validate, not tune away |
| **Contact-capacity overflows** | MJWarp stdout message; `[PhysX]` warnings | Dropped contacts invalidate every other number |
| **env-steps/s vs worlds** | Lab 5b | Kit-less throughput curve; where your GPU saturates |
| **Graph speedup** | Lab 5b `--no-graph` | How much launch overhead the scene carries |
| **VRAM per world** | Lab 5b slope; `nvidia-smi` | Preallocated capacity cost; how many worlds fit in 8 GB |
| **Cloth FPS and VRAM vs particles** | Lab 5c | Largest deformable that fits next to your other work |

---

## 11. Ship it

Commit to `physics_lab/` and `newton_lab/`:

* `box_stack.py` with `--engine`, and `engine_compare.csv` (both engines, both configurations, warm runs)
* `stack_bench.py`, `worlds_sweep.csv`, `worlds_sweep.png` (env-steps/s and VRAM against worlds, both solvers, graph on/off)
* `cloth_sweep.csv` and `cloth_sweep.png`
* `NEWTON_NOTES.md`, covering:
  * versions used: Isaac Sim's bundled Newton vs your venv
  * the asset constraints you hit
  * which Lecture 04 knobs had no Newton equivalent
  * the VRAM per world at your chosen `nconmax`
  * a recommendation: for your capstone task, which backend do you train on, and which do you validate on?

---

## Exit criteria

You can move on when you can:

* draw Newton's Model / State / Control / Contacts / solver loop and explain backend vs solver
* switch a standalone Isaac Sim script to Newton, configure `NewtonMjcScene`, and fix a contact-capacity overflow
* name four asset properties that make Newton refuse a PhysX-tuned asset
* explain why MuJoCo contacts show steady penetration, what hydroelastic contact adds, and what `ccd_iterations` is *not*
* describe the PP/PN/NN/NP transfer matrix and the contract that must match before PN means anything
* report kit-less env-steps/s and VRAM per world on your card, and the largest cloth you can run comfortably

---

## Self-check

1. A teammate installs `pip install -U newton` into the Isaac Sim 6.1 environment to "get the latest Newton", and Isaac Sim's Newton backend starts failing. Why, and how should they have set this up?
2. An imported quadruped runs on PhysX, but Newton refuses to start with "Reversed joints are not supported". What is wrong in the USD, and why doesn't PhysX complain?
3. Your PhysX-trained ANYmal-D policy falls within a few dozen steps in every environment when played with `physics=newton_mjwarp`. What do you check first, and what would gradual degradation instead of a sudden fall suggest?
4. Lab 5a shows Newton's box stack resting 1-2 mm lower than PhysX's. Is that a bug? Which parameter family would you inspect, and why is "match PhysX's rest offset" the wrong framing?
5. Lab 5b at 4,096 worlds uses far more VRAM than the bodies alone would suggest, and the slope doubles when you double `nconmax`. Explain, and say how you would choose `nconmax` for a training run.
6. You need a cloth towel that a Franka picks up, simulated alongside RL training on an 8 GB card. Which engine and solver path do the docs support today, what status does it have, and what do you measure before you commit?

---

## References

* Isaac Sim 6.1 — [Newton Physics Backend](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/physics/newton_physics.html) (switching, scene classes, limitations, tested robots)
* Isaac Sim 6.1 — [Physics Data Flow and Engine Integration](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/physics/new_physics_engine.html)
* Isaac Sim 6.1 — [Hydroelastic Contact](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/physics/hydroelastic_contact.html)
* Isaac Sim 6.1 — [release notes](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/overview/release_notes.html) (Newton 1.5.0, VBD/XPBD, hydroelastic contacts, `nconmax` config)
* Isaac Sim v6.1.0 source — [`NewtonMjcScene`](https://github.com/isaac-sim/IsaacSim/blob/v6.1.0/source/extensions/isaacsim.core.simulation_manager/python/impl/mjc_scene.py)
* Newton — [GitHub (v1.5.2)](https://github.com/newton-physics/newton/tree/v1.5.2), [Solvers guide (feature matrix)](https://newton-physics.github.io/newton/1.5.2/solvers/index.html), [overview](https://newton-physics.github.io/newton/1.5.2/guide/overview.html)
* MuJoCo-Warp — [GitHub](https://github.com/google-deepmind/mujoco_warp); MuJoCo — [computation (contact model)](https://mujoco.readthedocs.io/en/stable/computation/index.html)
* Isaac Lab 3.0 — [Physics Backends](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/concepts/physics_backends.html), [Solver Differences](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/concepts/solver_differences.html), [Tune MJWarp](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/concepts/solver-tuning/tune_mjwarp.html)
* Isaac Lab 3.0 — [Transfer Policies Between PhysX and Newton](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/how-to/transfer_policies_between_physx_and_newton.html), [Prepare an Asset for Newton](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/how-to/prepare_asset_for_newton.html), [Coupled Solvers](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/concepts/coupled_solvers.html)

---

## Next in this special course

* Next: [Lecture 06 — Robots: Import, Tune, and Move](Lecture-06.md)
* Previous: [Lecture 04 — PhysX for Robots](Lecture-04.md)
* Back: [Isaac Sim and Isaac Lab — Overview](README.md)
