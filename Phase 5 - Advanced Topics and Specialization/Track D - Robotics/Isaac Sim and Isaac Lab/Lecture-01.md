# Lecture 01: The Isaac Stack and Your First Simulation

## Overview

Isaac Sim looks like a 3D editor with a play button. Underneath it is a stack of four GPU-hungry systems glued together by a scene description format, and nearly every problem you will hit later — a robot that explodes, a camera that eats gigabytes of VRAM, a training loop that runs at 10% of the speed it should — comes from not knowing which layer you are talking to.

This lecture builds that map, installs Isaac Sim 6.1 on an **8 GB RTX 4060**, and runs a first standalone simulation that you **measure**: physics time per step, frame time with rendering, and VRAM. Those three numbers are the baseline every later lecture compares against.

By the end you should be able to:

* name the layers of Isaac Sim (Kit, USD, Fabric, physics, RTX) and say what each does in one frame
* explain application vs simulation vs stage vs scene, and how Isaac Sim 6.x replaced `World`
* install Isaac Sim 6.1 with pip, run the compatibility checker, and get assets loading
* choose between the GUI, extension, and standalone workflows for a given job
* write a standalone script with the Core Experimental API and time its physics and rendering separately
* account for where the 8 GB on your card goes

---

## 1. Why it matters: three jobs, one simulator

People use Isaac Sim for three different jobs, and they stress the GPU differently:

| Job | What runs | Bottleneck | Lectures |
|---|---|---|---|
| **Sim-in-the-loop** for a robot software stack (Nav2, MoveIt, perception) | One robot, several sensors, ROS 2 bridge, roughly real time | Rendering (RTX sensors) and **real-time factor** | 07, 08 |
| **Synthetic data generation** (SDG) for perception models | Randomized scenes, many rendered frames, annotations written to disk | Rendering throughput and disk I/O | 07 |
| **Robot learning** (RL, imitation) | Thousands of cloned environments, mostly physics, sometimes cameras | Physics throughput, **VRAM per environment** | 09-11 |

The same scene can be fine for one job and hopeless for another. A warehouse with four RTX lidars might hold real time for Nav2 testing and still be useless for RL, where you want 1,000× real time. Knowing which job you are doing tells you which numbers to watch.

---

## 2. Mental model: the stack

```text
 ┌──────────────────────────────────────────────────────────────────────────┐
 │  Omniverse Kit application  (isaacsim.exp.full, or your SimulationApp)   │
 │  extensions: UI, importers, sensors, ROS 2 bridge, your code             │
 ├──────────────────────────────────────────────────────────────────────────┤
 │  USD stage  — the scene description: prims + attributes + composition    │  authoring
 ├───────────────┬───────────────────────────────┬──────────────────────────┤
 │  Fabric /     │  Physics: PhysX 5 (default)   │  RTX renderer            │
 │  USDRT        │  or Newton (experimental)     │  (ray-traced, Hydra)     │  runtime
 │  GPU runtime  │  + tensor API for batched     │  cameras, lidar, radar   │
 │  store        │  state (zero-copy to torch)   │  via render products     │
 └───────────────┴───────────────────────────────┴──────────────────────────┘
        ▲                      ▲                              ▲
   transforms for         Python reads/writes            annotators → RGB,
   rendering              joint states, poses            depth, segmentation
```

* **Kit** is NVIDIA's application framework. Isaac Sim is a set of Kit **extensions** (the Isaac-specific ones have been open source under Apache-2.0 since 5.0; Kit itself stays closed and needs the Omniverse EULA). An **experience** (`.kit` file) is a list of extensions plus settings; `isaacsim.exp.full` is the standard app.
* **USD** (OpenUSD) is the scene description. Everything — robot, table, light, camera, physics scene — is a **prim** with **attributes**. USD is optimized for authoring and composition, not for being read 1,000 times a second.
* **Fabric** (with its Python API, USDRT) is the GPU-friendly runtime copy of the scene data that changes every frame, chiefly transforms for rendering.
* **Physics** steps the dynamic prims. PhysX is the default. Newton (a Warp-based, open-source engine using MuJoCo-Warp) is experimental in 6.x and selectable at launch (Lecture 05). The **physics tensor API** exposes batched state (`ArticulationView`, `RigidBodyView`) after the simulation starts, which is the fast path for training.
* **RTX** renders the viewport and every sensor. A camera is a **render product**; **annotators** turn it into RGB, depth, segmentation, and so on (Lecture 07).

### 2.1 Application, simulation, stage, scene, world

NVIDIA's docs explain these words with a theater: the **application** is the theater (your gateway: GUI, rendering, input); the **simulation** is the play, moving prims forward through time by changing their attributes; the **stage** is the USD stage where everything exists and where relationships between prims (a mug *on* a table) are expressed; a **scene** is the set of props and actors in the current act; and the **world** is the stage crew that runs the curtain, resets the props between scenes, and keeps the play on schedule.

That last role — the stage crew — is where Isaac Sim changed. In the classic Core API (5.x and earlier) it was one object, `World`. **Isaac Sim 6.0 deprecated `World` and `SimulationContext`** along with the rest of `isaacsim.core.api`. There is no single replacement; its jobs are split:

| Concept | Classic Core API (5.x, deprecated) | Isaac Sim 6.1 |
|---|---|---|
| Create / open a stage | `World()` creates one implicitly | `isaacsim.core.experimental.utils.stage` (`create_new_stage`, `add_reference_to_stage`) |
| Add objects | `world.scene.add(DynamicCuboid(...))` | Create prims directly: `Cube(...)`, `GeomPrim(..., apply_collision_apis=True)`, `RigidPrim(...)` |
| Play / step physics | `world.reset()`, `world.step(render=True)` | Timeline `play()`; `SimulationManager.step()` for physics; `simulation_app.update()` steps physics *and* renders |
| Render | inside `world.step(render=True)` | `RenderingManager.render()` |
| Per-step callbacks | `world.add_physics_callback(...)` | `SimulationManager` callbacks (e.g. post-physics-step events) |
| Read state | `cube.get_world_pose()` (one prim, NumPy) | `cube.get_world_poses()` (batched, **Warp** arrays; `.numpy()` to convert) |

Most tutorials, forum answers, and GitHub code you find still use the left column. Lecture 03 covers the migration properly. For now: **read the left, write the right.**

### 2.2 The Core API is a wrapper

The Core API (classic or experimental) is a robotics-shaped wrapper over raw USD and physics schemas. Adding one falling cube with raw USD means defining a physics scene, a ground plane, a `UsdGeom.Cube`, a transform op, and applying `UsdPhysics.RigidBodyAPI` and `UsdPhysics.CollisionAPI`:

```python
from pxr import Gf, UsdGeom, UsdPhysics
import omni.usd

stage = omni.usd.get_context().get_stage()
UsdPhysics.Scene.Define(stage, "/World/physics")            # gravity defaults, solver settings via PhysxSchema
cube = UsdGeom.Cube.Define(stage, "/World/Cube")
cube.CreateSizeAttr(0.5)
cube.AddTranslateOp().Set(Gf.Vec3f(0.5, 0.2, 1.0))
UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())                # "this prim is a dynamic body"
UsdPhysics.CollisionAPI.Apply(cube.GetPrim())                # "this prim collides"
```

The experimental Core API says the same thing in robot terms, and works on one prim or a thousand:

```python
from isaacsim.core.experimental.objects import Cube
from isaacsim.core.experimental.prims import GeomPrim, RigidPrim

shape = Cube(paths="/World/Cube", positions=np.array([[0.5, 0.2, 1.0]]), sizes=[0.5])
GeomPrim(paths=shape.paths, apply_collision_apis=True)       # collision
cube = RigidPrim(paths=shape.paths)                           # dynamics
```

Both produce the same USD. Knowing the raw form matters for two reasons: imported robots arrive as raw USD you will need to inspect (Lecture 02), and when a wrapper hides a physics parameter, you set it on the schema directly (Lecture 04).

---

## 3. The hardware view: one frame, one GPU

A standalone simulation loop does this every `simulation_app.update()`:

```text
  Python callbacks ──► physics step(s) ──► sync to Fabric ──► RTX render ──► annotators / UI
     (CPU)               (GPU or CPU)        (GPU)              (GPU)          (GPU → CPU)
```

**Physics and rendering run at different rates.** Physics typically steps at a fixed small timestep (and may take several substeps per rendered frame); rendering happens per app update or per sensor rate. In a training loop you often step physics many times without rendering at all. That is why this lecture times them separately.

**Where the 8 GB goes.** On a 4060, VRAM is the constraint you will hit first. These are the consumers:

| Consumer | Scales with | Lever |
|---|---|---|
| Display / desktop compositor | Monitors, other apps (browser tabs use GPU too) | Close them; run headless on a second machine if you have one |
| Kit + RTX renderer baseline | Renderer mode and settings; loaded once | Headless; lower-cost render settings |
| Scene assets (meshes, textures, materials) | Number of *unique* assets | USD instancing (Lecture 02); simpler materials |
| Physics GPU buffers | Bodies, shapes, contacts, articulations | Fewer collision shapes per body; contact buffer sizes (Lecture 04) |
| Render products (each camera) | Count × resolution × annotators | The biggest lever in Lectures 07 and 10 |
| PyTorch caching allocator | Policy, batch, replay buffers | `torch.cuda.memory_summary()`; free what you do not need |

> **8 GB budget.** On a 4060, close the browser and other GPU apps before measuring, run labs headless unless you need to look, and treat the GUI as a debugging tool rather than the default. Physics-only scenes with thousands of simple bodies fit; the first things to overflow are many RTX cameras and large textured environments (Lecture 07). If your fixed cost from Lab 1a leaves only a few GB, plan to run camera-heavy labs on a cloud L40S or RTX PRO 6000 and keep the 4060 for physics and policy work.

Rule for this course: **measure the baseline before adding anything.** If the empty app already takes a large part of your card, every later VRAM-per-env number has to fit in what is left.

---

## 4. Install Isaac Sim 6.1 on an RTX 4060

NVIDIA's minimum for 6.1 is an RTX 4080 with 16 GB. An 8 GB 4060 is below that: it runs, but the renderer, large scenes, and many cameras will hit the ceiling. That is acceptable for learning, as long as you know it.

**Prerequisites.** Ubuntu 22.04 or 24.04 (x86_64), GLIBC 2.35+, a recent NVIDIA driver (the 6.1 docs list 595.58.03 as the tested Linux driver), 32 GB RAM recommended, 50 GB+ free SSD.

```bash
# 1. Python 3.12 environment (Isaac Sim 6.x requires 3.12)
python3.12 -m venv ~/env_isaacsim && source ~/env_isaacsim/bin/activate
pip install --upgrade pip

# 2. PyTorch first (CUDA 12.8 wheels; cu130 also works with a new enough driver)
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128

# 3. Isaac Sim 6.1 from NVIDIA's index (large download)
pip install "isaacsim[all,extscache]==6.1.0.0" --extra-index-url https://pypi.nvidia.com

# 4. Accept the Omniverse EULA non-interactively
export OMNI_KIT_ACCEPT_EULA=YES

# 5. Check your system against the requirements
isaacsim isaacsim.exp.compatibility_check

# 6. Launch the full app (the first launch compiles shaders and can take many minutes)
isaacsim isaacsim.exp.full
```

Alternatives: the **workstation zip** (`./isaac-sim.sh`, scripts run with `./python.sh`), the **container** `nvcr.io/nvidia/isaac-sim:6.1.0` (Linux, rootless, `-e ACCEPT_EULA=Y`), or **building from source** from the GitHub repo. The pip path is the easiest to combine with Isaac Lab 3.0 later.

**Assets.** Since October 2025 there is no Omniverse Launcher or local Nucleus. Isaac Sim 6.1 loads its sample assets over HTTPS from a hosted service, and `isaacsim.storage.native.get_assets_root_path()` returns the root. Behind a firewall, set `ISAACSIM_ASSET_ROOT` to a mirror or download the offline asset packs. In 6.1 the robot assets moved from `Isaac/Robots/` to `Isaac/Robots_Multiphysics/`, so older tutorials' paths may 404.

**Run VS Code against it:** `python -m isaacsim --generate-vscode-settings` writes settings so imports resolve.

---

## 5. Three workflows

| Workflow | How it runs | Use it for | Avoid it for |
|---|---|---|---|
| **GUI** | `isaacsim isaacsim.exp.full`; build and inspect interactively | Inspecting imported robots, tuning physics visually, debugging | Anything you need to reproduce |
| **Extension** | Your code is a Kit extension; runs asynchronously inside the app; hot-reloads | Tools and UI panels, interactive demos | Precise control of step timing |
| **Standalone Python** | Your script creates a `SimulationApp` and owns the loop | Benchmarks, data generation, RL, CI, headless runs | Quick visual poking |

This course uses **standalone Python** for every lab, because it is the only workflow where you control exactly when physics steps and when rendering happens. Two rules:

1. **Create `SimulationApp` first.** `from isaacsim import SimulationApp` and construct it before importing anything from `omni.*` or `isaacsim.*`, because those modules only exist once Kit has started.
2. **Close it.** `simulation_app.close()` at the end, or the process can hang holding GPU memory.

---

## 6. Build it

Everything goes in `isaac_bench/`, the harness you will extend all course.

### Lab 1a — Baseline VRAM

Measure what the system costs before your scene adds anything. In one terminal:

```bash
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 1 > vram_log.csv
```

Record `memory.used` in four states, about 30 s each:

1. Desktop idle (nothing running)
2. `isaacsim isaacsim.exp.full` open on an empty stage
3. The same, after loading `Isaac/Environments/Grid/default_environment.usd`
4. A headless `SimulationApp({"headless": True})` with the same environment (Lab 1b script, rendering on)

The difference between 1 and 4 is your **fixed cost**. What remains of 8 GB is your **budget** for everything else in the course. Write both down.

### Lab 1b — First standalone simulation, timed

```python
# isaac_bench/falling_cubes.py
import argparse, time
parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=64)          # number of cubes
parser.add_argument("--headless", action="store_true")
parser.add_argument("--steps", type=int, default=600)
args = parser.parse_args()

from isaacsim import SimulationApp                          # must come before omni/isaacsim imports
simulation_app = SimulationApp({"headless": args.headless})

import numpy as np, omni.timeline
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.experimental.objects import Cube
from isaacsim.core.experimental.prims import GeomPrim, RigidPrim
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.storage.native import get_assets_root_path

stage_utils.add_reference_to_stage(
    usd_path=get_assets_root_path() + "/Isaac/Environments/Grid/default_environment.usd",
    path="/World/ground",
)

# A grid of n cubes, 25 cm apart, dropped from 1 m (side**2 >= n, so it is one layer: each cube lands on the ground).
side = int(np.ceil(np.sqrt(args.n)))
idx = np.arange(args.n)
pos = np.stack([(idx % side) * 0.25, (idx // side % side) * 0.25, 1.0 + (idx // side**2) * 0.3], axis=1)
paths = [f"/World/Cube_{i}" for i in range(args.n)]

t0 = time.perf_counter()
shape = Cube(paths=paths, positions=pos, sizes=[0.1] * args.n)   # one batched wrapper for all cubes
GeomPrim(paths=shape.paths, apply_collision_apis=True)
cubes = RigidPrim(paths="/World/Cube_.*")                         # regex: the same n prims
t_build = time.perf_counter() - t0

omni.timeline.get_timeline_interface().play()
simulation_app.update()                                           # first update starts physics

def timed(fn, steps):
    t = time.perf_counter()
    for _ in range(steps):
        fn()
    return (time.perf_counter() - t) / steps * 1e3                # ms per call

ms_physics = timed(SimulationManager.step, args.steps)            # physics only, no render
ms_update = timed(simulation_app.update, args.steps)              # physics + render (+ UI if not headless)

positions, _ = cubes.get_world_poses()                            # Warp array, shape (n, 3)
z = positions.numpy()[:, 2]
print(f"n={args.n} headless={args.headless} build={t_build:.2f}s "
      f"physics={ms_physics:.3f} ms/step update={ms_update:.3f} ms/frame "
      f"z: min={z.min():.3f} max={z.max():.3f}")

simulation_app.close()
```

```bash
python isaac_bench/falling_cubes.py --n 64              # with the GUI viewport
python isaac_bench/falling_cubes.py --n 64 --headless
```

Check the physics before trusting the timings: after 1,200 steps every cube should be resting near the ground (`z` close to half the cube size, 0.05 m). If cubes are still at 1 m, the simulation is not playing. If any `z` is far below 0, something fell through the floor.

If your build of the wrapper rejects a list of paths, create the cubes in a loop and keep the single regex `RigidPrim`. Building is a one-time cost; the regex wrapper is what matters for stepping.

### Lab 1c — Scaling sweep

Run Lab 1b for `n ∈ {1, 16, 64, 256, 1024, 4096}`, headless and windowed, while logging VRAM with `nvidia-smi`. Plot:

* physics ms/step vs `n` (log-log)
* app-update ms/frame vs `n`
* peak VRAM minus your Lab 1a fixed cost, vs `n`

What to expect, qualitatively. At small `n`, physics time is almost flat, because fixed per-step overhead dominates. At large `n` it grows with the number of bodies and contacts. Here every cube rests on the ground rather than on another cube, so contacts grow roughly linearly with `n`; dense stacks (Lecture 04) grow them faster. Update time includes rendering, so the gap between the two curves is the price of drawing a frame. Note where your 4060 stops being comfortable. That number feeds the environment-count estimates in Lecture 10.

---

## 7. Use it in the real stack

* **The canonical starting points** in the docs are the Core API tutorials (*Hello World*, then *Hello Robot*), now written against the experimental API. Read *Hello World* side by side with Lab 1b.
* **Standalone examples** ship with Isaac Sim; in the workstation install they live under `standalone_examples/`. Many still show the classic API, so treat them as reading material.
* **Isaac Lab** (Lectures 09-13) is itself a standalone-Python application on top of everything in this lecture. Its `--viz none` flag is the Isaac Lab 3.0 version of `headless=True`.

---

## 8. Measure it

| Metric | How | Why it matters |
|---|---|---|
| **Fixed VRAM cost** | Lab 1a, `nvidia-smi` | Your budget is 8 GB minus this |
| **Physics ms/step** | `SimulationManager.step()` timed | Upper bound on sim speed without rendering |
| **Frame ms** | `simulation_app.update()` timed | Cost when rendering is on: sim-in-the-loop and cameras |
| **Real-time factor** | `physics_dt / wall-clock per step` | Must be ≥ 1 for ROS 2 sim-in-the-loop; as high as possible for RL |
| **VRAM per body** | Slope of the Lab 1c VRAM curve | First data point for "how many envs fit" |
| **Build time** | `t_build` | Scene construction cost, which matters for reset-heavy workflows |

For the real-time factor you need the physics timestep. Read it from the physics scene in your stage, and record it with your results.

---

## 9. Ship it

Commit `isaac_bench/` with:

* `falling_cubes.py`, `run_sweep.sh`, and `vram_log.csv` files
* `results.csv` — one row per (n, headless) with build s, physics ms/step, frame ms, peak VRAM
* `scaling.png` — the three curves from Lab 1c
* `BASELINE.md` — your GPU, driver, Isaac Sim version, fixed VRAM cost, remaining budget, and one paragraph: at what `n` does rendering dominate frame time, and at what `n` does physics?

---

## Exit criteria

You can move on when you can:

* draw the stack from memory and say which layer owns transforms, contacts, and pixels
* explain why `World` is gone in 6.x, and what replaces each of its jobs
* install Isaac Sim 6.1 from scratch, and get assets loading with or without internet access
* write a standalone script that creates batched objects, steps physics, and reads poses as Warp arrays
* state your card's fixed VRAM cost and the physics vs render split for a 1,024-body scene

---

## Self-check

1. A teammate's script imports `isaacsim.core.experimental.prims` on line 1 and constructs `SimulationApp` on line 3. It fails with a `ModuleNotFoundError`. Why, and what is the fix?
2. Lab 1c shows physics at 2 ms/step and frame time at 25 ms for 1,024 cubes, windowed. Which layer is responsible for most of the frame, and name two changes that would shrink it without touching physics.
3. Your sim-in-the-loop test needs a real-time factor ≥ 1 with a 1/60 s physics timestep and rendering on. Lab 1b measures 21 ms per `update()`. Does it meet the contract? What are your options?
4. An old tutorial calls `world.scene.add(DynamicCuboid(...))` and `world.step(render=True)`. Write the Isaac Sim 6.1 equivalent in three or four lines, and name what changed about the return type of pose queries.
5. Your empty Isaac Sim GUI already uses a large share of your 8 GB card. List three things you can do to recover memory before a training run, in order of how much each is likely to save.

---

## References

* Isaac Sim 6.1 — [requirements](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/requirements.html), [pip install](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/install_python.html), [container](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/install_container.html), [accessing assets](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/accessing_assets.html)
* Development workflows — [docs](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/introduction/workflows.html)
* Standalone Python and `SimulationApp` — [docs](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/python_scripting/manual_standalone_python.html)
* Core API overview (wrappers, application/simulation/world/scene/stage) — [docs](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/python_scripting/core_api_overview.html)
* Core API Hello World tutorial — [docs](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/core_api_tutorials/tutorial_core_hello_world.html)
* Core API → Core Experimental migration guide — [docs](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/migration_guides/isaac_sim_6_0/core_api_to_core_experimental.html)
* Physics data pathways (USD, Fabric, tensors) — [docs](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/physics/new_physics_engine.html)
* Isaac Sim on GitHub — [repo](https://github.com/isaac-sim/IsaacSim)

---

## Next in this special course

* Next: [Lecture 02 — USD for Roboticists](Lecture-02.md)
* Back: [Isaac Sim and Isaac Lab — Overview](README.md)
