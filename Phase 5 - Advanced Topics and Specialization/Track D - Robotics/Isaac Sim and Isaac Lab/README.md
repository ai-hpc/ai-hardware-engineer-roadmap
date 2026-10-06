# Isaac Sim and Isaac Lab — Special Course

<div class="course-identity robotics" markdown="1">
<div class="course-identity__icon">SIM</div>
<div markdown="1">
<p class="course-identity__eyebrow">Phase 5 · Robotics · Special Course</p>
<p class="course-identity__title">Build, simulate, sense, and train robots in NVIDIA Isaac Sim and Isaac Lab — and know exactly where every millisecond and megabyte of GPU goes.</p>
<p class="course-identity__meta">Artifact: custom Isaac Lab task + trained policy + ROS 2 bridge + performance report · Measure: physics step time, render time, env-steps/s, VRAM per env, real-time factor</p>
</div>
</div>

> *A simulator is a GPU workload. Treat it like one.*

Isaac Sim is NVIDIA's robotics simulator: an Omniverse Kit application that composes a **USD** scene, steps it with **PhysX** (or the new **Newton** engine), renders sensors with **RTX**, and talks to **ROS 2**. Isaac Lab is the robot-learning framework on top of it: thousands of cloned environments on one GPU, managers for observations, rewards and randomization, and adapters for RL and imitation-learning libraries.

This course teaches both from the bottom up — USD prims to trained policies — and treats the simulator as what it is on your desk: **a GPU program competing for a fixed VRAM and compute budget.** Every lab measures step time, render time, and memory, and every lecture says which knob moves which number.

**Version pin (checked October 2026):**

| Component | Version | Notes |
|---|---|---|
| Isaac Sim | **6.1.0** (GA, Sept 2026) | Kit 110.3; Python 3.12; open-source extensions (Apache-2.0) on a closed Kit SDK, EULA required |
| Isaac Lab | **3.0.0-EA** (`release/3.0.0`, Sept 2026) | Requires Isaac Sim 6.1; GA targeted for late October 2026. Breaking changes from 2.x are called out in [Lecture 09](Lecture-09.md). |
| OS / ROS 2 | Ubuntu 24.04 + ROS 2 **Jazzy** | Humble on 22.04 also supported |
| Physics | PhysX 5 · Newton 1.5 | PhysX is Isaac Sim's default; Newton is experimental there. In Isaac Lab 3.0 the backend is selectable, and the core tasks (Cartpole, Lift) default to **Newton** (`newton_mjwarp`). |

The Core API changed under everyone's feet: Isaac Sim 5.0 introduced the **Core Experimental API**, and 6.0 **deprecated** the classic `isaacsim.core.api` (including `World` and `SimulationContext`). This course teaches the new API first and shows the old one only so you can read existing code.

**Reference hardware: an 8 GB RTX 4060.** That is **below** NVIDIA's official minimum for Isaac Sim 6.1 (RTX 4080, 16 GB). It runs, but not everything fits, so the course treats the limit as a design constraint. Labs run headless where they can, use state observations by default, and size camera counts and resolutions to a VRAM budget. Each lecture has an **8 GB budget** note. Where a lab genuinely needs more (many RTX cameras, VLA policies in the loop), it says so and points to a cloud L40S / RTX PRO 6000 as the fallback. Data-center GPUs without RT cores (A100, H100) are **not** supported for Isaac Sim rendering.

**Layer mapping:** L3-L8. Scene description and physics (simulation runtime), GPU rendering and sensor pipelines, ROS 2 integration, and the RL training loop.

**Role targets:** Robotics Simulation Engineer · Robot Learning Engineer · Synthetic Data Engineer · Sim-to-Real Engineer · Robotics Infrastructure / Platform Engineer.

**Prerequisites:**

* Python and PyTorch fluency; comfort on the Linux command line.
* Phase 5 — Robotics — [Advanced Robot Operating System](../Advanced%20Robot%20Operating%20System/Lecture-01.md) (ROS 2 nodes, topics, TF) before Lecture 08.
* Phase 5 — Robotics — [Deep RL for Robot Learning](../Deep%20RL%20for%20Robot%20Learning/README.md), at least Lectures 01, 04-06, before Lectures 09-10. This course is the simulator side of that one.

**What comes after:** use Isaac Lab as the rung-2 simulator in [Deep RL for Robot Learning](../Deep%20RL%20for%20Robot%20Learning/README.md), and the evaluation setup for [VLA Optimization and Action-Parity Harness](../VLA%20Optimization%20and%20Action-Parity%20Harness/README.md).

---

## Lecture map

<div class="lecture-map" markdown>

| # | Title | Focus |
|---|-------|-------|
| 01 | [The Isaac Stack and Your First Simulation](Lecture-01.md) | Kit, USD, PhysX/Newton, RTX · application vs simulation vs stage · install on an 8 GB card · GUI / extension / standalone workflows · a first measured simulation |
| 02 | [USD for Roboticists](Lecture-02.md) | Prims, attributes, stages, layers · references, payloads, variants, instancing · physics schemas · raw USD vs wrappers · USD vs Fabric vs tensors · hosted assets |
| 03 | [The Core Experimental API](Lecture-03.md) | Batched wrappers and regex paths · Warp arrays · data backends · `SimulationManager` / `RenderingManager` · callbacks · migrating from `World` |
| 04 | [PhysX for Robots](Lecture-04.md) | Rigid bodies, colliders, contact offsets, friction · TGS vs PGS, iterations, timestep · CCD and GPU dynamics · articulations and joint drives · stable grasping |
| 05 | [Newton and Multi-Physics](Lecture-05.md) | Newton and MuJoCo-Warp · switching engines · solvers (MJWarp, XPBD, VBD, Featherstone) · deformables · what transfers between engines |
| 06 | [Robots: Import, Tune, and Move](Lecture-06.md) | URDF / MJCF importers · collision and inertia validation · Robot Assembler, gain tuner, sysid · cuMotion motion generation · pick-and-place |
| 07 | [Sensors, Rendering, and Synthetic Data](Lecture-07.md) | RTX cameras, annotators, tiled cameras · RTX lidar · IMU, contact, raycast · Replicator domain randomization · Cosmos writer · the VRAM cost of pixels |
| 08 | [ROS 2 and OmniGraph](Lecture-08.md) | `isaacsim.ros2.bridge` · action graphs · clock, TF, sensors · `ros2_control` · Nav2 / MoveIt in the loop · latency and real-time factor |
| 09 | [Isaac Lab 3.0 Architecture](Lecture-09.md) | Packages · manager-based vs direct workflows · `InteractiveScene` and cloning · physics backends · `ProxyArray`, xyzw · events and randomization · kit-less mode · 2.x → 3.0 migration |
| 10 | [Training Policies in Isaac Lab](Lecture-10.md) | `isaaclab train` · RSL-RL PPO on Cartpole, Lift, and locomotion · `num_envs` and backend sweeps on 8 GB · camera-based tasks · writing your own direct task |
| 10b | [Worked Example: Spot Waypoint Navigation](Lecture-10b.md) | Hierarchical control: waypoint follower → 50 Hz learned locomotion policy → 500 Hz PD actuators → physics · the policy contract (48-dim observation, command ranges) · `RobotPolicyRunner` on PhysX and Newton · train your own Spot policy and deploy it · ROS 2 `/cmd_vel` bridge |
| 11 | [Imitation, Teleop, and Policy Evaluation](Lecture-11.md) | Teleop with Isaac Capture (formerly Isaac Teleop) · Mimic data generation · robomimic · Isaac Lab-Arena for VLA evaluation · RLinf · client-server policies |
| 12 | [Performance, Sim-to-Real, and Scale-Out](Lecture-12.md) | Profiling physics vs render vs Python · headless and streaming · containers and cloud · multi-GPU · determinism · actuator models and domain randomization |
| 13 | [Capstone: A Task, a Policy, and a Bridge](Lecture-13.md) | Build a custom manipulation task, train it within 8 GB, evaluate with seeds and CIs, and drive it over ROS 2, with a full performance report |

</div>

Each lecture follows the *Why it matters → Mental model → Build it → Use it in the real stack → Measure it → Ship it → Exit criteria* shape from the [Curriculum Authoring Guide](../../../Curriculum-Authoring-Guide.md), with a self-check and a runnable lab.

---

## Learning path: from "what is Isaac Sim" to physical AI

If you are coming from a five-level syllabus (mental model → build a robot → robotics AI → learning → advanced), this table maps each topic to where the course teaches it. The end goal is the same in both: **Isaac Sim → Isaac Lab → RL / VLA training → deployment on a real robot**.

| Level | Topics | Where |
|---|---|---|
| 1. Mental model | What Isaac Sim is; Isaac Sim vs Isaac Lab | [01](Lecture-01.md), [09](Lecture-09.md) |
| | USD, stages and prims | [02](Lecture-02.md) |
| | PhysX, and the articulated-body equation \(M(q)\ddot q + C\dot q + g = \tau + J^\top F\) | [04](Lecture-04.md) |
| | Simulation time vs wall-clock time; real-time factor | [01](Lecture-01.md) §3, [08](Lecture-08.md) |
| | Articulations, rigid bodies, joints | [03](Lecture-03.md), [04](Lecture-04.md) |
| | Extensions and Python scripting | [01](Lecture-01.md) §5, [03](Lecture-03.md) |
| 2. Build a robot | Import a robot; build from USD components | [06](Lecture-06.md), [02](Lecture-02.md) |
| | Joints, drives, controllers | [04](Lecture-04.md), [06](Lecture-06.md) |
| | Cameras, lidar, IMU, contact sensors | [07](Lecture-07.md) |
| | Programmatic control | [03](Lecture-03.md), [10b](Lecture-10b.md) |
| 3. Robotics AI | ROS 2 integration | [08](Lecture-08.md) |
| | Perception pipelines, synthetic data, domain randomization | [07](Lecture-07.md), [12](Lecture-12.md) §6 |
| | Navigation | [10b](Lecture-10b.md) (waypoints, learned locomotion), [08](Lecture-08.md) (Nav2) |
| | Manipulation and motion planning | [06](Lecture-06.md) (cuMotion), [04](Lecture-04.md) (grasping) |
| 4. Learning | Isaac Lab and RL environments | [09](Lecture-09.md), [10](Lecture-10.md) |
| | PPO and policy learning | [10](Lecture-10.md), [Deep RL 05-06b](../Deep%20RL%20for%20Robot%20Learning/Lecture-06b.md) |
| | Imitation learning | [11](Lecture-11.md), [Deep RL 02-03](../Deep%20RL%20for%20Robot%20Learning/Lecture-02.md) |
| | Vision-language-action systems | [11](Lecture-11.md) (Arena evaluation), [Deep RL 11, 14](../Deep%20RL%20for%20Robot%20Learning/Lecture-14.md) (RL post-training of π0.5) |
| | Sim-to-real | [12](Lecture-12.md) §6, [10b](Lecture-10b.md) (Spot RL Researcher Kit) |
| 5. Advanced | GPU simulation and parallel environments | [05](Lecture-05.md), [09](Lecture-09.md), [12](Lecture-12.md) |
| | Digital twins | [12](Lecture-12.md) §6.4 |
| | Large-scale RL | [10](Lecture-10.md), [12](Lecture-12.md) §4 |
| | Deploying policies on real robots and Jetson | [10b](Lecture-10b.md), [13](Lecture-13.md), [VLA Optimization](../VLA%20Optimization%20and%20Action-Parity%20Harness/README.md), [VLA on Jetson](../../../Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/5.%20Application%20Development/5.%20ML%20and%20AI/vla-deploy-jetson/Guide.md) |

**A vision-language robot agent** ("put the red cup in the box") is the natural project at the end of this path. It combines this course's simulation and evaluation stack (Lectures 07, 11, 13) with policy training from [Deep RL for Robot Learning](../Deep%20RL%20for%20Robot%20Learning/README.md) (Lecture 14 post-trains π0.5) and real-time inference from [VLA Optimization](../VLA%20Optimization%20and%20Action-Parity%20Harness/README.md).

---

## Why this is a hardware-first course

On an RTX 4060, every simulation decision is a resource decision:

* **Physics and rendering share one GPU.** PhysX GPU dynamics, Newton's Warp kernels, RTX ray tracing, your policy network, and the desktop compositor all draw from the same 8 GB and the same SMs. A camera at 640×480 in 64 environments can cost more than the physics of 4,096 state-only environments.
* **Step time has three parts:** physics, rendering, and Python/data movement. They respond to different knobs (solver iterations and substeps; render products, resolution and RTX settings; data backend and batch size). Optimizing the wrong one does nothing.
* **The data path is a design choice.** USD is for authoring, Fabric is the GPU runtime store for transforms and rendering, and the physics tensor API is the zero-copy path for training. Reading poses through USD in a training loop is the most common silent slowdown.
* **Real-time factor is a contract.** For ROS 2 sim-in-the-loop, the simulator has to keep up with wall-clock time; for RL it should run as far *above* real time as possible. The same scene can meet one and miss the other.

Every **Measure it** section asks for physics ms/step, render ms/frame, env-steps/s, peak VRAM, and VRAM per environment, so by the capstone you have a performance model of your own GPU.

---

## What you ship

By the end of the course you should have, in one repo:

* **`isaac_bench/`** — a reusable harness that measures physics step time, render time, env-steps/s and VRAM for a scene, with results for every lab on your GPU.
* **USD assets** — a robot and a work cell you imported, validated (collision, inertia, drives) and composed with references and variants.
* **A custom Isaac Lab task** (direct workflow), with randomization, a trained policy, and learning curves over ≥ 3 seeds.
* **A ROS 2 bridge demo** — the trained or scripted robot driven over ROS 2 Jazzy, with measured end-to-end latency and real-time factor.
* **`PERFORMANCE.md`** — the VRAM and step-time budget for your capstone on an 8 GB card, and what you would change on 16 GB and 48 GB cards.

---

## Exit criteria

You are done with this special course when you can:

* explain what Kit, USD, Fabric, PhysX/Newton, and RTX each do in one frame of simulation, and which one a given symptom points to
* write a standalone Isaac Sim script with the Core Experimental API, and port a classic `World`-based script to it
* tune a gripper until a grasp is stable, and justify every physics parameter you changed
* predict, before running it, roughly how many environments and cameras fit in 8 GB for a given task, then measure it
* build, train, and evaluate a custom Isaac Lab task, and explain each of its managers or step functions
* drive the simulation over ROS 2 and report its latency and real-time factor

---

## References for the whole course

* Isaac Sim 6.1 documentation — [docs](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/index.html) · [GitHub](https://github.com/isaac-sim/IsaacSim) · [release notes](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/overview/release_notes.html)
* Core API → Core Experimental migration guide — [docs](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/migration_guides/isaac_sim_6_0/core_api_to_core_experimental.html)
* Isaac Lab — [GitHub](https://github.com/isaac-sim/IsaacLab) · [3.0 installation](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/setup/installation/index.html) · [3.0 migration guide](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/migration/migrating_to_isaaclab_3-0.html) · [v3.0.0-EA release notes](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0-EA)
* Newton physics — [GitHub](https://github.com/newton-physics/newton)
* OpenUSD — [docs](https://openusd.org/release/index.html)

---

## Next

* Start: [Lecture 01 — The Isaac Stack and Your First Simulation](Lecture-01.md)
* Back: [Track D — Robotics Guide](../Guide.md)
