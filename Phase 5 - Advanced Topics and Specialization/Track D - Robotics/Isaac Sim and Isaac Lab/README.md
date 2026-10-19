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

This course teaches both from the bottom up — USD prims to trained policies — and treats the simulator as what it is on your desk: **a GPU program competing for a fixed VRAM and compute budget.** The labs connect mathematical and interface checks to measurements of physics, rendering, training and memory. Each result identifies the workload and execution boundary.

**Comprehensive VLA companion:** [VLA Foundations to Frontier 2026](../VLA%20Foundations%20to%20Frontier%202026/README.md) provides 32 lectures spanning history, architectures, data, imitation/flow/RL training, evaluation, deployment, and the research frontier with a source update reviewed on October 18, 2026. Use its [CPU labs](../VLA%20Foundations%20to%20Frontier%202026/Lab-Guide.md) and capstones alongside this specialization.

**Teaching release family (rechecked 2026-10-18):** Resolve an exact commit and lockfile using the [lab manifest](Lab-Guide.md#version-and-hardware-contract); a moving release branch is not an immutable pin. The [compatibility record](Compatibility.md) links the official evidence and distinguishes the full simulator, kit-less Lab workflows, and the locally executed ROS-only lab.

**Latest release update:** [Isaac Sim 6.2.0 GA](https://github.com/isaac-sim/IsaacSim/releases/tag/v6.2.0) shipped October 14, 2026. The lessons below retain their named 6.1 examples; use [the 6.2 upgrade checks](Compatibility.md#current-release-and-upgrade-checks) for changed sensor ownership, Newton views/substeps, actuator contracts, and current product compatibility. The moving Lab 3.0 branch now has different dependencies from its EA tag.

| Component | Version | Notes |
|---|---|---|
| Isaac Sim | **6.1.0** (GA, Sept 2026) | Kit 110.3; Python 3.12; open-source extensions (Apache-2.0) on a closed Kit SDK, EULA required |
| Isaac Lab | **3.0.0-EA** (tag commit `ae37b028ea415c91ea2bc32609efcd759ed2b974`, Sept 2026) | Full Isaac Sim workflows require Sim 6.1; kit-less workflows can omit it. GA remains an upstream target for late October 2026, not a released version. Breaking changes are in [Lecture 09](Lecture-09.md). |
| OS / ROS 2 | Ubuntu 24.04 + ROS 2 **Jazzy** | Humble on 22.04 also supported |
| Physics | PhysX 5 · Newton 1.5 | PhysX is Isaac Sim's default; Newton is experimental there. In Isaac Lab 3.0 the backend is selectable, and the core tasks (Cartpole, Lift) default to **Newton** (`newton_mjwarp`). |

The Core API changed under everyone's feet: Isaac Sim 5.0 introduced the **Core Experimental API**, and 6.0 **deprecated** the classic `isaacsim.core.api` (including `World` and `SimulationContext`). This course teaches the new API first and shows the old one only so you can read existing code.

**Reference hardware: an 8 GB RTX 4060.** That is **below** NVIDIA's official minimum for Isaac Sim 6.1 (RTX 4080, 16 GB). Operation on it is not guaranteed; use it only for reduced experiments after a compatibility check. The supported full-simulator route uses hardware meeting the documented requirements. Labs run headless where they can, use state observations by default, and size camera counts and resolutions to a VRAM budget. Each lecture has an **8 GB budget** note. Where a lab genuinely needs more (many RTX cameras, VLA policies in the loop), it says so and points to a cloud L40S / RTX PRO 6000 as the fallback. Data-center GPUs without RT cores (A100, H100) are **not** supported for Isaac Sim rendering.

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
| 01 | [The Isaac Stack and Your First Simulation](Lecture-01.md) | Kit, USD, PhysX/Newton, RTX · application vs simulation vs stage · full-simulator requirements and reduced CPU entry path · GUI / extension / standalone workflows · a first measured simulation |
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
| 13 | [Capstone: A Task, a Policy, and a Bridge](Lecture-13.md) | Build a custom manipulation task, size training to measured capacity, evaluate with seeds and CIs, and drive it over ROS 2, with a full performance report |
| 14 | [Reproducible Simulation Setup](Lecture-14.md) | Version/asset manifest, compatibility, zero/random-action smoke tests |
| 15 | [Frames, Units, and Asset Validation](Lecture-15.md) | Known-answer transforms, quaternion conventions, joint/asset contracts |
| 16 | [Task Correctness and Partial Resets](Lecture-16.md) | Reward probes, done/timeout semantics, reset isolation, action history |
| 17 | [Training and Held-Out Evaluation](Lecture-17.md) | Baselines, seed splits, sample/time efficiency, uncertainty |
| 18 | [Physics, Sensors, and Performance](Lecture-18.md) | Stage profiles, rate definitions, sensor/rollout memory, scaling |
| 19 | [ROS 2 Timing and Policy Export](Lecture-19.md) | Normalizers, observation age, clocks, deadlines, stale commands |
| 20 | [System Identification and Transfer](Lecture-20.md) | Actuator response, delay, randomization ranges, held-out shifts |
| 21 | [Language-Guided Robotics Capstone](Lecture-21.md) | Asynchronous planner, bounded skills, inference contention, closed-loop evaluation |

</div>

The lectures use the [Curriculum Authoring Guide](../../../Curriculum-Authoring-Guide.md): explain the concept, work a small example, run an experiment, and assess the evidence. Lectures 14–21 add self-check guidance and staged engineering exercises; the lab guide identifies which checks run on a CPU and which require Isaac.

---

## Executed CPU entry path

Start with [Simulation before the GPU](Simulation-CPU-Lab.md): 27 standard-library tests, actual Newton 1.5.2/Warp 1.16 CPU physics, actual OpenUSD 26.8 composition, measured ROS 2 message rejection, and a separate-process known-weight policy adapter. It also supplies delay-identification and paired-regression exercises. These results are recorded in the [validation manifest](labs/validation-2026-10-18.json).

The full Isaac Sim, Isaac Lab training, rendering and trained-policy examples remain optional and were not executed here. Their named source versions matter: 6.1 examples do not automatically become 6.2 examples when a link is updated. Follow the compatibility record and rerun the simulator-specific checks on the selected stack.

## Applied engineering labs

Follow the [engineering lab sequence](Lab-Guide.md) for Lectures 14–21 and runnable CPU contract exercises. Together with Lecture 10b, the course now has **22 lectures**. The sequence separates simulator, policy, and deployment milestones so learners can show exactly what they have validated.

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
* **Real-time factor is a contract.** A controller or operator tied to wall-clock time may require the simulator to keep up. An all-simulation-time ROS pipeline can run slower or faster if every component handles that clock consistently. For RL, throughput matters only after task correctness is established. Record which timing contract your scene must meet.

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
* Isaac Lab — [GitHub](https://github.com/isaac-sim/IsaacLab) · [3.0 installation](https://github.com/isaac-sim/IsaacLab/blob/ae37b028ea415c91ea2bc32609efcd759ed2b974/docs/source/setup/installation/index.rst) · [3.0 migration guide](https://github.com/isaac-sim/IsaacLab/blob/ae37b028ea415c91ea2bc32609efcd759ed2b974/docs/source/migration/migrating_to_isaaclab_3-0.rst) · [v3.0.0-EA release notes](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0-EA)
* Newton physics — [GitHub](https://github.com/newton-physics/newton)
* OpenUSD — [docs](https://openusd.org/release/index.html)

---

## Next

* Start: [Lecture 01 — The Isaac Stack and Your First Simulation](Lecture-01.md)
* Back: [Track D — Robotics Guide](../Guide.md)

**Study method:** use the [shared lesson cycle and completion rubric](../../../Learning-Guide.md) with this course's existing labs. Predict first, test one variation, retain a failure case, and revisit the self-check after two days and one week.


## Carry this course into the physical-AI system

**Start from:** A version/hardware manifest and a small scene that resets correctly.

**Build and challenge:** Vary physics step, sensor rate, and policy decimation independently. Compare simulated time with wall time, then inject delayed observations while measuring task outcome.

**Pass forward:** An environment contract, reset test, and observation-to-action traces. Continue with [the connected course](../VLA%20Foundations%20to%20Frontier%202026/README.md) and keep the artifact in the [Physical AI Design Studio](../../../Physical-AI-Design-Studio.md).
