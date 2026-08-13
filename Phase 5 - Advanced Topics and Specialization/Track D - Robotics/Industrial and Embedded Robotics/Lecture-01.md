# Lecture 2: Industrial and Embedded Robotics

## Overview

Industrial and embedded robotics is where **ROS 2 prototypes** meet **factory floors**, **field deployment**, and **resource limits**. This lecture is about **interfaces** (to PLCs and MES), **repeatability** (Docker, images), **simulation fidelity** (Gazebo vs Isaac Sim), and **real-time** behavior when a missed deadline means a safety fault or scrapped part.

**By the end of this lecture you should be able to:**

* Place **OPC UA**, **Modbus**, and **EtherCAT** in the automation stack and choose a default for greenfield vs retrofit.
* Explain why **ROS-I** exists and how it differs from “research ROS 2 on a laptop.”
* Sketch a **deployment** path: dev container → Jetson image on the robot → OTA updates (conceptually).
* Compare **Gazebo / Gazebo Sim** workflows with **Isaac Sim** for perception-heavy and RL-heavy projects.
* List the main **cobot** safety ideas (speed, separation, force limiting) and where ISO/TS 15066 fits.

---

## Recommended courses (this track)

* [Introduction to Gazebo Sim with ROS 2](https://app.theconstruct.ai/courses/introduction-to-gazebo-ignition-with-ros2-170/) and [Mastering Gazebo Simulator](https://app.theconstruct.ai/courses/mastering-gazebo-simulator-78/) — Gazebo Sim + worlds/models.
* [Docker for Robotics](https://app.theconstruct.ai/courses/docker-basics-for-robotics-114/) — containerized dev and deploy.
* [Linux for Robotics](https://app.theconstruct.ai/courses/linux-for-robotics-noetic-185/) — shell, permissions, workflows (ROS 1 in the title; skills transfer directly).
* [C++ for Robotics](https://app.theconstruct.ai/courses/c-for-robotics-59/) / [Python 3 for Robotics](https://app.theconstruct.ai/courses/python-3-for-robotics-58/) — language prep before ROS 2 nodes.
* Nvidia [Isaac Sim documentation](https://docs.omniverse.nvidia.com/isaacsim/latest/index.html) — tutorials; see [ROS 2 tutorials](https://docs.omniverse.nvidia.com/isaacsim/latest/ros2_tutorials/index.html) when bridging to ROS 2 work.

---

## 1. Industrial automation and ROS-Industrial

### 1.1 What ROS-Industrial adds

**ROS-Industrial (ROS-I)** packages bridge **industrial arms**, **grippers**, and **cell layouts** to the ROS ecosystem: drivers, calibration, and motion patterns suitable for **structured environments** (cells, fixtures, conveyors). In practice you combine **MoveIt 2** / trajectory execution with **vendor controllers** and **safety PLCs** that own emergency stops and light curtains.

### 1.2 Talking to the factory

| Technology | Strength | Typical role |
|------------|----------|--------------|
| **OPC UA** | Rich information model, security, pub/sub | MES ↔ robot / SCADA, modern lines |
| **Modbus TCP** | Simple, ubiquitous | Legacy PLCs, sensors, quick integration |
| **EtherCAT** | Deterministic, cyclic I/O | Tight motion + I/O; common in high-end arms |

**Design pattern:** ROS 2 nodes handle **perception and motion intent**; a **PLC or safety relay** enforces **hard stops** and **interlocks**. Do not confuse “I can send a Modbus coil” with “I am certified safe.”

### 1.3 Collaborative robots (cobots)

Cobots trade **payload and speed** for **force-limited** operation and simpler guarding—but **risk assessment** is still required. Standards such as **ISO/TS 15066** inform **speed separation** and **force** limits relative to humans. Software must respect **reduced mode** when a person enters a zone (often wired from safety-rated inputs).

---

## 2. Embedded deployment

### 2.1 Targets

| Platform | When it shows up |
|----------|------------------|
| **NVIDIA Jetson** | GPU perception, TensorRT, multi-camera on edge |
| **ARM SBCs** (e.g. Raspberry Pi class) | Lightweight I/O, bridges, teaching |
| **x86 mini-PCs** | Development, some production cells with thermal headroom |

ROS 2 on embedded means you care about **CPU budget**, **memory**, **storage wear**, and **thermal throttling** during sustained inference.

### 2.2 Real-time Linux

Standard Linux is **best-effort** scheduling. For **servo-grade joint control** or **tight I/O**, teams use **PREEMPT_RT** kernels or dedicated motion controllers. ROS 2 can run on RT kernels, but **determinism** still requires careful thread priority, **memory locking**, and **isolcpus**. Rule of thumb: **hard real-time loops** often live outside ROS in a vendor controller; ROS supplies **setpoints** at a slower rate.

### 2.3 Docker and reproducibility

**Why Docker for robotics**

* Same image on laptop, CI, and robot.
* Pin **distro + dependencies** for Nav2 / perception stacks.

**Caveats**

* GPU and **NVIDIA Container Toolkit** for CUDA / TensorRT inside containers.
* **Device nodes** (`/dev/video*`, CAN, GPIO) must be passed through deliberately.
* **Networking** for DDS across containers needs explicit configuration.

Pattern: **one** “robot runtime” image; version with git tags; document **kernel + JetPack** (or host OS) beside the image.

---

## 3. Simulation and testing

### 3.1 URDF, SDF, and models

* **URDF:** Tree-structured robot description; widely used with ROS.
* **SDF:** World and model format in **Gazebo Sim**; supports more physics features in some workflows.

A **simulation gap** often comes from **wrong inertia**, **wrong friction**, or **missing backlash**—tune enough to catch **integration bugs**, not to match reality to the micron.

### 3.2 Gazebo / Gazebo Sim + ROS 2

Use **`ros_gz`** bridge to connect **Gazebo Sim** and ROS 2 topics. Good for **Nav2** bring-up, **sensor** prototyping, and **CI** smoke tests.

### 3.3 Isaac Sim

**Isaac Sim** targets **high-fidelity rendering**, **sensor simulation**, **synthetic data**, and **Isaac Lab** for RL. The [ROS 2 bridge](https://docs.omniverse.nvidia.com/isaacsim/latest/ros2_tutorials/index.html) connects to the same graph patterns you learned in [Lecture 1](../Advanced%20Robot%20Operating%20System/Lecture-01.md), but the authoring environment is **USD-based** and GPU-heavy.

| Need | Often choose |
|------|----------------|
| Nav2 + LiDAR bring-up, lightweight CI | Gazebo Sim |
| Photoreal data, RL, digital twin | Isaac Sim |

---

## 4. Sim-to-real checklist

1. **Clock:** Sim time vs wall clock aligned with ROS 2 time (`use_sim_time`).
2. **Sensor delay:** Add realistic **latency** in perception pipelines before trusting tuning.
3. **Calibration:** Intrinsics/extrinsics for cameras; **LiDAR–camera** extrinsics for fusion.
4. **Dynamics:** Validate **mass/inertia** order-of-magnitude for manipulation forces.

---

## 5. Projects (from this roadmap)

* **Jetson deployment:** Run Nav2 or a perception + `ros2_control` stack on Jetson; measure **CPU/GPU** and **thermal** under sustained load.
* **Sim-to-real:** One feature (e.g. obstacle avoidance) in Gazebo Sim → same launch graph on hardware; document **what broke** (TF, QoS, calibration).

---

## 6. Self-check

1. Why might a **PLC** still own E-stop even if ROS 2 plans all motions?
2. Name one reason **Docker** is risky for **camera** nodes if `/dev` is not mapped.
3. When would you prefer **Isaac Sim** over **Gazebo Sim** for a project?

---

## Resources

* **Structured courses:** See **Recommended courses** above; the [main Robotics Application guide](../Guide.md) lists all tracks.
* **ROS-Industrial:** [rosindustrial.org](https://rosindustrial.org/) and distro-specific tutorials.
* **Gazebo:** [gazebosim.org/docs](https://gazebosim.org/docs)
* **Isaac Sim:** [docs.omniverse.nvidia.com/isaacsim](https://docs.omniverse.nvidia.com/isaacsim/latest/index.html)

---

## Next in this roadmap

* Previous: [Advanced Robot Operating System](../Advanced%20Robot%20Operating%20System/Lecture-01.md)
* Next: [Advanced Perception and AI for Robotics](../Advanced%20Perception%20and%20AI%20for%20Robotics/Lecture-01.md)
