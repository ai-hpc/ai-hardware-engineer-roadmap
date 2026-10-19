# Lecture 2: Industrial and Embedded Robotics

**Begin the practical work:** [read-only industrial integration lab](Integration-Lab.md).
Decode a Modbus TCP exchange by hand, reproduce it on localhost, then check the
same server with a pinned PyModbus client. The packet, application units, and
failure responses provide a concrete artifact for this survey.

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
* Nvidia [Isaac Sim 6.2 documentation](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/index.html) — current simulator documentation; use the course's [compatibility record](../Isaac%20Sim%20and%20Isaac%20Lab/Compatibility.md) when following the named 6.1 teaching baseline.

---

## 1. Industrial automation and ROS-Industrial

### 1.1 What ROS-Industrial adds

**ROS-Industrial (ROS-I)** packages bridge **industrial arms**, **grippers**, and **cell layouts** to the ROS ecosystem: drivers, calibration, and motion patterns suitable for **structured environments** (cells, fixtures, conveyors). In practice you combine **MoveIt 2** / trajectory execution with **vendor controllers** and **safety PLCs** that own emergency stops and light curtains.

### 1.2 Talking to the factory

| Technology | Strength | Typical role |
|------------|----------|--------------|
| **OPC UA** | Rich information model, security, pub/sub | MES ↔ robot / SCADA, modern lines |
| **Modbus TCP** | Simple, ubiquitous | Legacy PLCs, sensors, quick integration |
| **EtherCAT** | Designed for cyclic industrial I/O | Tight motion + I/O; actual bounds depend on the configured controller, topology, devices, and scheduling |

**Design pattern:** ROS 2 nodes handle **perception and motion intent**; a **PLC or safety relay** enforces **hard stops** and **interlocks**. Do not confuse “I can send a Modbus coil” with “I am certified safe.”

### 1.3 Collaborative robots (cobots)

Collaborative operation is a property of the integrated application: robot,
tool, workpiece, motion, sensing, and human access all matter. A robot marketed as
a cobot does not by itself establish that a particular task can run unguarded.
For industrial applications, the published editions checked on 2026-10-18 are
[ISO 10218-1:2025](https://www.iso.org/standard/73933.html) for the robot and
[ISO 10218-2:2025](https://www.iso.org/standard/73934.html) for application/cell
integration. [ISO/TS 15066:2016](https://committee.iso.org/standard/62996.html)
provides supplementary collaborative-operation requirements; its catalog still
identifies that edition as current. Apply the actual system's assessed operating
modes and protective functions rather than treating an ordinary ROS proximity
callback as a safety-rated function.

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

For tight joint-control or I/O deadlines, teams use appropriate real-time Linux
configurations or dedicated motion controllers. A PREEMPT_RT kernel alone does
not establish an end-to-end deadline bound. Account for scheduling priorities,
blocking operations, allocation/page faults, interrupts, middleware, and device
latency, then measure deadline misses under the declared workload.

CPU isolation is one possible tuning mechanism, not a universal requirement.
The current [Linux CPU-isolation documentation](https://docs.kernel.org/admin-guide/cpu-isolation.html)
recommends runtime-tunable cgroup v2 isolated cpuset partitions for scheduler
isolation; `isolcpus=domain` is a less flexible alternative. Follow the installed
kernel's interfaces. A vendor drive may own the fast inner loop while ROS supplies
slower setpoints; document those actual rates and interfaces.

### 2.3 Docker and reproducibility

**Why Docker for robotics**

* Reproduce userspace dependencies for each target architecture. An x86-64 laptop
  image is not automatically an ARM64 Jetson image; a
  [multi-platform image](https://docs.docker.com/build/building/multi-platform/)
  can contain different architecture-specific images behind one tag.
* Pin **distro + dependencies** for Nav2 / perception stacks.

**Caveats**

* GPU and **NVIDIA Container Toolkit** for CUDA / TensorRT inside containers.
* **Device nodes** such as `/dev/video*` and `/dev/gpiochip*` need explicit access
  and permissions. **SocketCAN** uses network interfaces and sockets; consider
  its network namespace/interface access instead of assuming a `/dev/can` node.
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

**Isaac Sim** targets **high-fidelity rendering**, **sensor simulation**, **synthetic data**, and **Isaac Lab** for RL. The [ROS 2 bridge](../Isaac%20Sim%20and%20Isaac%20Lab/Lecture-08.md) connects to the same graph patterns you learned in [Lecture 1](../Advanced%20Robot%20Operating%20System/Lecture-01.md), but the authoring environment is **USD-based** and GPU-heavy.

| Need | Often choose |
|------|----------------|
| Nav2 + LiDAR bring-up, lightweight CI | Gazebo Sim |
| Photoreal data, RL, digital twin | Isaac Sim |

---

## 4. Sim-to-real checklist

1. **Clock:** Nodes using simulation time consume `/clock`; observations, TF, and
   age checks must share that clock and reset epoch. Wall-clock performance is a
   separate measurement; simulation time need not advance at real-time speed.
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

<details markdown="1"><summary>Answer guidance</summary>

1. The application's protective functions need their own specified and validated
   implementation. Successful motion planning or ordinary ROS message delivery
   does not establish the behavior of the stop/interlock chain.
2. The container may start successfully while its camera device is absent or
   inaccessible. Test actual frame delivery, timestamps, permissions, and recovery.
3. Choose based on the required sensors, rendering, task integrations, supported
   hardware, and validated behavior. Photoreal synthetic-data workflows may favor
   Isaac; a lightweight navigation smoke test may favor Gazebo. Neither choice
   demonstrates physical transfer without additional evidence.

</details>

For a worked controller/actuator example, complete
[torque, gravity, and limits](../Robot%20Dynamics%20and%20Task%20Execution/Lecture-01-Dynamics.md)
before choosing a drive from a learned policy's action rate.

---

## Resources

* **Structured courses:** See **Recommended courses** above; the [main Robotics Application guide](../Guide.md) lists all tracks.
* **ROS-Industrial:** [rosindustrial.org](https://rosindustrial.org/) and distro-specific tutorials.
* **Gazebo:** [gazebosim.org/docs](https://gazebosim.org/docs)
* **Isaac Sim:** [release compatibility and official sources](../Isaac%20Sim%20and%20Isaac%20Lab/Compatibility.md)

---

## Next in this roadmap

* Previous: [Advanced Robot Operating System](../Advanced%20Robot%20Operating%20System/Lecture-01.md)
* Next: [Advanced Perception and AI for Robotics](../Advanced%20Perception%20and%20AI%20for%20Robotics/Lecture-01.md)
