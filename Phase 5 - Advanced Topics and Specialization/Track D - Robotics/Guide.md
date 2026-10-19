# Phase 5 — Track D: Robotics

<div class="course-identity robotics" markdown="1">
<div class="course-identity__icon">ROS</div>
<div markdown="1">
<p class="course-identity__eyebrow">Track 5D · Robotics</p>
<p class="course-identity__title">Integrate perception, planning, ROS 2, real-time constraints, and embodied AI systems.</p>
<p class="course-identity__meta">Artifact: robot subsystem demo · Measure: loop rate, latency, safety, robustness</p>
</div>
</div>


**Contribution to the physical-AI SoC:** robots and drones define the workloads and timing constraints the chip must serve. Use the VLA and Isaac courses to measure model quality, observation-to-action latency, memory, energy, and controller requirements. Carry that evidence into the [custom accelerator, compiler, Linux, and wireless capstone](../Track%20F%20-%20AI%20Chip%20Design/Guide.md#physical-ai-soc-capstone).

**New to robotics? Start with [Robotics Foundations](Robotics%20Foundations/README.md).** Six lessons teach frames, kinematics, feedback, camera geometry, uncertainty, planning, and the ROS interface through worked examples and CPU-only labs. You do not need a GPU or simulator to learn these prerequisites. Use its ten-minute diagnostic to skip material you can already demonstrate.

Next, [Robot Dynamics and Task Execution](Robot%20Dynamics%20and%20Task%20Execution/README.md)
connects those basics to torque and current limits, LQR/MPC, swept collision
checks, trajectory timing, and grasp/object-level success. Its four worked
lessons and CPU labs explain why a reachable pose or accurate policy prediction
can still fail during execution.

After the foundations, this phase has **four application lecture tracks** (each has its own folder and `Lecture-01.md`) and deeper robot-learning courses. Continue with **Advanced Robot Operating System**, then choose the application that matches your task.

| # | Track | Folder / lecture |
|---|--------|------------------|
| 0 | **Robotics Foundations** (six beginner lessons; standard-library labs; geometry, feedback, estimation, planning, and ROS contracts) | [Robotics Foundations/README.md](Robotics%20Foundations/README.md) |
| 0+ | **Robot Dynamics and Task Execution** (gravity/actuation, optimal control, timed collision-aware motion, and grasp outcomes) | [Robot Dynamics and Task Execution/README.md](Robot%20Dynamics%20and%20Task%20Execution/README.md) |
| 1 | **Advanced Robot Operating System** (ROS / ROS 2, Nav2, SLAM, MoveIt 2) | [Advanced Robot Operating System/Lecture-01.md](Advanced%20Robot%20Operating%20System/Lecture-01.md) |
| 2 | **Industrial and Embedded Robotics** (ROS-I, embedded, Docker, Gazebo / Isaac Sim) | [Industrial and Embedded Robotics/Lecture-01.md](Industrial%20and%20Embedded%20Robotics/Lecture-01.md) |
| 3 | **Advanced Perception and AI for Robotics** (applied ROS 2 perception loop, deep learning perception, robot learning) | [Advanced Perception and AI for Robotics/Lecture-01.md](Advanced%20Perception%20and%20AI%20for%20Robotics/Lecture-01.md) |
| 4 | **Multi-Robot Systems and Swarm Robotics** | [Multi-Robot Systems and Swarm Robotics/Lecture-01.md](Multi-Robot%20Systems%20and%20Swarm%20Robotics/Lecture-01.md) |
| ★ | **Special course — Isaac Sim and Isaac Lab** (USD, PhysX/Newton, RTX sensors, ROS 2 bridge, Isaac Lab 3.0 training; named 6.1 teaching baseline plus 6.2 GA update and compatibility checks) | [Isaac Sim and Isaac Lab/README.md](Isaac%20Sim%20and%20Isaac%20Lab/README.md) |
| ★ | **Special course — Deep RL for Robot Learning** (imitation, policy gradients, PPO, SAC, model-based and offline RL, RL post-training of a flow-matching VLA; follows UC Berkeley CS 185/285) | [Deep RL for Robot Learning/README.md](Deep%20RL%20for%20Robot%20Learning/README.md) |
| ★ | **Professional course — VLA Foundations to Frontier 2026** (32 detailed lectures: history, architectures, data, training, RL, evaluation, Isaac/ROS deployment, October 13 frontier, CPU labs, two capstones) | [VLA Foundations to Frontier 2026/README.md](VLA%20Foundations%20to%20Frontier%202026/README.md) |
| ★ | **Special course — VLA Optimization and Action-Parity Harness** (deploy and validate Vision-Language-Action policies on real robots) | [VLA Optimization and Action-Parity Harness/README.md](VLA%20Optimization%20and%20Action-Parity%20Harness/README.md) |

Each `Lecture-01.md` is a **full lecture**: learning objectives, worked conceptual sections, tables/diagrams where helpful, **self-check** questions, and links to the next/previous track.

**Optional depth after the working loop:** Lectures 3–4 include the **expanded** topics that were previously labeled “Phase 2 (Significantly Expanded)” in older versions of this file (deep perception, robot learning, multi-robot / swarm). All of that material now lives in **Lecture-01.md** for tracks 3 and 4.

---

**Recommended courses (online)**

Use these **structured courses** alongside each lecture’s **Recommended courses** section. Official ROS 2 and vendor docs stay the source of truth; courses help when you want guided projects, browser-based simulators (e.g. The Construct), or university-style theory. Full catalog (The Construct): [Robotics & ROS courses](https://www.theconstruct.ai/robotigniteacademy_learnros/ros-courses-library/).

* **§1 — ROS 2 core, Nav2, MoveIt (maps to [Lecture 1](Advanced%20Robot%20Operating%20System/Lecture-01.md))**
    * [ROS 2 Basics in Python](https://app.theconstruct.ai/courses/ros2-basics-in-5-days-v2-python-268/) or [ROS 2 Basics in C++](https://app.theconstruct.ai/courses/ros2-basics-in-5-days-c-325/) — pick one language first, then add the other.
    * [TF ROS 2](https://app.theconstruct.ai/courses/tf-ros2-217/) — frames and `robot_state_publisher` patterns.
    * [Intermediate ROS 2](https://app.theconstruct.ai/courses/intermediate-ros2-113/) — launch, parameters, QoS, lifecycle.
    * [ROS 2 Navigation](https://app.theconstruct.ai/courses/ros2-navigation-galactic-109/) and [Advanced ROS 2 Navigation](https://app.theconstruct.ai/courses/advanced-ros2-navigation-116/) — Nav2-style stacks.
    * [ROS 2 Manipulation Basics](https://app.theconstruct.ai/courses/ros2-manipulation-basics-81/) or [ROS 2 Manipulation & Perception](https://app.theconstruct.ai/courses/ros2-manipulation-perception-master-103/) — MoveIt 2–style manipulation.
    * [Robotics Specialization](https://www.coursera.org/specializations/robotics) (University of Pennsylvania) — aerial robotics, planning, perception, mobility, capstone; strong on math and control (often MATLAB in assignments; not ROS-version-specific).

* **§2 — Simulation, embedded Linux, Docker, industrial context (maps to [Lecture 2](Industrial%20and%20Embedded%20Robotics/Lecture-01.md))**
    * [Introduction to Gazebo Sim with ROS 2](https://app.theconstruct.ai/courses/introduction-to-gazebo-ignition-with-ros2-170/) and [Mastering Gazebo Simulator](https://app.theconstruct.ai/courses/mastering-gazebo-simulator-78/) — Gazebo Sim + worlds/models.
    * [Docker for Robotics](https://app.theconstruct.ai/courses/docker-basics-for-robotics-114/) — containerized dev and deploy.
    * [Linux for Robotics](https://app.theconstruct.ai/courses/linux-for-robotics-noetic-185/) — shell, permissions, workflows (ROS 1 in the title; skills transfer directly).
    * [C++ for Robotics](https://app.theconstruct.ai/courses/c-for-robotics-59/) / [Python 3 for Robotics](https://app.theconstruct.ai/courses/python-3-for-robotics-58/) — language prep before ROS 2 nodes.
    * Nvidia [Isaac Sim documentation](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/index.html) — tutorials; see [ROS 2 tutorials](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/ros2_tutorials/index.html) when bridging to ROS 2 work.

* **§3 — Applied perception, tracking, control, drones (maps to [Lecture 3](Advanced%20Perception%20and%20AI%20for%20Robotics/Lecture-01.md), Part A)**
    * [Kalman Filters](https://app.theconstruct.ai/courses/kalman-filters-52/) — estimation intuition tied to mobile robots.
    * [ROS 2 Perception](https://app.theconstruct.ai/courses/ros-2-perception-in-5-days-239/) — sensors and perception stack in ROS 2.
    * [Behavior Trees for ROS 2](https://app.theconstruct.ai/courses/behavior-trees-for-ros2-131/) — ties to Nav2 / high-level behaviors.
    * [ROS 2 Control Framework](https://app.theconstruct.ai/courses/ros-2-control-framework-jazzy-404/) — `ros2_control` and hardware interfaces.
    * [Programming Drones with ROS](https://app.theconstruct.ai/courses/programming-drones-with-ros-24/) — complements PX4/MAVLink self-study from docs.
    * [State Estimation and Localization for Self-Driving Cars](https://www.coursera.org/learn/state-estimation-localization-self-driving-cars) (Coursera) — Kalman, EKF, UKF, particle filters (transferable to tracking and fusion).
    * ETH Zürich [Autonomous Mobile Robots](https://www.edx.org/learn/autonomous-robotics/eth-zurich-autonomous-mobile-robots) (edX) — kinematics, localization, mapping, planning (theory; pair with ROS 2 projects).
    * ETH RSL [Programming for Robotics (ROS)](https://rsl.ethz.ch/education-students/lectures/ros.html) — official ROS 2 lecture materials (free outline/readings; on-campus course; use as a syllabus alongside your own labs).


## Recommended project progression

Start with the [CPU foundation experiments](Robotics%20Foundations/Lab-Guide.md), then a frame- and timestamp-correct ROS 2 loop, a validated Isaac scene, and a tested Isaac Lab task. Train only after action/reset/reward checks pass. Evaluate held-out episodes before exporting a policy. Finish with the [Isaac engineering labs](Isaac%20Sim%20and%20Isaac%20Lab/Lab-Guide.md) and the language-guided robotics capstone; the language service and local controller have separate timing contracts.


## Carry this course into the physical-AI system

**Start from:** One manipulation task and the required observation/action interfaces.

**Build and challenge:** Learn ROS 2 integration, establish an Isaac baseline, then study the matching VLA family. Add RL or multi-robot work only when it answers a demonstrated task limitation.

**Pass forward:** One working robot loop with explicit model/controller timing boundaries. Continue with [the connected course](Isaac%20Sim%20and%20Isaac%20Lab/README.md) and keep the artifact in the [Physical AI Design Studio](../../Physical-AI-Design-Studio.md).
