**4. Robotics (12-18 months)**

<div class="course-identity robotics" markdown="1">
<div class="course-identity__icon">ROS</div>
<div markdown="1">
<p class="course-identity__eyebrow">Track 5D · Robotics</p>
<p class="course-identity__title">Integrate perception, planning, ROS 2, real-time constraints, and embodied AI systems.</p>
<p class="course-identity__meta">Artifact: robot subsystem demo · Measure: loop rate, latency, safety, robustness</p>
</div>
</div>


This phase is organized into **four lecture tracks** (each has its own folder and `Lecture-01.md`). Start with **Advanced Robot Operating System**, then proceed in order or jump by topic.

| # | Track | Folder / lecture |
|---|--------|------------------|
| 1 | **Advanced Robot Operating System** (ROS / ROS 2, Nav2, SLAM, MoveIt 2) | [Advanced Robot Operating System/Lecture-01.md](Advanced%20Robot%20Operating%20System/Lecture-01.md) |
| 2 | **Industrial and Embedded Robotics** (ROS-I, embedded, Docker, Gazebo / Isaac Sim) | [Industrial and Embedded Robotics/Lecture-01.md](Industrial%20and%20Embedded%20Robotics/Lecture-01.md) |
| 3 | **Advanced Perception and AI for Robotics** (applied ROS 2 perception loop, deep learning perception, robot learning) | [Advanced Perception and AI for Robotics/Lecture-01.md](Advanced%20Perception%20and%20AI%20for%20Robotics/Lecture-01.md) |
| 4 | **Multi-Robot Systems and Swarm Robotics** | [Multi-Robot Systems and Swarm Robotics/Lecture-01.md](Multi-Robot%20Systems%20and%20Swarm%20Robotics/Lecture-01.md) |

Each `Lecture-01.md` is a **full lecture**: learning objectives, worked conceptual sections, tables/diagrams where helpful, **self-check** questions, and links to the next/previous track.

**Extended timeline (18-36 months):** Lectures 3–4 include the **expanded** topics that were previously labeled “Phase 2 (Significantly Expanded)” in older versions of this file (deep perception, robot learning, multi-robot / swarm). All of that material now lives in **Lecture-01.md** for tracks 3 and 4.

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
    * Nvidia [Isaac Sim documentation](https://docs.omniverse.nvidia.com/isaacsim/latest/index.html) — tutorials; see [ROS 2 tutorials](https://docs.omniverse.nvidia.com/isaacsim/latest/ros2_tutorials/index.html) when bridging to ROS 2 work.

* **§3 — Applied perception, tracking, control, drones (maps to [Lecture 3](Advanced%20Perception%20and%20AI%20for%20Robotics/Lecture-01.md), Part A)**
    * [Kalman Filters](https://app.theconstruct.ai/courses/kalman-filters-52/) — estimation intuition tied to mobile robots.
    * [ROS 2 Perception](https://app.theconstruct.ai/courses/ros-2-perception-in-5-days-239/) — sensors and perception stack in ROS 2.
    * [Behavior Trees for ROS 2](https://app.theconstruct.ai/courses/behavior-trees-for-ros2-131/) — ties to Nav2 / high-level behaviors.
    * [ROS 2 Control Framework](https://app.theconstruct.ai/courses/ros-2-control-framework-jazzy-404/) — `ros2_control` and hardware interfaces.
    * [Programming Drones with ROS](https://app.theconstruct.ai/courses/programming-drones-with-ros-24/) — complements PX4/MAVLink self-study from docs.
    * [State Estimation and Localization for Self-Driving Cars](https://www.coursera.org/learn/state-estimation-localization-self-driving-cars) (Coursera) — Kalman, EKF, UKF, particle filters (transferable to tracking and fusion).
    * ETH Zürich [Autonomous Mobile Robots](https://www.edx.org/learn/autonomous-robotics/eth-zurich-autonomous-mobile-robots) (edX) — kinematics, localization, mapping, planning (theory; pair with ROS 2 projects).
    * ETH RSL [Programming for Robotics (ROS)](https://rsl.ethz.ch/education-students/lectures/ros.html) — official ROS 2 lecture materials (free outline/readings; on-campus course; use as a syllabus alongside your own labs).
