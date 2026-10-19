# Lecture 3: Advanced Perception and AI for Robotics

**Before this survey:** work through [camera geometry and uncertainty](../Robotics%20Foundations/Lecture-04.md) and its CPU projection/filter lab. A detector's pixel output is not yet a robot-frame pose; calibration, depth, timestamp, and uncertainty assumptions connect the two. **Artifact:** a perception-to-estimation trace with known frame/unit checks, held-out quality metrics, and capture-to-use age.

**Worked practice:** the [capture-time tracking lab](Tracking-Lab.md) derives a
camera-to-world measurement, propagates uncertainty, runs Kalman prediction and
gating, and exposes an identity ambiguity. Its optional TF2 experiment was
executed with the actual Jazzy library; no detector checkpoint is required.

## Overview

This lecture has **three parts**:

* **Part A — Applied perception, tracking, and control (ROS 2):** Closing the loop from **pixels** to **actuation** with standard messages, estimators, and middleware—what many **applied research / perception engineer** roles require.
* **Part B — Advanced perception:** Deep learning **2D/3D** perception, **semantics**, **VIO**, and **multi-modal** sensing for manipulation and navigation.
* **Part C — Robot learning:** **RL**, **imitation**, **foundation models**, and **whole-body / legged** control—how learning stacks sit **above** or **beside** classical ROS 2 stacks.

**By the end of this lecture you should be able to:**

* Sketch a **ROS 2 graph** for detector → tracker → planner/controller, with **`vision_msgs`** and **TF2**.
* Explain **Kalman prediction + association** for multi-object tracking at a systems level.
* Name **when** learned depth fails and **geometry** (stereo, structure-from-motion) still matters.
* Describe **sim-to-real** levers (randomization, latency, calibration) for **RL** policies.
* Place **LLM / VLM** planners as **high-level** supervisors over **ROS 2** primitives.

---

## Part A — Applied perception, tracking, and control (ROS 2 focus)

This block maps **detection → tracking → estimation → actuation** to **ROS 2** primitives: nodes, topics, QoS, launch, `ros2 bag`, and bridges to **non-ROS** services.

### A.1 Detection in the ROS graph

**Goal:** Turn a **camera** or **LiDAR** stream into **stable, typed messages** for downstream nodes.

| Piece | Role |
|-------|------|
| `sensor_msgs/Image` | Raw image; `image_transport` can select transports with different wire representations |
| `cv_bridge` | Convert ROS images ↔ OpenCV; inspect encoding, channel order, row stride, ownership, and copies |
| `vision_msgs/Detection2D` / `Detection3D` | Typed detections; retain the source frame and acquisition timestamp |

`cv_bridge` is not a promise of zero-copy conversion. `message_filters` can pair
messages according to their timestamps and policy; it does not synchronize
hardware clocks or repair a receive-time stamp mislabeled as capture time.
Inspect the relevant [image bridge](https://github.com/ros-perception/vision_opencv)
and [message filter](https://github.com/ros2/message_filters) implementation for
your release.

**Deployment path:** Train in PyTorch → export **ONNX** → **TensorRT** on Jetson → thin ROS 2 node that only runs inference and publishes.

### A.2 Multi-object tracking (MOT)

**Core loop:** predict each track using a motion model, such as constant velocity
inside a Kalman filter; gate unlikely measurements; solve an assignment problem;
update matched tracks; and manage unmatched detections/tracks. The Kalman filter
and the constant-velocity model are different layers of the same example.

**ByteTrack** is a tracking-by-detection method that associates high-confidence
detections and then uses lower-confidence detections to recover additional
matches; its association is not inherently a learned appearance model. Track
identity can remain ambiguous when objects overlap or leave view. In ROS 2,
publish track IDs and markers for RViz2 to expose these failures.
[ByteTrack paper](https://arxiv.org/abs/2110.06864)

**Roadmap deep dive:** [Phase 3 — Multi-Object Tracking guide](../../../Phase%203%20-%20Artificial%20Intelligence/Track%20A%20-%20Hardware%20and%20Edge%20AI/4.%20Sensor%20Fusion/multi-object-tracking/Guide.md) (Kalman + assignment + ROS 2 patterns).

### A.3 State estimation and `robot_localization`

Use the sensor/frame contracts in [the fusion section](../Advanced%20Robot%20Operating%20System/Lecture-01.md#6-sensor-fusion-robot_localization)
before combining wheel odometry, IMU, visual odometry, or geographically transformed
GPS. A wrong `odom` → `base_link` or an incorrect transform timestamp corrupts
both tracking and Nav2. Two outputs derived from the same image are not
automatically independent evidence.

### A.4 Semantic layer and behavior trees

Nav2 already uses **BehaviorTree.CPP**. For **semantic** goals (“only drive on labeled floor”), feed **segmentation** into **costmap plugins** or **BT conditions** that branch on class labels.

### A.5 Control and visual servoing

**Visual servoing:** Regulate **image features** (points, lines) to desired positions; **control law** outputs **twist** or **joint velocities**. Always **transform** setpoints through **TF2** (`geometry_msgs`) so the controller operates in the **correct frame**.

**Aerial / PX4:** PX4's [uXRCE-DDS client/agent bridge](https://docs.px4.io/main/en/middleware/uxrce_dds)
connects selected PX4 messages with DDS/ROS 2. This is not another name for the
micro-ROS client stack. Pin compatible PX4/message definitions and check frame
conventions: PX4's NED/FRD conventions and ROS's common ENU/FLU conventions need
explicit conversion, including orientation. Begin with SITL and documented
offboard timing/failsafe behavior before interpreting a published setpoint as
an accepted or executed command.

### A.6 Monocular geometry

Pure geometric reconstruction from monocular images has a scale ambiguity. Additional information such as inertial measurements under suitable motion and calibration, known dimensions, or learned priors can constrain scale; it does not make every monocular estimate metrically correct. For a VIO package, verify the actual sensor configuration, ROS interface/version, initialization requirements, output frames, and covariance before fusion.

### A.7 Simulation and sim-to-real

* **Gazebo** + [`ros_gz`](https://github.com/gazebosim/ros_gz): ground robots, sensors.
* **PX4 SITL:** aerial stacks before flight.

**Sim-to-real:** Inject **latency**, **noise**, and **miscalibration** in sim before claiming hardware readiness.

### A.8 Service integration (FastAPI, NATS)

**Pattern:** keep the sensing/control path separate from dashboard HTTP and
analytics events. A ROS 2 graph is not inherently real-time. Bound bridge queues,
carry acquisition timestamps, and avoid blocking the executor callback with an
HTTP request. Measure the resulting age and missed deadlines under load.

**QoS:** work through the [executed compatibility and backlog lab](../Advanced%20Robot%20Operating%20System/Integration-Lab.md)
for sensor streams versus commands.

### A.9 GPU, Docker, GStreamer

**NVIDIA Container Toolkit** for GPU nodes in Docker. **`gscam`** or **pipeline shims** when `v4l2` is not enough for **camera ingest**.

### Recommended courses (Part A)

* [Kalman Filters](https://app.theconstruct.ai/courses/kalman-filters-52/)
* [ROS 2 Perception](https://app.theconstruct.ai/courses/ros-2-perception-in-5-days-239/)
* [Behavior Trees for ROS 2](https://app.theconstruct.ai/courses/behavior-trees-for-ros2-131/)
* [ROS 2 Control Framework](https://app.theconstruct.ai/courses/ros-2-control-framework-jazzy-404/)
* [Programming Drones with ROS](https://app.theconstruct.ai/courses/programming-drones-with-ros-24/)
* [State Estimation and Localization for Self-Driving Cars](https://www.coursera.org/learn/state-estimation-localization-self-driving-cars) (Coursera)
* ETH [Autonomous Mobile Robots](https://www.edx.org/learn/autonomous-robotics/eth-zurich-autonomous-mobile-robots) (edX)
* ETH RSL [Programming for Robotics (ROS)](https://rsl.ethz.ch/education-students/lectures/ros.html)

### Canonical links (Part A)

* [ROS 2 Jazzy tutorial sources](https://github.com/ros2/ros2_documentation/tree/jazzy/source/Tutorials) · [QoS](https://github.com/ros2/ros2_documentation/blob/jazzy/source/Concepts/Intermediate/About-Quality-of-Service-Settings.rst) · [distribution choices](../Advanced%20Robot%20Operating%20System/Integration-Lab.md)
* [Nav2](https://navigation.ros.org/) · [Behavior trees in Nav2](https://navigation.ros.org/behavior_trees/index.html)
* [ros2_control](https://control.ros.org/)
* [`vision_msgs`](https://github.com/ros-perception/vision_msgs) · [cv_bridge source](https://github.com/ros-perception/vision_opencv)
* Barfoot, *State Estimation for Robotics* — [resource page](http://asrl.utias.utoronto.ca/~tdb/bib.html)
* [Gazebo](https://gazebosim.org/docs) · [ros_gz](https://github.com/gazebosim/ros_gz)
* [PX4 ROS 2](https://docs.px4.io/main/en/ros2/user_guide.html) · [MAVSDK](https://mavsdk.mavlink.io/)
* [Docker + ROS 2](https://github.com/osrf/docker_images) — choose an image for the same distribution and target architecture
* [FastAPI](https://fastapi.tiangolo.com/) · [NATS](https://docs.nats.io/)

### How Part A connects to other lectures

| Topic | Where |
|-------|--------|
| Nav2, SLAM, `robot_localization` | [Lecture 1 — Advanced ROS](../Advanced%20Robot%20Operating%20System/Lecture-01.md), [Lecture 2 — Industrial](../Industrial%20and%20Embedded%20Robotics/Lecture-01.md) |
| Multi-robot | [Lecture 4 — Multi-Robot](../Multi-Robot%20Systems%20and%20Swarm%20Robotics/Lecture-01.md) |

### Projects (Part A)

1. **Detector → tracker → RViz2:** `vision_msgs` + track markers + `ros2 bag`.
2. **PX4 SITL or Gazebo + Nav2:** Compare **command latency** sim vs hardware.
3. **Bridge:** ROS 2 state → **FastAPI** + **NATS** events; measure end-to-end delay.

---

## Part B — Advanced perception and AI (expanded track)

### B.1 Deep learning–based perception

* **2D detection:** YOLO-family, RT-DETR, etc.—optimize for **latency** on Jetson (TensorRT).
* **6D pose:** FoundationPose, DenseFusion-style methods for **grasp**—outputs must feed **MoveIt 2** via **TF** and collision-aware planning.
* **3D point clouds:** **Open3D** / **PCL** for classical geometry; **PointNet++** / **voxel** nets for learned segmentation in structured scenes.

### B.2 VIO / SLAM

**VIO** fuses **IMU** (high rate, biased) with **camera** (lower rate, rich). Failure modes: **motion blur**, **rolling shutter**, **textureless** regions. Always compare against **wheel odometry** or **mocap** when possible.

### B.3 Semantics and scene graphs

**Semantic segmentation** (e.g. drivable vs obstacle) feeds **Nav2** costmaps. **3D scene graphs** attach **objects** and **relations** for **task planning** and **HRI** (“the cup on the left table”).

**Open-vocabulary** detectors (Grounding DINO, CLIP-based) reduce retraining but require **latency** and **grounding** validation on your robot.

### B.4 Tactile and multi-modal sensing

**Tactile** arrays estimate **slip** and **contact**; fusion with **vision** helps **in-hand** manipulation. **Audio** can flag **collision** or **motor** anomalies—treat as **asynchronous** cues to **supervisors**, not hard real-time control unless validated.

### Resources (Part B)

* [Open3D](http://www.open3d.org/docs/)
* Siciliano et al., *Robotics: Modelling, Planning and Control*
* Berkeley Robot Sensing (BRS) line of work (papers)

### Projects (Part B)

* **6D pose + grasp:** Jetson Orin + table-top objects + MoveIt 2.
* **VIO benchmark:** ORB-SLAM3 vs ground truth.
* **Open-vocabulary pick-and-place:** language → detection → grasp.

---

## Part C — Robot learning and autonomous behaviors

> **Go deeper:** this part is a survey. The full treatment — behavioral cloning and DAgger, flow-matching policies, policy gradients, PPO, SAC, model-based and offline RL, and RL post-training of a VLA — is the [Deep RL for Robot Learning](../Deep%20RL%20for%20Robot%20Learning/README.md) special course.

### C.1 Reinforcement learning

**Sim-to-real:** Randomize **dynamics**, **friction**, **sensor noise**, **latency**; **domain randomization** reduces **overfitting** to one simulator build.

**Algorithms:** PPO and SAC are common for continuous control; **TD-MPC** and **model-based** variants sample-efficiently in some setups.

**Frameworks:** Stable-Baselines3, RLlib, **Isaac Lab** for GPU-heavy training.

### C.2 Imitation and offline RL

**Behavior cloning** can fail **out of distribution**; **DAgger** collects expert labels on states visited by the learner to address this distribution shift. **Diffusion Policy** models multimodal action sequences; its denoising procedure does not impose hard smoothness, velocity, acceleration, or collision constraints. Validate the executed trajectory and object-level outcome separately. See the [Diffusion Policy paper](https://arxiv.org/abs/2303.04137) and the [dynamics and task-execution course](../Robot%20Dynamics%20and%20Task%20Execution/README.md).

### C.3 Foundation models for robotics

**Vision-language-action (VLA)** models aim to map **images + language** to **actions**. In deployment, **LLMs** often act as **high-level planners** that call **ROS 2** skills (navigate, pick, place)—**verify** each step with **executable** checks.

### C.4 Whole-body and legged control

**WBC** coordinates **many DoF** under constraints (contacts, COM). **Legged** systems blend **model-based** (MPC, WBC) with **RL** policies. **Contact-rich** manipulation uses **hybrid** force/position control.

### Resources (Part C)

* [Isaac Lab compatibility and source pins](../Isaac%20Sim%20and%20Isaac%20Lab/Compatibility.md) — full treatment in the [Isaac Sim and Isaac Lab](../Isaac%20Sim%20and%20Isaac%20Lab/README.md) special course
* Sutton & Barto, *Reinforcement Learning: An Introduction*
* [Lerobot](https://github.com/huggingface/lerobot) (Hugging Face)

### Projects (Part C)

* **Sim-to-real locomotion:** Isaac Sim → real quadruped (document transfer).
* **Diffusion policy:** Teleop demos → train → evaluate on arm.
* **LLM task planner:** LLM outputs **skill sequence** executed via ROS 2 actions/services.

---

## Self-check (whole lecture)

**Part A:** (1) What does `vision_msgs` buy you vs a custom float array topic? (2) Name one reason to keep **FastAPI** out of the **critical** DDS callback path.

**Part B:** When does **monocular depth** fail outdoors at **high speed**?

**Part C:** What is **one** failure mode of **behavior cloning** without **DAgger**?

**Answer guidance:** Part A: typed detections provide explicit structure that tools and downstream consumers can inspect; they still need a shared semantic contract. Blocking an HTTP request inside the critical callback can delay sensor processing or control. Part B: blur, changing exposure, weak texture, and distribution shift can degrade depth; fast motion also makes an old estimate spatially wrong. Part C: small imitation errors can move the policy into states absent from its demonstrations, where errors accumulate; DAgger addresses coverage by gathering expert labels on policy-visited states, under its assumptions.

---

## Next in this roadmap

* Previous: [Industrial and Embedded Robotics](../Industrial%20and%20Embedded%20Robotics/Lecture-01.md)
* Next: [Multi-Robot Systems and Swarm Robotics](../Multi-Robot%20Systems%20and%20Swarm%20Robotics/Lecture-01.md)
