# Lecture 3: Advanced Perception and AI for Robotics

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
| `sensor_msgs/Image` | Raw image (often compressed with `image_transport`) |
| `cv_bridge` | Convert ROS images ↔ OpenCV without copy mistakes |
| `vision_msgs/Detection2D` / `Detection3D` | Standard bounding boxes + class ids; use **message_filters** for sync |

**Deployment path:** Train in PyTorch → export **ONNX** → **TensorRT** on Jetson → thin ROS 2 node that only runs inference and publishes.

### A.2 Multi-object tracking (MOT)

**Core loop:** **Predict** each track with a **Kalman** (or constant-velocity) model → **associate** detections to tracks (Hungarian / IoU / Mahalanobis gating) → **create** / **delete** tracks.

**Learning-assisted** trackers (e.g. **ByteTrack**-style) add **association** robustness when detections flicker. In ROS 2, publish **track IDs** and **markers** for RViz2 so debugging is visual.

**Roadmap deep dive:** [Phase 3 — Multi-Object Tracking guide](../../../Phase%203%20-%20Artificial%20Intelligence/Track%20A%20-%20Hardware%20and%20Edge%20AI/4.%20Sensor%20Fusion/multi-object-tracking/Guide.md) (Kalman + assignment + ROS 2 patterns).

### A.3 State estimation and `robot_localization`

Same mathematics as Lecture 1, applied to **perception-driven** systems: fuse **wheel odom**, **IMU**, **visual odometry**, or **GPS** (outdoor). **TF2** must stay **consistent**—a wrong `odom` → `base_link` corrupts both tracking and Nav2.

### A.4 Semantic layer and behavior trees

Nav2 already uses **BehaviorTree.CPP**. For **semantic** goals (“only drive on labeled floor”), feed **segmentation** into **costmap plugins** or **BT conditions** that branch on class labels.

### A.5 Control and visual servoing

**Visual servoing:** Regulate **image features** (points, lines) to desired positions; **control law** outputs **twist** or **joint velocities**. Always **transform** setpoints through **TF2** (`geometry_msgs`) so the controller operates in the **correct frame**.

**Aerial / PX4:** Use [PX4 ↔ ROS 2](https://docs.px4.io/main/en/ros2/user_guide.html) (micro-ROS / uXRCE-DDS) rather than reinventing the autopilot; ROS 2 supplies **missions**, **offboard** setpoints, or **perception** hooks.

### A.6 Monocular geometry

**Monocular** systems lack absolute scale; **IMU** or **known object size** provides scale. **VIO** packages (ORB-SLAM3, OpenVINS, Kimera-VIO) expose **ROS** interfaces—treat outputs as **noisy** and **rate-limited** for fusion.

### A.7 Simulation and sim-to-real

* **Gazebo** + [`ros_gz`](https://github.com/gazebosim/ros_gz): ground robots, sensors.
* **PX4 SITL:** aerial stacks before flight.

**Sim-to-real:** Inject **latency**, **noise**, and **misp calibration** in sim before claiming hardware readiness.

### A.8 Service integration (FastAPI, NATS)

**Pattern:** ROS 2 owns **real-time-ish** sensing and control; **FastAPI** exposes **REST/WebSocket** for dashboards; **NATS** fans out **events** to analytics. **Bridge** with small nodes; **do not** starve the DDS thread with blocking HTTP.

**QoS:** [DDS tuning](https://docs.ros.org/en/humble/Concepts/About-Quality-of-Service-Settings.html) for sensor streams vs commands.

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

* [ROS 2 Humble tutorials](https://docs.ros.org/en/humble/Tutorials.html) · [QoS](https://docs.ros.org/en/humble/Concepts/About-Quality-of-Service-Settings.html) · [`ros2 bag`](https://docs.ros.org/en/humble/Tutorials/Beginner-CLI-Tools/Recording-And-Playing-Back-Data/Recording-And-Playing-Back-Data.html)
* [Nav2](https://navigation.ros.org/) · [Behavior trees in Nav2](https://navigation.ros.org/behavior_trees/index.html)
* [ros2_control](https://control.ros.org/)
* [`vision_msgs`](https://github.com/ros-perception/vision_msgs) · [cv_bridge](https://docs.ros.org/en/humble/Tutorials/Advanced/Cvbridge/Cvbridge.html)
* Barfoot, *State Estimation for Robotics* — [resource page](http://asrl.utias.utoronto.ca/~tdb/bib.html)
* [Gazebo](https://gazebosim.org/docs) · [ros_gz](https://github.com/gazebosim/ros_gz)
* [PX4 ROS 2](https://docs.px4.io/main/en/ros2/user_guide.html) · [MAVSDK](https://mavsdk.mavlink.io/)
* [Docker + ROS 2](https://github.com/osrf/docker_images) · [Docker how-to](https://docs.ros.org/en/humble/How-To-Guides/Run-2-nodes-in-single-or-separate-docker-containers.html)
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

### C.1 Reinforcement learning

**Sim-to-real:** Randomize **dynamics**, **friction**, **sensor noise**, **latency**; **domain randomization** reduces **overfitting** to one simulator build.

**Algorithms:** PPO and SAC are common for continuous control; **TD-MPC** and **model-based** variants sample-efficiently in some setups.

**Frameworks:** Stable-Baselines3, RLlib, **Isaac Lab** for GPU-heavy training.

### C.2 Imitation and offline RL

**Behavior cloning** is fragile **out of distribution**; **DAgger** reduces **covariate shift** by mixing expert and policy data. **Diffusion Policy** outputs **smooth** multi-modal action trajectories.

### C.3 Foundation models for robotics

**Vision-language-action (VLA)** models aim to map **images + language** to **actions**. In deployment, **LLMs** often act as **high-level planners** that call **ROS 2** skills (navigate, pick, place)—**verify** each step with **executable** checks.

### C.4 Whole-body and legged control

**WBC** coordinates **many DoF** under constraints (contacts, COM). **Legged** systems blend **model-based** (MPC, WBC) with **RL** policies. **Contact-rich** manipulation uses **hybrid** force/position control.

### Resources (Part C)

* [Isaac Lab](https://docs.omniverse.nvidia.com/isaacsim/latest/isaac_lab_tutorials/index.html)
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

---

## Next in this roadmap

* Previous: [Industrial and Embedded Robotics](../Industrial%20and%20Embedded%20Robotics/Lecture-01.md)
* Next: [Multi-Robot Systems and Swarm Robotics](../Multi-Robot%20Systems%20and%20Swarm%20Robotics/Lecture-01.md)
