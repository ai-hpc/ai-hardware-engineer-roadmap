# Module 1 — Autonomous Driving Fundamentals

<div class="course-identity auto-course" style="--course-accent: #db2777; --course-accent-rgb: 219, 39, 119;" markdown="1">
<div class="course-identity__icon">MADF</div>
<div markdown="1">
<p class="course-identity__eyebrow">Deep Dive · Specialization</p>
<p class="course-identity__title">Specialized course identity for Module 1 — Autonomous Driving Fundamentals.</p>
<p class="course-identity__meta">Artifact: specialization case study · Measure: performance, reliability, role fit</p>
</div>
</div>


**Parent:** [Phase 5 — Autonomous Driving](../Guide.md)

**Time:** 3–4 months

**Prerequisites:** Phase 3 (Computer Vision, Sensor Fusion), Phase 1 §4 (C++/CUDA).

---

## Why this comes first

Before studying openpilot's codebase or advanced research, you need the foundational vocabulary: how autonomous vehicles perceive, plan, and control. This module covers the algorithms and theory that every ADAS engineer uses daily.

---

## 1. Computer Vision for Driving

* **Lane detection:**
    * Traditional: Hough transform, sliding window, IPM (inverse perspective mapping).
    * Deep learning: LaneNet, SCNN, U-Net for lane segmentation.
    * Output: polylines or polynomial coefficients for lane boundaries.

* **Object detection:**
    * 2D: YOLO family, CenterNet, FCOS — bounding boxes for vehicles, pedestrians, cyclists.
    * 3D: PointPillars, SECOND (LiDAR); FCOS3D, DETR3D (monocular camera).
    * Bird's Eye View (BEV): lifting 2D camera features to BEV space for unified 3D detection.

* **Semantic segmentation:**
    * Drivable area, lane markings, curbs, obstacles.
    * Models: DeepLab, SegFormer, lightweight edge variants.
    * Panoptic segmentation: combining semantic + instance for complete scene understanding.

* **Depth estimation:**
    * Monocular depth (MiDaS, DepthAnything) — useful when no LiDAR is available.
    * Stereo matching: classic block matching, deep stereo (AANet, RAFT-Stereo).

**Projects:**
* Implement lane detection on a dashcam video using both Hough transform and a U-Net. Compare robustness in curves, shadows, and night.
* Train a YOLO model on KITTI or nuScenes for 2D vehicle detection. Measure mAP and inference FPS on Jetson.

---

## 2. Planning and Decision Making

* **Behavior planning:**
    * High-level decisions: lane keep, lane change, merge, turn, stop.
    * Finite state machines (FSM) and behavior trees.
    * Rule-based vs. learning-based behavior planning.

* **Motion planning:**
    * Path planning: A*, RRT, RRT*, hybrid A* (for non-holonomic vehicles).
    * Trajectory optimization: minimize jerk/acceleration while avoiding collisions.
    * Frenet frame: decompose planning into longitudinal (speed) and lateral (lane offset) components.
    * Lattice planners: discretize trajectory space, evaluate candidates against cost functions.

* **Prediction:**
    * Constant velocity / constant turn-rate models (baseline).
    * Learning-based: TNT, LaneGCN, MTR — multi-modal trajectory prediction.
    * Interaction-aware: graph neural networks for modeling agent-agent interactions.
    * Map-conditioned: use lane geometry and traffic rules to constrain predictions.

**Projects:**
* Implement RRT* for path planning in a 2D grid with obstacles. Extend to a bicycle model for non-holonomic constraints.
* Build a simple behavior FSM: CRUISE → FOLLOW → LANE_CHANGE → STOP. Test in CARLA with traffic.

---

## 3. Control Theory for Autonomous Vehicles

* **Vehicle dynamics:**
    * Bicycle model: front/rear axle, slip angle, yaw rate.
    * Tire models: linear cornering stiffness, Pacejka magic formula (overview).
    * Vehicle state estimation: combine IMU + wheel odometry + GPS for localization.

* **Lateral control:**
    * **Stanley controller:** cross-track error + heading error, used in early AV and openpilot.
    * **Pure pursuit:** geometric path following using a lookahead point.
    * **MPC (Model Predictive Control):** optimize a sequence of steering commands over a horizon, subject to dynamics constraints.

* **Longitudinal control:**
    * PID for speed tracking (adaptive cruise control).
    * MPC for combined speed + following distance.
    * Jerk-limited profiles for passenger comfort.

* **Combined lateral + longitudinal:**
    * LQR (Linear Quadratic Regulator) for path tracking with speed control.
    * Cascaded MPC: separate lateral/longitudinal MPC with coordination.

**Projects:**
* Implement a Stanley controller for lane following in CARLA. Tune gains for different speeds.
* Implement PID-based ACC (Adaptive Cruise Control): maintain target speed, slow for lead vehicle, stop-and-go. Test in CARLA.
* (Advanced) Implement a simple MPC for combined lateral + longitudinal control. Compare smoothness and tracking error vs Stanley + PID.

---

## Resources

| Resource | Why |
|----------|-----|
| [CARLA Simulator](https://carla.org/) | Test all algorithms in simulation before touching real hardware |
| *Autonomous Driving in the Real World* (Shaoshan Liu et al.) | Comprehensive AV textbook |
| [Apollo Auto](https://github.com/ApolloAuto/apollo) | Reference full-stack AV for architectural comparison |
| [KITTI](http://www.cvlibs.net/datasets/kitti/) / [nuScenes](https://www.nuscenes.org/) | Benchmark datasets |
| Phase 3 — Computer Vision, Sensor Fusion | Prerequisite modules in this roadmap |

---

## Next

→ **[Module 2 — openpilot Reference Stack](../2.%20openpilot%20Reference%20Stack/Guide.md)** — Study a real, production-deployed ADAS end-to-end.
