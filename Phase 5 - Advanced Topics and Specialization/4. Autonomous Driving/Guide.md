**4. Autonomous Driving (12-24 months)**

**1. Openpilot and Open-Source ADAS**

* **Openpilot Deep Dive:**
    * **Architecture Overview:** Study the Openpilot (comma.ai) architecture—perception (camera-based), planning (longitudinal and lateral control), and actuation. Understand the modular design and Python/C++ codebase.
    * **Hardware and Software Stack:** Learn about supported hardware (comma devices, compatible vehicles), CAN bus integration, and the openpilot software pipeline from camera input to steering/acceleration commands.
    * **Forking and Customization:** Fork Openpilot, understand the build system, and customize for different vehicles or features. Explore the community forks and contributions.

* **Perception for Autonomous Driving:**
    * **Camera-Based Perception:** Understand how Openpilot uses single or multi-camera setups for lane detection, vehicle detection, and driver monitoring. Study the model architecture and training pipeline.
    * **Sensor Fusion (Optional):** Explore extending Openpilot with radar or LiDAR for enhanced perception. Understand sensor calibration and fusion strategies.
    * **End-to-End vs. Modular:** Compare end-to-end neural network approaches with modular pipelines (detection → planning → control). Understand Openpilot's hybrid approach.

* **Planning and Control:**
    * **Longitudinal Control:** Study adaptive cruise control (ACC), stop-and-go, and lead vehicle following. Understand the MPC (Model Predictive Control) or PID-based control logic.
    * **Lateral Control:** Learn lane-keeping and lane-change logic. Understand path planning from perceived lane geometry.
    * **Safety and Fallbacks:** Study driver monitoring, take-over requests, and fail-safe behaviors in Openpilot.

**Resources:**

* **Openpilot GitHub:** https://github.com/commaai/openpilot — Source code, documentation, and community.
* **tinygrad:** Openpilot uses tinygrad for neural network inference. See [tinygrad](tinygrad/) (learning materials) and [tinygrad-source](tinygrad-source/) (source code).
* **comma.ai Blog and Videos:** Technical deep dives and development updates.
* **"Openpilot Development Guide" (community docs):** Fork documentation and development setup.

**Projects:**

* **Run Openpilot in Simulation:** Use openpilot's tools or CARLA integration to test in simulation.
* **Analyze Openpilot Codebase:** Trace the data flow from camera input to CAN output. Document the perception and control pipeline.
* **Contribute to Openpilot:** Fix a bug, add a feature, or improve documentation for a supported vehicle.


**2. Autonomous Driving Fundamentals**

* **Computer Vision for Driving:**
    * **Lane Detection:** Implement lane detection using traditional (Hough transform, sliding window) and deep learning (LaneNet, U-Net) approaches.
    * **Object Detection:** Study 2D/3D object detection for vehicles, pedestrians, and cyclists. Explore YOLO, CenterNet, and BEV (Bird's Eye View) detectors.
    * **Semantic Segmentation:** Use segmentation for drivable area, lane boundaries, and obstacle detection.

* **Planning and Decision Making:**
    * **Behavior Planning:** Understand behavior planning—lane change decisions, merging, intersection handling. Study finite state machines and behavior trees.
    * **Motion Planning:** Explore path planning algorithms (A*, RRT, hybrid A*) and trajectory optimization for smooth, collision-free paths.
    * **Prediction:** Learn to predict other agents' trajectories for safe planning. Study constant velocity models, physics-based models, and learning-based prediction.

* **Control Theory:**
    * **Vehicle Dynamics:** Understand bicycle model, tire models, and vehicle state estimation.
    * **MPC and LQR:** Apply Model Predictive Control and Linear Quadratic Regulators for path tracking and speed control.
    * **Stanley Controller:** Study the Stanley controller used in early autonomous vehicles and Openpilot.

**Resources:**

* **"Autonomous Driving in the Real World" by Shaoshan Liu et al.:** Comprehensive autonomous driving textbook.
* **CARLA Simulator:** Open-source simulator for autonomous driving research.
* **Apollo Auto (Baidu):** Reference implementation for full-stack autonomous driving.

**Projects:**

* **Build a Lane-Keeping Assist:** Implement a minimal lane-keeping system using a camera and steering control.
* **Compare Openpilot with Apollo:** Analyze architectural differences between Openpilot and Apollo for perception and planning.
