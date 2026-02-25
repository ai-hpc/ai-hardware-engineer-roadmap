**4. Autonomous Driving (12-24 months)**

This phase uses **Openpilot** (comma.ai) as a reference implementation for ADAS—perception, planning, and control. Openpilot is open-source, well-documented, and uses tinygrad for inference, making it an ideal case study for this curriculum.

> **📊 [Flow Diagram](flow-diagram.md)** — End-to-end pipeline from perception → planning → control → actuation.
> **📷 [camerad Guide](camerad/Guide.md)** — Camera capture, ISP, auto exposure, VisionIpc.

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


**Phase 2 (Significantly Expanded): Autonomous Driving (24-48 months)**

**1. Advanced Sensor Hardware and Calibration**

* **Sensor Technologies for Autonomous Driving:**
    * **LiDAR Deep Dive:**  Understand LiDAR working principles—mechanical spinning (Velodyne), solid-state (Livox, Innoviz), FMCW LiDAR (Aeva, Luminar). Evaluate range, resolution, FoV, and latency trade-offs for production ADAS.
    * **Radar Systems:**  Study automotive radar—frequency bands (24 GHz, 77 GHz), FMCW modulation, range-Doppler processing, and angular resolution. Understand how radar complements cameras and LiDAR for all-weather sensing.
    * **Camera Systems:**  Compare sensor types (CCD vs. CMOS, global vs. rolling shutter), optics (FoV, focal length, aperture), and ISP pipelines for automotive imaging. Study HDR techniques for handling high-contrast scenes.

* **Multi-Sensor Calibration:**
    * **Intrinsic Calibration:**  Calibrate camera intrinsics (focal length, principal point, distortion coefficients) using checkerboard or ArUco patterns with OpenCV or Kalibr.
    * **Extrinsic Calibration:**  Calibrate the spatial transforms between multiple sensors (camera-LiDAR, camera-radar, LiDAR-IMU) using target-based and targetless methods (LI-Calib, ACSC).
    * **Temporal Calibration:**  Align sensor timestamps across different hardware clocks using PPS signals, IEEE 1588 PTP, or cross-correlation of observed events. Understand the impact of time misalignment on fusion accuracy.

* **Sensor Simulation:**
    * **Synthetic Data Generation:**  Use CARLA, SUMO+CARLA, or Nvidia DRIVE Sim to generate photorealistic synthetic training data with ground-truth labels—bounding boxes, semantic masks, depth, and radar detections.
    * **Sensor Model Simulation:**  Model sensor physics in simulation—LiDAR ray casting, radar multipath, and camera noise models—to reduce the sim-to-real gap in training data.
    * **Scenario Generation:**  Create corner-case scenarios (fog, rain, night, occlusions, near-miss events) in simulation that are rare or unsafe to collect in real driving.

**Resources:**

* **"3D Object Detection for Autonomous Driving: A Review" (survey paper):**  Comprehensive review of LiDAR, camera, and fusion-based 3D detection methods.
* **[Kalibr — Multi-Sensor Calibration](../../Phase 4 - Nvidia Jetson and Edge AI/4. Sensor Fusion/Kalibr/Guide.md):**  Complete practical guide to Kalibr (ETH Zurich). Covers camera intrinsic calibration, multi-camera extrinsic calibration, camera-IMU spatial+temporal calibration, Allan variance for IMU noise modelling, output interpretation, and integration with BEVFusion, ORB-SLAM3, and OpenVINS. Essential before any sensor fusion work.
* **CARLA Simulator:**  Open-source autonomous driving simulator with sensor models, traffic scenarios, and Python API.

**Projects:**

* **Multi-Sensor Calibration Rig:**  Build a calibration target and calibrate a camera-LiDAR pair using Kalibr or ACSC. Verify calibration quality by projecting LiDAR points onto the camera image.
* **CARLA Synthetic Dataset:**  Generate a 10,000-frame synthetic dataset in CARLA with varying weather, lighting, and traffic density. Train and evaluate a 3D object detector on synthetic vs. real data.
* **Radar-Camera Fusion:**  Implement a simple radar-camera fusion pipeline that associates radar detections with camera bounding boxes for velocity-augmented object detection.


**2. Production Perception and Prediction Pipelines**

* **BEV (Bird's-Eye View) Perception:**
    * **BEV Transformers:**  Study BEVFormer, BEVFusion, and Tesla's Occupancy Network for multi-camera Bird's-Eye View perception. Understand attention-based cross-view feature lifting from perspective to BEV space.
    * **3D Occupancy Prediction:**  Implement voxel-based 3D occupancy prediction (Occ3D, SurroundOcc) as an alternative to explicit object detection—represent the environment as a dense 3D grid of occupied voxels.
    * **Temporal Fusion:**  Incorporate temporal information into BEV models using recurrent architectures or long-range temporal attention for tracking implicit object states across frames.

* **Agent Prediction and Behavior Modeling:**
    * **Trajectory Prediction:**  Implement multi-modal trajectory prediction models (TNT, MTR, Wayformer) that predict multiple plausible future trajectories for other agents with probability scores.
    * **Social Force and Interaction Models:**  Model agent-agent interactions using social force models, graph neural networks (GRIP, HYPER), and transformer-based interaction encoders.
    * **Map-Conditioned Prediction:**  Incorporate HD map information (lane geometry, traffic rules) into trajectory prediction using VectorNet or MapTR-style map encoding.

* **HD Maps and Map-Based Systems:**
    * **HD Map Components:**  Understand HD map layers—lane geometry, road markings, traffic signs, traffic lights, and 3D landmarks. Study OpenDRIVE and Lanelet2 map formats.
    * **Online Map Building:**  Implement online HD map prediction from sensor data (MapTR, BeMapNet) to reduce dependence on pre-built maps and support uncharted areas.
    * **Map Localization:**  Implement map-based localization using point cloud matching (NDT, ICP) or camera-map matching for precise positioning within an HD map.

**Resources:**

* **"BEVFormer" and "BEVFusion" papers:**  Foundational BEV perception architectures for autonomous driving.
* **nuScenes and Waymo Open Dataset:**  Large-scale autonomous driving datasets with multi-modal sensor data, HD maps, and 3D annotations.
* **"A Survey on Motion Prediction and Risk Assessment for Intelligent Vehicles" (survey paper):**  Review of trajectory prediction methods for autonomous driving.

**Projects:**

* **BEV Object Detection:**  Train a BEVFusion-style model on nuScenes for camera+LiDAR 3D object detection. Analyze BEV feature quality and compare with LiDAR-only baseline.
* **Trajectory Prediction Pipeline:**  Implement TNT or MTR trajectory prediction for the Waymo Open Motion Dataset. Evaluate minADE/minFDE metrics and visualize multi-modal predictions.
* **Online Map Prediction:**  Deploy a MapTR-based online map building system in CARLA. Compare online map quality against the ground-truth HD map.


**3. Safety, Standards, and Deployment**

* **Functional Safety Standards:**
    * **ISO 26262 (Road Vehicles Functional Safety):**  Understand the ISO 26262 standard for automotive functional safety—ASIL (Automotive Safety Integrity Level) classification, safety goals, hazard analysis, and risk assessment (HARA).
    * **SOTIF (ISO 21448):**  Study the Safety of the Intended Functionality standard for autonomous driving—addressing risks from sensor limitations, algorithm uncertainty, and unpredictable environments.
    * **Safety Architecture Patterns:**  Implement safety patterns—redundancy (dual-channel monitoring), diverse redundancy (different algorithms/sensors for the same function), and plausibility monitoring.

* **V2X (Vehicle-to-Everything) Communication:**
    * **DSRC and C-V2X:**  Understand DSRC (Dedicated Short-Range Communications) and C-V2X (Cellular V2X, 5G NR-V2X) for vehicle-to-vehicle (V2V), vehicle-to-infrastructure (V2I), and vehicle-to-pedestrian (V2P) communication.
    * **Cooperative Perception:**  Implement cooperative perception where vehicles share sensor data or object detections via V2X to extend the effective sensing range beyond individual vehicle FoV.
    * **V2X Security:**  Study V2X security standards (IEEE 1609.2, ETSI ITS Security) for authenticated, privacy-preserving V2X communication using certificate authorities and pseudonyms.

* **ADAS Validation and Testing:**
    * **Scenario-Based Testing:**  Design scenario databases (OpenSCENARIO, ASAM OSI) for systematic ADAS testing—covering edge cases, SOTIF-relevant scenarios, and ODD (Operational Design Domain) boundaries.
    * **Hardware-in-the-Loop (HIL) Testing:**  Set up HIL simulation environments that inject synthetic sensor data into production ECUs to validate ADAS software under controlled, repeatable conditions.
    * **Shadow Mode Deployment:**  Deploy perception algorithms in shadow mode—running in parallel with production systems without actuating, logging disagreements for offline evaluation and dataset curation.

**Resources:**

* **ISO 26262 Standard (automotive functional safety):**  Foundational safety standard for road vehicle electronics.
* **"Autonomous Vehicles and Functional Safety: A Practitioner's Guide" (various Tier 1 supplier guides):**  Practical guides to applying ISO 26262 and SOTIF to ADAS systems.
* **5GAA (5G Automotive Association) Resources:**  C-V2X standards, use cases, and deployment guidance.

**Projects:**

* **HARA for a Lane-Keeping System:**  Perform a Hazard Analysis and Risk Assessment (HARA) for a lane-keeping assist function. Assign ASIL levels to identified safety goals and propose mitigations.
* **HIL Sensor Injection:**  Build a simple HIL test rig that injects synthetic camera frames into an ADAS perception node and validates detection accuracy across day/night/fog scenarios.
* **Shadow Mode Evaluator:**  Deploy an experimental perception algorithm in shadow mode alongside a baseline. Collect and analyze disagreements to identify algorithmic weaknesses and curate a targeted test set.
