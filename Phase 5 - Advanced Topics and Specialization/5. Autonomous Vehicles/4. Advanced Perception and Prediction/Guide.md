# Module 4 — Advanced Perception and Prediction

<div class="course-identity auto-course" style="--course-accent: #0d9488; --course-accent-rgb: 13, 148, 136;" markdown="1">
<div class="course-identity__icon">MAPA</div>
<div markdown="1">
<p class="course-identity__eyebrow">Deep Dive · Specialization</p>
<p class="course-identity__title">Specialized course identity for Module 4 — Advanced Perception and Prediction.</p>
<p class="course-identity__meta">Artifact: specialization case study · Measure: performance, reliability, role fit</p>
</div>
</div>


**Parent:** [Phase 5 — Autonomous Driving](../Guide.md)

**Time:** 6–12 months

**Prerequisites:** Modules 1–3, Phase 3 (Sensor Fusion — Kalman filtering, multi-sensor math).

---

## Why this is advanced

Modules 1–3 teach you to understand and work with a single-camera, production-deployed ADAS (openpilot). This module goes beyond: multi-sensor perception, BEV architectures, trajectory prediction, HD maps, and simulation — the research and engineering frontier for L3+ autonomy.

---

## 1. Advanced Sensor Hardware

* **LiDAR:**
    * Working principles: mechanical spinning (Velodyne), solid-state (Livox, Innoviz), FMCW (Aeva, Luminar).
    * Trade-offs: range, resolution, FoV, latency, cost.
    * Point cloud representations: raw points, voxels, range images, pillars.

* **Radar:**
    * Automotive radar: 77 GHz FMCW, range-Doppler processing, angular resolution.
    * 4D imaging radar: elevation + azimuth + range + velocity.
    * Radar advantages: all-weather, direct velocity measurement, long range.

* **Camera systems:**
    * Sensor types: CCD vs CMOS, global vs rolling shutter.
    * Optics: FoV, focal length, aperture, HDR techniques.
    * ISP pipelines for automotive (connection to Module 2 camerad).

---

## 2. Multi-Sensor Calibration

* **Intrinsic calibration:**
    * Camera intrinsics (focal length, principal point, distortion) using checkerboard or ArUco patterns with OpenCV or Kalibr.

* **Extrinsic calibration:**
    * Spatial transforms between sensors: camera-LiDAR, camera-radar, LiDAR-IMU.
    * Target-based (checkerboard, ArUco) and targetless methods (LI-Calib, ACSC).

* **Temporal calibration:**
    * Aligning sensor timestamps across hardware clocks.
    * PPS signals, IEEE 1588 PTP, cross-correlation of observed events.
    * Impact of time misalignment on fusion accuracy.

**Projects:**
* Build a calibration target and calibrate a camera-LiDAR pair using Kalibr or ACSC. Verify by projecting LiDAR points onto camera image.
* Implement a radar-camera fusion pipeline: associate radar detections with camera bounding boxes for velocity-augmented detection.

---

## 3. BEV (Bird's-Eye View) Perception

* **BEV transformers:**
    * BEVFormer: attention-based cross-view feature lifting from perspective to BEV.
    * BEVFusion: multi-modal (camera + LiDAR) fusion in BEV space.
    * Tesla's Occupancy Network: dense 3D voxel prediction from cameras.

* **3D occupancy prediction:**
    * Voxel-based occupancy (Occ3D, SurroundOcc) as alternative to explicit object detection.
    * Represent environment as dense 3D grid of occupied/free voxels.

* **Temporal fusion:**
    * Incorporate temporal information using recurrent architectures or long-range temporal attention.
    * Track implicit object states across frames without explicit tracking.

**Projects:**
* Train a BEVFusion-style model on nuScenes for camera+LiDAR 3D object detection. Compare with LiDAR-only baseline.

---

## 4. Trajectory Prediction

* **Multi-modal prediction:**
    * TNT, MTR, Wayformer: predict multiple plausible future trajectories with probability scores.
    * Why multi-modal: a vehicle at an intersection might go straight, turn left, or turn right.

* **Interaction modeling:**
    * Social force models, graph neural networks (GRIP, HYPER).
    * Transformer-based interaction encoders for agent-agent reasoning.

* **Map-conditioned prediction:**
    * Incorporate lane geometry and traffic rules.
    * VectorNet, MapTR-style map encoding.
    * Constraining predictions to physically plausible, lane-following trajectories.

**Projects:**
* Implement TNT or MTR on the Waymo Open Motion Dataset. Evaluate minADE/minFDE. Visualize multi-modal predictions.

---

## 5. HD Maps and Online Mapping

* **HD map components:**
    * Layers: lane geometry, road markings, traffic signs, traffic lights, 3D landmarks.
    * Formats: OpenDRIVE, Lanelet2.

* **Online map building:**
    * MapTR, BeMapNet: predict HD map from sensor data in real-time.
    * Reduces dependence on pre-built maps, supports uncharted areas.

* **Map-based localization:**
    * Point cloud matching: NDT, ICP.
    * Camera-map matching for precise positioning within HD map.

**Projects:**
* Deploy a MapTR-based online map system in CARLA. Compare online map quality against ground-truth HD map.

---

## 6. Sensor Simulation and Synthetic Data

* **Synthetic data generation:**
    * CARLA, SUMO+CARLA, NVIDIA DRIVE Sim for photorealistic training data with ground-truth labels.
    * Labels: bounding boxes, semantic masks, depth, radar detections.

* **Sensor model simulation:**
    * LiDAR ray casting, radar multipath, camera noise models.
    * Reducing sim-to-real gap in training data.

* **Scenario generation:**
    * Corner cases: fog, rain, night, occlusions, near-miss events.
    * Rare or unsafe scenarios that cannot be collected in real driving.

**Projects:**
* Generate a 10,000-frame synthetic dataset in CARLA with varying weather, lighting, and traffic density. Train a 3D detector on synthetic vs. real data and compare.

---

## Resources

| Resource | Why |
|----------|-----|
| [nuScenes](https://www.nuscenes.org/) | Multi-modal AV dataset with 3D annotations and HD maps |
| [Waymo Open Dataset](https://waymo.com/open/) | Large-scale dataset with motion prediction benchmarks |
| BEVFormer / BEVFusion papers | Foundational BEV perception architectures |
| [Kalibr](https://github.com/ethz-asl/kalibr) | Multi-sensor calibration toolkit |
| [CARLA](https://carla.org/) | Sensor simulation and synthetic data |

---

## Next

→ **[Module 5 — Safety Standards and Deployment](../5.%20Safety%20Standards%20and%20Deployment/Guide.md)** — ISO 26262, SOTIF, V2X, HIL testing, and production deployment.
