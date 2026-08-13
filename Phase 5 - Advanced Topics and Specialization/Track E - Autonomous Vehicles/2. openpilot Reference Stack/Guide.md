# Module 2 — openpilot Reference Stack

<div class="course-identity auto-course" style="--course-accent: #0891b2; --course-accent-rgb: 8, 145, 178;" markdown="1">
<div class="course-identity__icon">MORS</div>
<div markdown="1">
<p class="course-identity__eyebrow">Deep Dive · Specialization</p>
<p class="course-identity__title">Specialized course identity for Module 2 — openpilot Reference Stack.</p>
<p class="course-identity__meta">Artifact: specialization case study · Measure: performance, reliability, role fit</p>
</div>
</div>


**Parent:** [Phase 5 — Autonomous Driving](../Guide.md)

**Time:** 3–4 months

**Prerequisites:** Module 1 (Fundamentals), Phase 4 Track B (Jetson — CUDA, Linux BSP, device trees).

---

## Why openpilot

[openpilot](https://github.com/commaai/openpilot) is comma.ai's open-source ADAS — production-deployed, well-documented, and using tinygrad for inference. It's the best publicly available end-to-end reference for studying how an autonomous driving system actually works: from raw camera pixels to CAN bus steering commands.

---

## 1. Architecture Overview

* **End-to-end pipeline:**
    * `camerad` → `modeld` → `plannerd` → `controlsd` → `pandad` → CAN bus → vehicle actuators.
    * Each process communicates via **cereal** (msgpack-based IPC) and **VisionIpc** (shared-memory zero-copy frames).

* **Hardware:**
    * comma 3X / comma four: Snapdragon 845/8 Gen 2, 3 cameras (wide road, road, driver), GPS, IMU.
    * **panda**: CAN-to-USB interface for vehicle communication.

* **Software layers:**
    * **AGNOS**: Custom Linux distro (fork of upstream kernel) for comma devices.
    * **openpilot**: Python/C++ application layer — perception, planning, control.
    * **tinygrad**: Inference runtime for neural network models.

* **[Flow Diagram](flow-diagram.md)** — Detailed data flow from camera input to CAN actuation with process-to-source mapping.

**Projects:**
* Clone openpilot. Trace the data flow from `camerad` to `pandad` by reading `selfdrive/` source. Document the IPC messages between each process.
* Run openpilot in simulation using comma's replay tools or CARLA integration.

---

## 2. Camera Pipeline (camerad)

> **Deep dive:** [camerad Guide](camerad/Guide.md)

* **Sensor hardware:** OX03C10 / OS04C10 image sensors, Qualcomm Spectra ISP.
* **Capture flow:** Sensor RAW → CSI → IFE (demosaic, CCM, gamma) → BPS → YUV NV12.
* **Auto exposure (AE):** Software algorithm with 3-frame latency, PI-like control loop, DC gain hysteresis for night driving.
* **VisionIpc:** Shared-memory IPC for zero-copy frame transport from `camerad` to `modeld`.
* **V4L2 integration:** Linux Video4Linux2 API for camera device control, request manager, buffer flow.

**Projects:**
* Read `system/camerad/` source. Trace a single frame from sensor DQBUF to VisionIpc publish.
* Modify the AE target grey fraction. Observe the effect on exposure in different lighting conditions.

---

## 3. AGNOS Operating System

> **Deep dive:** [AGNOS + OS Course](agnos/Guide.md)

* **What AGNOS is:** Fork of Linux kernel (`agnos-kernel-sdm845`) + custom build system (`agnos-builder`) for comma devices.
* **Key kernel customizations:** Camera drivers (V4L2/media), device tree for SDM845, SPI (CAN-over-SPI for panda), thermal/power management, cgroups for process isolation.
* **Maps to Phase 1 OS lectures:** All 26 OS lectures from Phase 1 are mapped to specific kernel paths and openpilot use cases in the AGNOS guide.

**Projects:**
* Clone `agnos-kernel-sdm845`. Follow the AGNOS guide to trace how Phase 1 OS concepts (processes, interrupts, scheduling, device tree) manifest in a production ADAS kernel.
* Identify the device tree nodes for the 3 camera sensors. Understand how CSI lanes and clocks are configured.

---

## 4. Perception (modeld)

* **Model architecture:** End-to-end neural network taking multi-camera input, outputting lanes, lead vehicles, pose, plan, and desired path.
* **Warp matrix:** Calibration-based image warping applied before inference to normalize camera viewpoint.
* **tinygrad inference:** Models run through tinygrad on Snapdragon GPU (Adreno) — see [Module 3](../3.%20tinygrad%20for%20Inference/Guide.md).
* **Outputs:** `modelV2` cereal message — lanes, road edges, pose, plan trajectory, action (gas/brake/steer), forward collision warning (FCW).

**Projects:**
* Read `selfdrive/modeld/`. Trace model input preparation (frame + warp) through tinygrad inference to modelV2 output.
* Log modelV2 outputs during a drive replay. Visualize predicted lanes and lead vehicle positions.

---

## 5. Planning and Control

* **plannerd:**
    * `LongitudinalPlanner`: speed profile, following distance, stop-and-go.
    * `LaneDepartureWarning`: lateral safety monitoring.
    * Inputs: modelV2 (perception), radarState, carState.

* **controlsd:**
    * `LatControl`: lateral PID/INDI/torque controller for steering.
    * `LongControl`: longitudinal PID for gas/brake.
    * Outputs: `carControl` cereal message → `CarInterface` → CAN commands.

* **pandad + CAN:**
    * `pandad` translates carControl into raw CAN messages via the panda device.
    * Vehicle-specific `CarInterface` implementations handle DBC encoding per make/model.

**Projects:**
* Trace a steering command from plannerd's desired path through controlsd's lateral controller to the final CAN message. Document the control loop.
* Compare openpilot's lateral control (torque-based) with the Stanley controller from Module 1.

---

## 6. Forking and Contributing

* **Fork workflow:** Fork openpilot, set up development environment, build and test.
* **Vehicle porting:** Add support for a new vehicle — CAN DBC reverse engineering, `CarInterface` implementation, fingerprinting.
* **Community:** Active Discord, community forks, bounty programs for contributions.

**Projects:**
* Fork openpilot. Make a small change (e.g., adjust a UI element or tuning parameter). Build, test in simulation, and submit a PR.
* (Advanced) Study the CAN DBC for your vehicle. Implement basic parsing and actuation.

---

## Resources

| Resource | URL |
|----------|-----|
| openpilot source | https://github.com/commaai/openpilot |
| agnos-kernel-sdm845 | https://github.com/commaai/agnos-kernel-sdm845 |
| agnos-builder | https://github.com/commaai/agnos-builder |
| comma.ai Blog | https://blog.comma.ai/ |
| openpilot community docs | https://github.com/commaai/openpilot/wiki |

---

## Next

→ **[Module 3 — tinygrad for Inference](../3.%20tinygrad%20for%20Inference/Guide.md)** — Deep dive into the inference engine that powers openpilot's perception.
