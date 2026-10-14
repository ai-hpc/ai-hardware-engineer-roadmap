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

[openpilot](https://github.com/commaai/openpilot) is comma.ai's open-source ADAS — production-deployed, well-documented, and using tinygrad for inference. It is a useful inspectable reference for a driver-assistance system: from raw camera pixels to CAN bus steering commands.

---

## 1. Architecture Overview

* **End-to-end pipeline:**
    * Conceptually: camera capture → model inference → planning/control → vehicle-specific CAN encoding → panda transport/safety → vehicle. Process ownership changes by revision; trace the actual publishers and subscribers.
    * Each process communicates via **cereal** (Cap’n Proto message schemas, carried over the messaging transport) and **VisionIpc** (shared-memory zero-copy frames).

* **Hardware:**
    * Use one named comma device revision and its published hardware specification. Do not infer that comma 3X and comma four share the same SoC, sensor, or kernel configuration.
    * **panda**: CAN-to-USB interface for vehicle communication.

* **Software layers:**
    * **AGNOS**: Custom Linux distro (fork of upstream kernel) for comma devices.
    * **openpilot**: Python/C++ application layer — perception, planning, control.
    * **tinygrad**: Inference runtime for neural network models.

* **[Flow Diagram](flow-diagram.md)** — Detailed data flow from camera input to CAN actuation with process-to-source mapping.

**Projects:**
* Clone openpilot. Trace the data flow from `camerad` to `pandad` by reading `selfdrive/` source. Document the IPC messages between each process.
* Run a recorded log replay first. Replay consumes fixed observations; closed-loop simulation requires the selected simulator bridge and feeds commands back into the world.

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
* On a bench camera or recorded test setup, vary the AE target and compare exposure convergence and saturation; do not modify a driving device for this exercise.

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
* **Outputs:** `modelV2` contains model predictions defined by the pinned schema; vehicle commands are produced through the downstream control/vehicle interface. Do not equate a predicted trajectory with final gas/brake/steer actuation.

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
    * Vehicle-specific controller/interface code encodes CAN messages; `pandad` handles communication with the panda device. Trace the `sendcan` publisher at the selected revision.
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
* (Advanced) Study the CAN DBC for your vehicle. Implement parsing and compare generated commands on an isolated simulated/bench bus, without connection to vehicle actuators.

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

## Source-tracing lab: prove who owns each boundary

Use a fixed repository tag/commit. The [v0.9.7 `card.py` snapshot](https://github.com/commaai/openpilot/blob/v0.9.7/selfdrive/car/card.py) and [message schemas](https://github.com/commaai/openpilot/blob/v0.9.7/cereal/log.capnp) are historical references for the distinction between model output, vehicle control, and CAN transport; they are not a claim that current paths are identical.

For one replay frame, make a table of message name, publisher, subscriber, timestamp, frame ID, coordinate frame, and stale-data policy. Follow one desired lateral command through the vehicle interface to serialized CAN bytes, then locate the independent safety checks. Carry the input preprocessing and output contract into the tinygrad module so a compiler optimization cannot silently change units or frames.

openpilot is driver assistance with an attentive driver requirement. Keep these coursework changes in replay/simulation/isolated benches; passing those exercises is not a road-deployment approval. [Official safety model](https://docs.comma.ai/concepts/safety/).

## Practice and assessment

**Entry diagnostic (prerequisite check):** Trace camera capture through model, plan, and control processes. If this is unfamiliar, use the prerequisite material above or the [roadmap diagnostics](../../../Learning-Guide.md#diagnose-before-reading) before starting the lab.

1. **Baseline lab:** Inspect a pinned openpilot path and reproduce a recorded-data or simulator experiment.
2. **Guided variation:** Inject a replay timing or missing-message fault and identify the responsible process boundary.
3. **Independent transfer:** repeat the method on a new input or configuration and explain where your original assumptions stop applying.

**Evidence to submit:** Message age, model/plan outputs, latency, process logs, and scenario outcome; use replay/simulation. Keep raw results and reproduction steps; distinguish measured results from estimates or work not run.

**Self-check:** Which observation would disprove your explanation of the result? Identify a competing explanation and the smallest experiment that separates them. A passing answer names a concrete predicted difference, not only another tool to try.

**Exit criteria:** reproduce the baseline, demonstrate the negative/variation check, and meet the [shared completion rubric](../../../Learning-Guide.md#completion-rubric). Use the [lesson cycle](../../../Learning-Guide.md#a-repeatable-lesson-cycle) for retrieval practice and spaced review.
