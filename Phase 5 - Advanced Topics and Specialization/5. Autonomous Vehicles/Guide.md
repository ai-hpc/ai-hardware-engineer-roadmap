# Phase 5 — Autonomous Vehicles

<div class="course-identity autonomous-vehicles" markdown="1">
<div class="course-identity__icon">AV</div>
<div markdown="1">
<p class="course-identity__eyebrow">Track 5E · Autonomous Vehicles</p>
<p class="course-identity__title">Study production autonomy stacks, perception, prediction, safety, deployment, and vehicle debugging.</p>
<p class="course-identity__meta">Artifact: AV stack analysis · Measure: sensor timing, model latency, safety constraints</p>
</div>
</div>


**Timeline:** 12–24 months (modules 1–3); 24–48 months with advanced modules (4–5). Module 6 is optional tooling.

**Prerequisites:** Phase 3 (**Computer Vision**, **Sensor Fusion**), Phase 4 Track B — Jetson (CUDA, TensorRT, edge deployment). Phase 4 Track C recommended for tinygrad compiler context.

**Role targets:** ADAS Software Engineer · Autonomous Driving Perception Engineer · Motion Planning Engineer · Functional Safety Engineer · AV Systems Engineer

---

## Overview

This track uses **[openpilot](https://github.com/commaai/openpilot)** (comma.ai) as the primary reference implementation — an open-source, production-deployed ADAS that runs tinygrad for on-device inference. You study a real system end-to-end: camera capture, neural network perception, planning, control, and CAN actuation.

The track progresses from fundamentals through the openpilot codebase, then into advanced perception research and safety/deployment standards.

---

## Module Map

| # | Module | What you learn | Time |
|---|--------|---------------|------|
| **1** | [Fundamentals](1.%20Fundamentals/Guide.md) | CV for driving, planning algorithms, control theory, vehicle dynamics | 3–4 months |
| **2** | [openpilot Reference Stack](2.%20openpilot%20Reference%20Stack/Guide.md) | Full openpilot architecture: camera pipeline, AGNOS kernel, data flow, forking | 3–4 months |
| **3** | [tinygrad for Inference](3.%20tinygrad%20for%20Inference/Guide.md) | tinygrad internals: lazy eval, 3 op types, compiler pipeline, backends, custom ops | 3–4 months |
| **4** | [Advanced Perception and Prediction](4.%20Advanced%20Perception%20and%20Prediction/Guide.md) | Sensors (LiDAR, radar), calibration, BEV perception, trajectory prediction, HD maps, simulation | 6–12 months |
| **5** | [Safety Standards and Deployment](5.%20Safety%20Standards%20and%20Deployment/Guide.md) | ISO 26262, SOTIF, V2X, HIL testing, shadow mode, scenario-based validation | 3–6 months |
| **6** | [Lauterbach TRACE32 Debug](6.%20Lauterbach%20TRACE32%20Debug/Guide.md) | In-circuit debug and trace for automotive ECUs (optional professional tooling) | 2–3 months |

---

## Recommended Order

```
Module 1 (Fundamentals)
    ↓
Module 2 (openpilot) ←→ Module 3 (tinygrad)   [study in parallel or interleaved]
    ↓
Module 4 (Advanced Perception)
    ↓
Module 5 (Safety & Deployment)
    ↓
Module 6 (Lauterbach — optional, for automotive ECU roles)
```

Modules 2 and 3 reinforce each other: openpilot is the system context, tinygrad is the inference engine inside it. Study them together.

---

## Reference Projects Used Throughout

| Project | Module 1 | Module 2 | Module 3 | Module 4 |
|---------|----------|----------|----------|----------|
| **[openpilot](https://github.com/commaai/openpilot)** | Lane/object detection context | Full stack: camera→ISP→modeld→planning→CAN | Inference engine inside modeld | Real perception workloads |
| **[tinygrad](https://github.com/tinygrad/tinygrad)** | — | Runtime for openpilot models | Full compiler + backend study | Custom backend for accelerators |
| **[CARLA](https://carla.org/)** | Simulation for planning algorithms | Test openpilot in simulation | — | Synthetic data, sensor models |

---

## Key Resources

| Resource | URL |
|----------|-----|
| openpilot | https://github.com/commaai/openpilot |
| tinygrad | https://github.com/tinygrad/tinygrad |
| CARLA Simulator | https://carla.org/ |
| comma.ai Blog | https://blog.comma.ai/ |
| nuScenes Dataset | https://www.nuscenes.org/ |
| Waymo Open Dataset | https://waymo.com/open/ |
