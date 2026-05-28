# Application Development

<div class="course-identity jetson-apps" markdown="1">
<div class="course-identity__icon">APP</div>
<div markdown="1">
<p class="course-identity__eyebrow">Track B5 · Jetson Application Development</p>
<p class="course-identity__title">Build applications across peripherals, networking, UI, multimedia, ML inference, and ROS 2.</p>
<p class="course-identity__meta">Artifact: integrated Jetson app · Measure: latency, resources, UX, reliability</p>
</div>
</div>


**Phase 4 — Track B — Nvidia Jetson** · Module 5 of 7

> **Focus:** Build production application software on the **Jetson Orin Nano 8GB** — from low-level peripheral access (GPIO, UART, SPI, I2C, CAN) through networking, GUI, multimedia pipelines, **ML/AI inference** (TensorRT, DLA, tinygrad), and **ROS 2** integration.
>
> **Primary hardware:** Jetson Orin Nano 8GB on custom or dev-kit carrier

**Previous:** [4. FSP Customization](../4.%20FSP%20%28Firmware%20Support%20Package%29%20Customization/Guide.md) · **Next:** [6. Security and OTA](../6.%20Security%20and%20OTA/Guide.md)

---

## Sub-modules

| # | Sub-module | Focus |
|---|-----------|-------|
| 1 | [**Peripheral Access**](1.%20Peripheral%20Access/Guide.md) | GPIO, PWM, UART, SPI, I2C, CAN, USB, NVMe/SD, backlight — Linux userspace and kernel driver interfaces |
| 2 | [**Network and Connectivity**](2.%20Network%20and%20Connectivity/Guide.md) | Ethernet, Wi-Fi, Bluetooth, VPN, web server — wired and wireless connectivity stack |
| 3 | [**GUI**](3.%20GUI/Guide.md) | Qt, LVGL, web-based UI, framebuffer — display and touch for embedded products |
| 4 | [**Multimedia**](4.%20Multimedia/Guide.md) | Audio, cameras (USB + CSI), GStreamer, display output, video encode/decode — hardware-accelerated media pipelines |
| 5 | [**ML and AI**](5.%20ML%20and%20AI/Guide.md) | Quantization, TensorRT, DLA, tinygrad, DeepStream, profiling — edge inference optimization |
| 6 | [**ROS 2**](6.%20ROS2/Guide.md) | Nodes, Nav2, DDS, real-time, Jetson deployment — robot software framework |

---

## How the sub-modules fit the product flow

```
Custom carrier board (Module 2) + L4T BSP (Module 3) + FSP (Module 4)
  │
  ├─ Peripheral Access ─── talk to sensors, actuators, buses
  ├─ Network ──────────── connect to cloud, fleet, local network
  ├─ GUI ──────────────── user-facing display (if applicable)
  ├─ Multimedia ────────── cameras, audio, video pipelines
  ├─ ML / AI ──────────── inference optimization on Orin Nano
  └─ ROS 2 ────────────── robot middleware tying it all together
        │
        ▼
  Security & OTA (Module 6) → Compliance (Module 7)
```

Work through sub-modules 1–4 in any order based on your product's needs. **ML/AI** (sub-module 5) and **ROS 2** (sub-module 6) typically build on the peripheral and multimedia foundations.
