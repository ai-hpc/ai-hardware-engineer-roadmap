# Edge AI

<div class="course-identity auto-course" style="--course-accent: #ca8a04; --course-accent-rgb: 202, 138, 4;" markdown="1">
<div class="course-identity__icon">EA</div>
<div markdown="1">
<p class="course-identity__eyebrow">Deep Dive · AI Workloads</p>
<p class="course-identity__title">Specialized course identity for Edge AI.</p>
<p class="course-identity__meta">Artifact: model or workload study · Measure: accuracy, latency, memory, throughput</p>
</div>
</div>


**Phase 3 — Artificial Intelligence** (standalone topic). Learn **where** and **why** models run on-device; learn **what** they compute in **[Neural Networks](../../1.%20Neural%20Networks/Guide.md)**.

> **Goal:** Map the edge stack—latency, privacy, power tiers, and the train → optimize → deploy pipeline—so Phase 4 (Xilinx or Jetson) and specialization tracks have clear context.

**Previous:** [Phase 1 §4 — C++ and Parallel Computing](../../../Phase%201%20-%20Foundational%20Knowledge/4.%20C%2B%2B%20and%20Parallel%20Computing/Guide.md) · **Companion:** [Neural Networks](../../1.%20Neural%20Networks/Guide.md) (MLPs, CNNs, training, tinygrad) · **Next (deployment depth):** [Phase 4 Track B — Jetson](../../../Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/1.%20Nvidia%20Jetson%20Platform/Guide.md)

---


## 1. Definition

**Edge AI** = running AI algorithms **locally on a device** (the "edge") instead of sending data to a remote cloud server.

```
Traditional Cloud AI:
  Device → [internet] → Cloud Server (GPU farm) → [internet] → Result
  Latency: 50–300ms   Privacy risk   Needs connectivity

Edge AI:
  Device → Local Chip (CPU/GPU/NPU/FPGA) → Result
  Latency: <1ms       Data stays local   Works offline
```

---

## 2. Why edge AI exists

| Problem with Cloud AI        | Edge AI Solution                        |
|------------------------------|-----------------------------------------|
| Network latency (~100ms)     | Sub-millisecond local inference         |
| Bandwidth cost (video data)  | Only send results, not raw data         |
| Privacy (face/voice/medical) | Data never leaves the device            |
| Reliability (no internet)    | Works fully offline                     |
| Cloud cost at scale          | One-time hardware cost                  |

---

## 3. Where edge AI runs (tiers)

```
Tier 1 — Microcontrollers (MCU):
  STM32, Arduino, RP2040
  RAM: 256KB–512KB
  Power: <1W
  Use: keyword spotting, gesture detection

Tier 2 — Embedded Linux SBCs:
  Raspberry Pi, BeagleBone
  RAM: 1–8GB
  Power: 2–10W
  Use: image classification, object detection

Tier 3 — AI Accelerator SoCs:
  Nvidia Jetson, Google Coral (TPU), Apple Neural Engine
  RAM: 4–64GB
  Power: 5–30W
  Use: real-time video inference, NLP, robotics

Tier 4 — Edge Servers:
  FPGA + GPU combinations, industrial PCs
  Power: 50–300W
  Use: factory automation, autonomous vehicles
```

---

## 4. The edge AI pipeline

```
1. Train model on a powerful workstation/cloud (large data, many epochs)
2. Optimize model for edge (quantization, pruning, distillation)
3. Convert model to edge runtime format (ONNX, TensorRT, TFLite)
4. Deploy to edge device
5. Run inference locally in real-time
```

Step 1 is grounded in **[Neural Networks](../../1.%20Neural%20Networks/Guide.md)**. Steps 2–5 are expanded in **Phase 4 Track B** ([ML and AI](../../../Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/5.%20Application%20Development/5.%20ML%20and%20AI/Guide.md)) and, for custom silicon, **Phase 4 Track A** and **Phase 5 — AI Chip Design**.

---

## 5. How this fits the roadmap

| You want… | Start here | Then |
|-----------|------------|------|
| Intuition for tensors, backprop, CNNs | [Neural Networks](../../1.%20Neural%20Networks/Guide.md) | tinygrad hands-on, [micrograd](../../2.%20Deep%20Learning%20Frameworks/micrograd/Guide.md) |
| Product and deployment context (this guide) | Skim tiers + pipeline above | Phase 4 Jetson or FPGA track |
| Vision preprocessing and classical CV | [Computer Vision](../3.%20Computer%20Vision/Guide.md) | Phase 4 perception pipelines |
| Multi-sensor calibration, tracking, BEV fusion | [Sensor Fusion](../4.%20Sensor%20Fusion/Guide.md) | Phase 4 Jetson + [ROS2](../../../Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/5.%20Application%20Development/6.%20ROS2/Guide.md) for integration |

**Hub:** [Phase 3 — Artificial Intelligence](../../Guide.md)
