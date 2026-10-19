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


## 1. Define the deployment boundary

Edge inference runs some or all model computation near the sensor or user. Local execution removes a network round trip from that path; it does not guarantee sub-millisecond latency, privacy, or offline operation. Check which preprocessing, model files, authorization, logging, and updates still depend on a service.

| Constraint | What local inference can change | What to measure |
|---|---|---|
| Response time | Avoid remote queueing and transport | Observation-to-action latency, including capture and queueing |
| Data handling | Keep selected inputs on-device | Actual logs, telemetry, and upload paths |
| Connectivity | Continue with local assets and fallback | Link-loss behavior and recovery |
| Cost | Trade hosted inference for device resources | Hardware, energy, maintenance, and fleet updates over one interval |
| Power and heat | Fit computation into a local envelope | Sustained clock, temperature, energy per accepted result |

## 2. Choose a target from the workload

MCUs suit small fixed workloads; embedded Linux boards support richer I/O and software; accelerator SoCs add specialized compute; edge servers provide more capacity. These categories overlap. Read the **exact board/SKU** memory, supported operators, I/O, and power documentation instead of assigning one RAM or wattage range to an entire class.

Start with a tensor and state inventory: weights, live activations, cache/recurrent state, workspace, camera buffers, and the rest of the system. A model fitting on disk does not establish that its peak runtime allocation fits DRAM.

## 3. Work a deadline backward

Suppose a simulated manipulation task allows **50 ms** from observation capture to accepted action. This is a teaching target, not a universal robot deadline. Allocate 8 ms to capture/preprocessing, 30 ms to policy inference, 5 ms to action validation/dispatch, and 7 ms to reserve. The controller may tick at a different rate and consume an action chunk between policy updates.

Measure the whole path using trace IDs and a common clock. Stage p99 values cannot generally be added to obtain the end-to-end p99. Record distributions, deadline misses, observation age, and sustained thermal behavior. If inference takes 20 ms but the queued image is already 80 ms old, the action fails this contract.

## 4. The deployment experiment

**CPU practice before hardware:** complete the [PyTorch robot-stop lab](../../2.%20Deep%20Learning%20Frameworks/PyTorch/CPU-Robot-Stop-Lab/Guide.md). It trains a real network, selects the checkpoint/threshold using validation data, evaluates held-out episodes and a distance-sensor shift, then saves/reloads/exports and checks output parity. Its `.pt2` artifact is a PyTorch exported program, not a TensorRT engine or a guarantee of support on an embedded target.

1. Preserve a reference model and held-out task/data split.
2. Measure the reference on the intended target before optimizing.
3. Change one supported component: precision, model size, layout, or scheduling.
4. Export an exchange graph such as ONNX where supported, then build the target runtime artifact. ONNX is a representation; TensorRT is a compiler/runtime, not an interchangeable file format.
5. Compare quality, timing, memory, and recovery with the reference. Local inference qualifies for a stated real-time requirement only after deadline behavior is established.

Carry the report to the Jetson or compiler courses. Use the same recorded inputs and preprocessing manifest so a runtime change remains a controlled experiment.

---

## 5. Worked memory and energy checks

One million FP32 weights require 4,000,000 payload bytes (about 3.81 MiB). INT8 payload is one quarter as large, but scales, unquantized layers, activations, workspace, and buffering remain. If weights occupy 4 MiB and everything else peaks at 20 MiB, reducing weights to 1 MiB changes the total from 24 to 21 MiB: 12.5% less, not 75% less. Measure simultaneous live allocations rather than adding every tensor ever created.

Energy per accepted result is measured joules divided by accepted results over the same interval. A 10 W device taking 20 ms uses 0.2 J during that interval; this simple arithmetic omits idle/transition energy unless the measurement includes it. A low-power device can use more energy per task if it takes much longer.

**Pass:** preserve the same held-out inputs/quality gate, produce one actual baseline measurement, change one supported optimization, and explain quality, memory, timing, and energy separately. State device SKU, software/engine versions, warmup, concurrency, shape, power mode, and measurement boundary. For Jetson use its supported TensorRT release: current TensorRT 11.x desktop/server documentation does not establish JetPack compatibility.

## 6. How this fits the roadmap

| You want… | Start here | Then |
|-----------|------------|------|
| Intuition for tensors, backprop, CNNs | [Neural Networks](../../1.%20Neural%20Networks/Guide.md) | tinygrad hands-on, [micrograd](../../2.%20Deep%20Learning%20Frameworks/micrograd/Guide.md) |
| Product and deployment context (this guide) | Skim tiers + pipeline above | Phase 4 Jetson or FPGA track |
| Vision preprocessing and classical CV | [Computer Vision](../3.%20Computer%20Vision/Guide.md) | Phase 4 perception pipelines |
| Multi-sensor calibration, tracking, BEV fusion | [Sensor Fusion](../4.%20Sensor%20Fusion/Guide.md) | Phase 4 Jetson + [ROS2](../../../Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/5.%20Application%20Development/6.%20ROS2/Guide.md) for integration |

**Hub:** [Phase 3 — Artificial Intelligence](../../Guide.md)


## Carry this course into the physical-AI system

**Start from:** The policy reference, memory inventory, and a task deadline.

**Build and challenge:** Optimize one supported component and repeat under sustained load. Include preprocessing, queueing, and action dispatch; a fast kernel alone does not establish a fresh action.

**Pass forward:** A quality/timing/memory report ready for target deployment. Continue with [the connected course](../../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20D%20-%20Robotics/VLA%20Optimization%20and%20Action-Parity%20Harness/README.md) and keep the artifact in the [Physical AI Design Studio](../../../Physical-AI-Design-Studio.md).
