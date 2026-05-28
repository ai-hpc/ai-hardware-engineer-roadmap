# 3. Edge AI

> Design, optimize, deploy, and operate AI systems on constrained hardware where latency, memory, power, thermals, sensors, and reliability matter.

**Layer mapping:** L1-L5. This track connects edge workloads, model optimization, inference runtimes, embedded Linux, sensor pipelines, accelerators, and product deployment.

**Role targets:** Edge AI Engineer · Embedded AI Engineer · Jetson Engineer · TinyML Engineer · Robotics Perception Engineer · Edge Inference Optimization Engineer

**Prerequisites:** [Phase 2 — Embedded Systems](../../Phase%202%20-%20Embedded%20Systems/Guide.md), [Phase 3 — AI Workloads](../../Phase%203%20-%20Artificial%20Intelligence/Guide.md), and preferably [Phase 4 Track B — NVIDIA Jetson](../../Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/1.%20Nvidia%20Jetson%20Platform/Guide.md).

**What comes after:** [ML Systems Engineering](../7.%20ML%20Systems%20Engineering/Guide.md), [Robotics](../4.%20Robotics/Guide.md), [Autonomous Vehicles](../5.%20Autonomous%20Vehicles/Guide.md), or [AI Chip Design](../6.%20AI%20Chip%20Design/Guide.md).

---

## Why This Track Exists

Edge AI is not "cloud inference on a smaller box." The system has different constraints:

- limited memory
- limited power
- thermal throttling
- sensor timing
- camera/audio preprocessing
- local reliability requirements
- intermittent network access
- update and rollback risk
- hardware-specific runtimes and accelerators

The core engineering question is:

```text
Can this model meet the product requirement on this device under real power,
thermal, memory, sensor, and latency constraints?
```

This track teaches you to answer that with measurements.

---

## Course Outcomes

By the end, you should be able to:

- choose an edge model architecture for a target device
- compress and quantize models without guessing
- deploy through TensorRT, ONNX Runtime, LiteRT/TFLite, TFLM, or a vendor runtime
- profile latency, memory, throughput, power, and thermals
- build camera, audio, and sensor-to-model pipelines
- reason about CPU/GPU/DLA/NPU/DSP partitioning
- design OTA, rollback, telemetry, and fleet update flows
- explain when edge inference should stay local and when it should offload

---

## Course Map

| Unit | Focus | Artifact |
|------|-------|----------|
| 1 | Edge constraints and platform selection | target-device decision memo |
| 2 | Efficient model architectures | model comparison benchmark |
| 3 | Compression and quantization | PTQ/QAT/precision tradeoff report |
| 4 | Runtime deployment | reproducible runtime deployment |
| 5 | TinyML and MCU inference | constrained-memory inference demo |
| 6 | Sensor pipelines | camera/audio/sensor pipeline benchmark |
| 7 | Power, thermal, and reliability | long-run stability report |
| 8 | Fleet and product operation | OTA/telemetry/rollback design |

---

## Unit 1: Edge Constraints And Platform Selection

### Learn

- MCU versus MPU versus edge GPU versus dedicated accelerator
- latency, throughput, memory, power, thermals, cost, and enclosure constraints
- TOPS versus useful throughput
- model size versus activation/KV-cache/runtime memory
- sensor bandwidth and preprocessing cost
- local-only versus edge/cloud hybrid deployment

### Build It

Pick one target product:

- wake-word speaker
- smart camera
- robot perception module
- industrial anomaly detector
- local LLM appliance
- battery-powered wildlife camera

Create a target-device decision memo comparing at least three platforms:

- Jetson Orin Nano/NX/AGX
- Raspberry Pi + accelerator
- Coral Edge TPU
- Hailo
- Qualcomm/Android device
- STM32/nRF/ESP32-class MCU

### Measure It

- memory budget
- latency target
- power budget
- sensor bandwidth
- model size
- expected update cadence

### Ship It

A platform-selection memo that explains why one device fits the workload better than the alternatives.

---

## Unit 2: Efficient Model Architectures

### Learn

- MobileNet, EfficientNet, EfficientDet, EfficientViT, MobileViT, FastViT
- YOLO family and real-time detection tradeoffs
- lightweight transformers
- keyword spotting models
- tiny segmentation and pose models
- distilled LLMs and adapter-based local models
- hardware-aware neural architecture search

### Build It

Benchmark three model families on the same task and device:

- detector: YOLO variant versus EfficientDet-style model
- classifier: MobileNetV3 versus EfficientNet-Lite versus FastViT
- audio: DS-CNN versus small transformer or conformer
- local LLM: quantized small model variants

### Measure It

- accuracy or task metric
- latency
- throughput
- peak memory
- power
- model size
- preprocessing and postprocessing cost

### Ship It

A model-selection report with an accuracy/latency/power table and a clear recommendation.

---

## Unit 3: Compression And Quantization

### Learn

- FP16, BF16, INT8, INT4, mixed precision
- post-training quantization
- quantization-aware training
- calibration datasets
- per-tensor versus per-channel quantization
- pruning and structured sparsity
- distillation
- LLM quantization methods such as AWQ, GPTQ, SmoothQuant, and GGUF quant families
- accuracy recovery and regression testing

### Build It

Take one model through at least two compression paths:

1. baseline precision
2. PTQ
3. QAT or calibrated INT8
4. optional INT4 or LLM quantization path

### Measure It

- task accuracy before/after
- latency before/after
- memory before/after
- power before/after
- layer-level fallback to higher precision

### Ship It

A quantization report that explains what changed, what broke, and which precision you would ship.

---

## Unit 4: Runtime Deployment

### Learn

- TensorRT engine building and calibration
- ONNX export and graph cleanup
- ONNX Runtime execution providers
- LiteRT/TFLite delegates
- TFLite Micro arena allocation
- TensorRT DLA offload on Jetson
- `trtexec`, Nsight Systems, and runtime profiling
- deployment packaging and versioning

### Build It

Deploy the same model through two runtimes where possible:

- PyTorch eager baseline
- ONNX Runtime
- TensorRT
- LiteRT/TFLite
- TFLite Micro
- vendor accelerator runtime

### Measure It

- cold start
- warm latency
- throughput
- peak memory
- model load time
- engine build time
- CPU/GPU/DLA utilization

### Ship It

A reproducible runtime deployment with exact conversion commands, runtime commands, and benchmark output.

---

## Unit 5: TinyML And MCU Inference

### Learn

- Cortex-M class constraints
- SRAM and flash budgeting
- TFLite Micro arena allocation
- CMSIS-NN and DSP kernels
- fixed-point arithmetic
- duty cycling and always-on sensing
- MCU OTA model updates
- drift and local adaptation limits

### Build It

Build one MCU-scale inference demo:

- keyword spotting
- gesture recognition from IMU
- vibration anomaly detection
- low-resolution person detection
- environmental anomaly detection

### Measure It

- arena size
- flash size
- inference latency
- active and sleep power
- battery-life estimate
- false positive/false negative behavior

### Ship It

A constrained-memory inference demo with firmware, model artifact, and power or latency measurements.

---

## Unit 6: Sensor Pipelines

### Learn

- camera ingest: MIPI CSI-2, USB, GigE, V4L2
- ISP pipeline: RAW, debayer, denoise, tone map, resize, color convert
- audio pipeline: I2S, ALSA, VAD, keyword spotting, ASR
- GStreamer and DeepStream
- multi-stream inference
- tracking and postprocessing
- zero-copy paths and buffer ownership

### Build It

Build one end-to-end sensor pipeline:

- camera -> preprocess -> detection -> tracking -> output
- microphone -> VAD -> feature extraction -> keyword/ASR -> output
- IMU -> filtering -> model -> anomaly/event output

### Measure It

- sensor-to-output latency
- preprocessing time
- inference time
- postprocessing time
- dropped frames or audio underruns
- memory copies
- CPU/GPU utilization

### Ship It

A sensor-to-model pipeline report with a latency breakdown and at least one zero-copy or copy-reduction improvement.

---

## Unit 7: Power, Thermal, And Reliability

### Learn

- thermal throttling
- nvpmodel/jetson_clocks-style power modes
- DVFS
- battery budgeting
- watchdogs
- model health checks
- long-run stability testing
- offline behavior and recovery

### Build It

Run a long-duration edge inference test:

- fixed workload
- realistic sensor input or replayed stream
- power/thermal logging
- automatic restart on failure
- basic telemetry

### Measure It

- sustained latency
- sustained throughput
- temperature
- throttling events
- power draw
- memory growth
- crash or restart behavior

### Ship It

A stability report that says what the device can sustain, not only what it can do for one benchmark run.

---

## Unit 8: Fleet And Product Operation

### Learn

- OTA model and software updates
- rollback and A/B slots
- device telemetry
- model/version compatibility
- privacy and local data retention
- edge/cloud routing
- monitoring drift and field failures
- secure boot and signed artifacts at a systems level

### Build It

Design a deployment plan for a small fleet:

- model packaging
- rollout stages
- rollback trigger
- telemetry schema
- health checks
- failure triage

### Measure It

- update time
- rollback time
- telemetry volume
- offline recovery behavior
- version compatibility checks

### Ship It

An edge AI operations plan that another engineer could use to ship the model to devices safely.

---

## Featured Deep Dives

Use these lectures as the technical core for LLM and wireless-oriented edge work:

- [Edge LLM Inference Internals](Edge%20LLM%20Inference%20Internals/Lecture-01.md)
- [Qwen Inference Optimization](Qwen%20Inference%20Optimization/README.md)
- [AI-Driven Wireless Communication](AI-Driven%20Wireless%20Communication/Lecture-01.md)

---

## Capstone

Build an edge AI system that includes:

- real sensor or replayed production-like input
- optimized model
- runtime deployment
- latency and throughput benchmark
- memory report
- power or thermal report
- health checks
- update or rollback plan

Good capstone examples:

- Jetson multi-camera detection and tracking system
- local LLM runtime with memory and thermal controls
- MCU keyword spotter with power budget
- industrial anomaly detector with OTA model updates
- multimodal robot perception node

The capstone is complete when another engineer can reproduce the deployment and understand the constraints that shaped the design.

---

## Exit Criteria

You are ready to claim edge AI specialization when you can:

- select hardware from workload requirements
- quantize and deploy a model with measured tradeoffs
- profile an edge inference runtime end to end
- build a sensor-to-model pipeline
- explain power and thermal behavior under sustained load
- design update, rollback, and telemetry paths
- ship a reproducible edge AI benchmark or case study
