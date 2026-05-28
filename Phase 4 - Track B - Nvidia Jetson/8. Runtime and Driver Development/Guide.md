# Track B §8 — Runtime & Driver Development for GPU/Jetson Inference

<div class="course-identity jetson-runtime" markdown="1">
<div class="course-identity__icon">RT</div>
<div markdown="1">
<p class="course-identity__eyebrow">Track B8 · Jetson Runtime & Drivers</p>
<p class="course-identity__title">Own the runtime and driver layer between Linux, CUDA, TensorRT, sensors, and application code.</p>
<p class="course-identity__meta">Artifact: runtime/driver case study · Measure: latency, memory copies, utilization, failures</p>
</div>
</div>


> *Own the software stack between the compiled model and the GPU hardware — CUDA runtime, TensorRT engine execution, DLA scheduling, and Linux driver interfaces.*

**Prerequisites:** Track B §1 (Jetson Platform, CUDA basics), Phase 1 §3 (Operating Systems — drivers, memory management), Phase 1 §4 (C++/CUDA).

**Layer mapping:** **Layer 3** (Runtime & Driver) of the AI chip stack. This module connects Layer 2 (compiler output — TensorRT engines, CUDA kernels) to Layer 5 (GPU/DLA hardware) through the driver and runtime software.

**Role targets:** GPU/Accelerator Runtime Engineer · CUDA Runtime Engineer · Inference Platform Engineer · Embedded Linux BSP Engineer (Jetson)

---

## Why runtime & driver matters for Jetson/GPU

The compiler produces optimized kernels and execution plans. The runtime makes them actually run: managing GPU memory, scheduling kernels across CUDA cores and DLA engines, handling multi-stream concurrency, and communicating with the Linux kernel driver. Understanding this layer is essential for hitting latency and throughput targets on real hardware.

---

## 1. CUDA Runtime & Driver Architecture

* **Two-level CUDA API:**
    * **Runtime API** (`cudart`): `cudaMalloc`, `cudaMemcpy`, `cudaLaunchKernel`, streams, events.
    * **Driver API** (`cuda`): `cuCtxCreate`, `cuModuleLoad`, `cuLaunchKernel` — lower-level, explicit context management.
    * When to use which: runtime API for most work; driver API for multi-context, dynamic module loading, or building frameworks.

* **CUDA context and streams:**
    * Context = per-GPU state (address space, modules, streams).
    * Streams = ordered queues of GPU work. Default stream vs explicit streams.
    * Multi-stream concurrency: overlap compute, DMA (H2D, D2H), and peer transfer.
    * Events for synchronization and timing.

* **Memory management:**
    * Device memory (`cudaMalloc`), pinned host memory (`cudaMallocHost`), unified memory (`cudaMallocManaged`).
    * Async memcpy with streams: `cudaMemcpyAsync`.
    * Memory pools (`cudaMallocAsync` / `cudaMemPool`) — reduce allocation overhead.
    * Jetson unified memory architecture: GPU and CPU share LPDDR5 — implications for zero-copy vs explicit copy.

* **Kernel launch mechanics:**
    * Grid → block → thread hierarchy. Occupancy calculator.
    * Launch configuration: `<<<grid, block, shared_mem, stream>>>`.
    * Cooperative groups, dynamic parallelism (advanced).

**Projects:**
* Write a multi-stream pipeline on Jetson: stream 1 does H2D + kernel A, stream 2 does kernel B + D2H, with event-based synchronization. Measure overlap with `nsys`.
* Compare unified memory (`cudaMallocManaged`) vs explicit copy (`cudaMalloc` + `cudaMemcpy`) for a CNN inference workload on Jetson Orin Nano. Profile with Nsight Systems.

---

## 2. TensorRT Runtime Deep Dive

* **Engine lifecycle:**
    * Build phase: `IBuilder` → `INetworkDefinition` → `IBuilderConfig` → `ICudaEngine`.
    * Serialization: save/load engine (`.engine` / `.plan` file). Why engines are hardware-specific.
    * Execution: `IExecutionContext` — one engine, multiple contexts for concurrent inference.

* **Memory management in TensorRT:**
    * Bindings: input/output tensor addresses. Static vs dynamic shapes.
    * Workspace memory: scratch space for tactics. Sizing trade-offs.
    * Device memory: pre-allocate all inference buffers; avoid per-inference allocation.

* **Execution and scheduling:**
    * Enqueue-based async execution: `context->enqueueV3(stream)`.
    * Multi-stream inference: separate streams for different models or batch sizes.
    * DLA integration: `config->setDeviceType(layer, DeviceType::kDLA)`.
    * DLA + GPU split: layers that DLA doesn't support fall back to GPU; runtime manages handoff.

* **Plugins:**
    * `IPluginV2DynamicExt` interface: custom layers not natively supported.
    * Plugin lifecycle: `getOutputDimensions`, `enqueue`, `serialize`.
    * When to write a plugin vs when to rely on TensorRT's fusion.

* **Dynamic shapes and batching:**
    * Optimization profiles: min/opt/max dimensions for each input.
    * Runtime shape specification before each inference.
    * Dynamic batching in serving: collect requests → pad to batch → infer → split results.

**Projects:**
* Build a TensorRT engine for YOLOv8 on Jetson Orin Nano with INT8 quantization. Benchmark FPS at different batch sizes (1, 4, 8).
* Write a TensorRT plugin for a custom activation function. Integrate into the engine and verify correctness.
* Implement a dual-engine pipeline: detection model on GPU, classification model on DLA, running concurrently with separate CUDA streams.

---

## 3. DLA (Deep Learning Accelerator) Runtime

* **DLA architecture on Orin:**
    * Two DLA engines (NVDLA-based) on Orin Nano.
    * Supported layers: Conv, deconv, pooling, LRN, batch norm, element-wise, softmax.
    * Limitations: no custom ops, limited dynamic shapes, restricted concat/split behavior.

* **DLA programming model:**
    * Compile-time: TensorRT marks layers for DLA during engine build.
    * Runtime: TensorRT runtime dispatches DLA-assigned layers to DLA engine via `nvhost` driver.
    * Fallback: unsupported layers run on GPU; data transfer between DLA and GPU memory.

* **DLA driver stack:**
    * `nvhost-nvdla` kernel driver: submits DLA tasks, manages DLA memory.
    * `nvdla_runtime` firmware on DLA engine.
    * NVDLA open-source reference: [github.com/nvdla](http://nvdla.org) — study for understanding DLA runtime concepts.

* **DLA performance tuning:**
    * Maximize DLA-resident layers: fewer GPU↔DLA transitions.
    * DLA-friendly model design: avoid unsupported ops, prefer standard convolutions.
    * Power efficiency: DLA is more power-efficient than GPU for supported ops.

**Projects:**
* Take a MobileNetV2 and deploy with `setDeviceType(kDLA)` for all supported layers. Log which layers fall back to GPU. Optimize the model to maximize DLA coverage.
* Measure power consumption (Jetson `tegrastats`) for GPU-only vs DLA-only vs mixed inference.

---

## 4. NVIDIA Kernel Driver Stack on Jetson

* **GPU driver (`nvgpu`):**
    * Open-source Tegra GPU driver (not the proprietary desktop driver).
    * Device tree configuration: GPU clocks, power domains, memory carve-outs.
    * `nvgpu` sysfs: frequency scaling, power capping, hardware counters.
    * How user-space CUDA calls reach `nvgpu`: `ioctl` interface, channel submission.

* **Video and camera drivers:**
    * `nvcsi` → `vi` (Video Input) → `isp` pipeline.
    * Sensor driver (`v4l2_subdev`): I2C register programming, mode tables.
    * How sensor data flows to GPU memory for inference (zero-copy path via NvBufSurface).

* **Display and multimedia:**
    * `tegra-dc` / `nvdisplay`: display controller driver.
    * NVDEC/NVENC: hardware video decode/encode. GStreamer integration via `nvv4l2decoder`.

* **Power management drivers:**
    * DVFS (Dynamic Voltage and Frequency Scaling): `devfreq` framework.
    * Power domains and clock tree: `clk_tegra`, `tegra-pmc`.
    * Thermal management: `thermal_zone`, throttling policies.
    * `nvpmodel`: power mode profiles, MAX_N vs 15W vs 7W modes.
    * `jetson_clocks`: pin max clocks for benchmarking.

* **Kernel module development on Jetson:**
    * Building out-of-tree modules against L4T kernel headers.
    * Device tree overlay for custom hardware on carrier board.
    * Debugging: `dmesg`, `ftrace`, `/sys/kernel/debug`.

**Projects:**
* Write a minimal kernel module that reads GPU utilization from `nvgpu` sysfs and exposes a `/proc/gpu_monitor` interface.
* Add a device tree overlay for a custom SPI sensor on your Jetson carrier board. Write a `v4l2_subdev` stub driver that reads sensor ID over SPI.
* Profile the camera-to-inference pipeline: measure latency from sensor capture to inference result using `nsys` + kernel tracepoints.

---

## 5. Multi-Engine Scheduling & System-Level Runtime

* **Concurrent engine execution:**
    * GPU + DLA + NVDEC + PVA (Programmable Vision Accelerator) running simultaneously.
    * Per-engine CUDA streams: avoid serialization across engines.
    * Priority scheduling: high-priority streams (`cudaStreamCreateWithPriority`) for latency-critical inference.

* **Real-time inference scheduling:**
    * Frame-rate-driven scheduling: sensor frame → preprocess → infer → postprocess → actuate.
    * Deadline-aware execution: drop frames vs queue frames vs adaptive batch.
    * CPU-GPU synchronization: minimizing CPU blocking with async APIs + events.

* **Multi-model deployment:**
    * Multiple TensorRT engines sharing GPU: time-slicing vs MPS (Multi-Process Service).
    * Memory partitioning: pre-allocate buffers per model to avoid fragmentation.
    * Model switching: engine loading latency, warm-up strategies.

* **DeepStream as system-level runtime:**
    * GStreamer-based pipeline: decode → preprocess → infer → track → display.
    * `nvinfer` plugin: wraps TensorRT engine inside a GStreamer element.
    * Multi-stream video: process N cameras in one pipeline.
    * Custom `GstBuffer` metadata for passing inference results downstream.

* **Triton Inference Server on Jetson:**
    * Model repository, dynamic batching, concurrent model execution.
    * Backend options: TensorRT, ONNX Runtime, PyTorch, custom C++ backend.
    * Jetson deployment considerations: memory constraints, power budget.

**Projects:**
* Build a 4-camera DeepStream pipeline on Jetson: decode → detect (GPU) → classify (DLA) → track → OSD → display. Measure end-to-end latency.
* Deploy two models (detection + segmentation) on Triton Inference Server on Jetson. Configure dynamic batching and measure throughput under load.
* Implement a priority-based scheduler: safety-critical model gets high-priority stream, telemetry model gets low-priority. Verify with `nsys` that high-priority is never starved.

---

## Relationship to Other Modules

| This module (B §8) | Connects to |
|---------------------|-------------|
| CUDA runtime & memory | B §1 (Jetson Platform — CUDA basics) |
| TensorRT engine execution | B §5 (Application Development — ML/AI) |
| DLA runtime | B §1 deep dives (DLA architecture, tensor cores) |
| Kernel driver stack | B §3 (L4T Customization — kernel, device tree) |
| Power management | B §4 (FSP — SPE firmware, power control) |
| Inference runtime (Triton, DeepStream) | Phase 3 (Neural Networks — models to deploy) |
| Compiler output → runtime input | Track C (ML Compiler — generates engines/kernels) |

---

## Build Summary

| Section | Hands-on deliverable |
|---------|---------------------|
| §1 CUDA runtime | Multi-stream pipeline, unified vs explicit memory comparison |
| §2 TensorRT runtime | INT8 engine build, custom plugin, dual-engine pipeline |
| §3 DLA runtime | DLA coverage maximization, power measurement |
| §4 Kernel drivers | GPU monitor module, device tree overlay + sensor driver |
| §5 System runtime | 4-camera DeepStream pipeline, Triton deployment, priority scheduler |
