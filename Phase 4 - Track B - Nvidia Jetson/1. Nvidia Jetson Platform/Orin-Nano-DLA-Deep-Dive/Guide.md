# Orin Nano 8GB — Deep Learning Accelerator (DLA) Deep Dive

<div class="course-identity auto-course" style="--course-accent: #be123c; --course-accent-rgb: 190, 18, 60;" markdown="1">
<div class="course-identity__icon">ON8D</div>
<div markdown="1">
<p class="course-identity__eyebrow">Deep Dive · Jetson Track</p>
<p class="course-identity__title">Specialized course identity for Orin Nano 8GB — Deep Learning Accelerator (DLA) Deep Dive.</p>
<p class="course-identity__meta">Artifact: Jetson integration demo · Measure: latency, memory, power, logs</p>
</div>
</div>


> **Scope:** Production-level understanding of the DLA on Jetson Orin Nano 8GB — hardware architecture, memory interaction, software stack, TensorRT integration, layer support, multi-engine scheduling, performance profiling, and production deployment patterns.
>
> **Prerequisites:** Familiarity with the [Orin Nano memory architecture](../Orin-Nano-Memory-Architecture/Guide.md) (CMA, SMMU, zero-copy) and [kernel internals](../Orin-Nano-Kernel-Internals/Guide.md) (driver model, module loading).

---


## 1. What Is DLA

DLA (Deep Learning Accelerator) is a **dedicated hardware block** inside the Tegra SoC designed specifically for neural network inference. It is not a GPU, not a CPU — it is a fixed-function accelerator optimized for low-power, high-efficiency AI workloads.

Key characteristics:

* **Purpose-built for inference** — convolution, pooling, activation, normalization
* **Power-efficient** — achieves high TOPS/watt compared to GPU
* **Deterministic latency** — no contention with other GPU workloads
* **Runs in parallel** with CPU and GPU — true heterogeneous computing

DLA does **not** support training, only inference. It does **not** support all neural network operations — unsupported layers fall back to GPU.

---

## 2. DLA on Orin Nano 8GB — Specifications

| Specification            | Value                                |
|--------------------------|--------------------------------------|
| Number of DLA engines    | 1                                    |
| Peak performance (INT8)  | Up to 10 TOPS                        |
| Peak performance (FP16)  | Up to 5 TFLOPS                       |
| Supported precisions     | INT8, FP16                           |
| On-chip SRAM             | Small buffer for weights/activations |
| Memory access            | System DRAM via DMA + SMMU           |
| Power consumption        | Significantly lower than GPU         |

Note: Orin Nano 8GB has **1 DLA engine**. Higher-end Orin modules (NX, AGX) have 2 DLA engines, enabling parallel inference on two models simultaneously.

---

## 3. DLA vs GPU vs CPU — When to Use Each

| Feature              | CPU                  | GPU                    | DLA                           |
|----------------------|----------------------|------------------------|-------------------------------|
| Architecture         | General-purpose      | Massively parallel     | Fixed-function AI accelerator |
| Best workload        | OS, control logic    | Parallel FP/INT ops    | CNN/RNN inference             |
| Supported operations | Everything           | Everything (CUDA)      | Subset (conv, pool, etc.)     |
| Power consumption    | High for compute     | Medium-high            | Low                           |
| Latency              | Higher               | Medium                 | Very low (deterministic)      |
| Precision            | FP32/FP64            | FP32/FP16/INT8/TF32    | INT8/FP16 only                |
| Programmability      | Full (C/C++/Python)  | Full (CUDA)            | Via TensorRT only             |
| Parallel with others | Yes                  | Yes                    | Yes                           |

### Decision Matrix

| Scenario                                     | Best Engine |
|----------------------------------------------|-------------|
| Single model, maximum throughput             | GPU         |
| Single model, minimum power                  | DLA         |
| Two models simultaneously                   | DLA + GPU   |
| Model with many unsupported layers           | GPU         |
| Battery-powered device                       | DLA         |
| Pre/post-processing + inference              | CPU + DLA   |
| Real-time video + inference + display        | GPU + DLA   |

### The Power Argument

On Orin Nano 8GB (15W TDP total):

* Running inference on GPU: GPU consumes ~5–8W, leaving less for CPU and I/O
* Running inference on DLA: DLA consumes ~1–3W, freeing power budget for GPU (display, encode) and CPU

In power-constrained systems (battery, solar, thermal-limited enclosures), DLA can be the difference between meeting and missing the power budget.

---

## 4. DLA Hardware Architecture

### Block Diagram

```
┌─────────────────────────────────────────────────┐
│                  DLA Engine                      │
│                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ Convolution│  │ SDP      │  │ PDP      │       │
│  │ Core      │  │ (Single  │  │ (Planar  │       │
│  │           │  │  Data    │  │  Data    │       │
│  │ MAC array │  │  Proc.)  │  │  Proc.)  │       │
│  └─────┬─────┘  └─────┬────┘  └─────┬────┘       │
│        │              │              │            │
│  ┌─────┴──────────────┴──────────────┴─────┐     │
│  │           Internal Data Bus              │     │
│  └─────────────────┬────────────────────────┘     │
│                    │                              │
│  ┌─────────────────┴────────────────────────┐     │
│  │          SRAM Buffer (on-chip)            │     │
│  └─────────────────┬────────────────────────┘     │
│                    │                              │
│  ┌─────────────────┴────────────────────────┐     │
│  │          DMA Engine                       │     │
│  │  (reads/writes tensors from/to DRAM)      │     │
│  └─────────────────┬────────────────────────┘     │
│                    │                              │
└────────────────────┼──────────────────────────────┘
                     │
                     ↓
              SMMU → System DRAM
```

### Core Components

#### Convolution Core

* Contains the MAC (Multiply-Accumulate) array
* Heart of DLA — performs convolution, matrix multiplication, deconvolution
* Optimized for INT8 and FP16 data types
* Supports various kernel sizes (1x1, 3x3, 5x5, 7x7, etc.)
* Handles strided and dilated convolutions

#### SDP (Single Data Processor)

* Performs element-wise operations after convolution
* Handles: bias addition, batch normalization, ReLU, PReLU, sigmoid, tanh
* Operates on the output of the convolution core before writing to memory
* Fuses multiple post-convolution operations to avoid memory round-trips

#### PDP (Planar Data Processor)

* Performs pooling operations (max pool, average pool)
* Operates on 2D spatial data
* Supports various pool sizes and strides

#### CDP (Channel Data Processor)

* Performs channel-wise operations
* Local Response Normalization (LRN)
* Channel-wise scaling

#### SRAM Buffer

* Small on-chip memory for staging weights and activations
* Reduces DRAM bandwidth consumption for frequently accessed data
* DLA compiler decides what to cache in SRAM vs. stream from DRAM

#### DMA Engine

* Moves input tensors from DRAM into DLA processing cores
* Writes output tensors from DLA back to DRAM
* Uses IOVA addresses (mapped through SMMU)
* Supports scatter-gather for non-contiguous buffers

---

## 5. DLA Memory Interaction

DLA does **not** have large dedicated memory. It uses system DRAM for all tensor storage.

### Memory Flow

```
System DRAM (8GB LPDDR5, shared with CPU/GPU)
      ↑ ↓
    SMMU (ARM SMMU v2)
      ↑ ↓
  IOVA address space (DLA's view of memory)
      ↑ ↓
  DMA Engine (inside DLA)
      ↑ ↓
  SRAM Buffer (small, on-chip)
      ↑ ↓
  Processing Cores (Conv, SDP, PDP, CDP)
```

### Buffer Allocation

DLA buffers are allocated by the TensorRT runtime / DLA driver:

1. **Input tensors** — allocated from CMA (contiguous) or carved-out memory
2. **Weight tensors** — loaded from the serialized TensorRT engine file
3. **Intermediate tensors** — allocated for layer-to-layer data flow
4. **Output tensors** — allocated from CMA, returned to the caller

All buffers are mapped through SMMU so DLA accesses them via IOVA.

### Zero-Copy With GPU

When a DLA layer's output feeds into a GPU layer (or vice versa):

```
DLA output buffer (in DRAM)
   ↓
Same physical pages
   ↓
GPU SMMU maps same pages at different IOVA
   ↓
GPU reads data — no copy needed
```

TensorRT handles this automatically when building a hybrid DLA+GPU engine.

### Memory Budget Impact

DLA buffers consume system DRAM just like GPU and CPU allocations:

| Component          | Typical Memory   |
|--------------------|------------------|
| Model weights      | 10–100 MB        |
| Input tensor       | 1–12 MB          |
| Intermediate       | 10–50 MB         |
| Output tensor      | < 1 MB           |
| **Total per model** | **20–160 MB**   |

On an 8GB system, this is significant. Plan memory budgets across CPU + GPU + DLA workloads.

---

## 6. Software Stack — From Model to DLA Execution

```
┌──────────────────────────────────────────┐
│  User Application                         │
│  (Python/C++ — inference request)         │
└──────────────────┬───────────────────────┘
                   ↓
┌──────────────────────────────────────────┐
│  TensorRT Runtime                         │
│  (engine deserialization, execution)      │
│  Selects DLA or GPU per layer             │
└──────────────────┬───────────────────────┘
                   ↓
┌──────────────────────────────────────────┐
│  libnvdla (DLA runtime library)           │
│  Programs DLA registers                   │
│  Manages DMA descriptors                  │
│  Handles synchronization                  │
└──────────────────┬───────────────────────┘
                   ↓
┌──────────────────────────────────────────┐
│  Kernel Driver (nvdla.ko / nvhost)        │
│  Allocates CMA buffers                    │
│  Creates SMMU/IOVA mappings               │
│  Submits work to DLA hardware             │
│  Handles completion interrupts            │
└──────────────────┬───────────────────────┘
                   ↓
┌──────────────────────────────────────────┐
│  DLA Hardware                             │
│  Executes neural network layers           │
│  DMA reads/writes tensors from/to DRAM    │
└──────────────────────────────────────────┘
```

---

## 7. TensorRT DLA Integration

### Building a DLA-Enabled Engine

```python
import tensorrt as trt

logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

# Parse ONNX model
with open("model.onnx", "rb") as f:
    parser.parse(f.read())

config = builder.create_builder_config()
config.max_workspace_size = 1 << 30  # 1 GB

# Enable DLA
config.default_device_type = trt.DeviceType.DLA
config.DLA_core = 0  # Use DLA core 0

# Allow GPU fallback for unsupported layers
config.set_flag(trt.BuilderFlag.GPU_FALLBACK)

# Use FP16 or INT8 (DLA does not support FP32)
config.set_flag(trt.BuilderFlag.FP16)
# Or for INT8:
# config.set_flag(trt.BuilderFlag.INT8)
# config.int8_calibrator = MyCalibrator()

engine = builder.build_engine(network, config)

# Serialize engine
with open("model_dla.engine", "wb") as f:
    f.write(engine.serialize())
```

### Key TensorRT DLA Options

| Option                   | Purpose                                        |
|--------------------------|------------------------------------------------|
| `default_device_type`    | Set all layers to DLA by default               |
| `DLA_core`               | Select which DLA engine (0 or 1)               |
| `GPU_FALLBACK`           | Allow unsupported layers to run on GPU          |
| `FP16` / `INT8`          | DLA requires reduced precision                  |
| `set_device_type(layer)` | Override per-layer device assignment            |

### Per-Layer Device Assignment

For fine-grained control, assign specific layers to DLA or GPU:

```python
for i in range(network.num_layers):
    layer = network.get_layer(i)
    if can_run_on_dla(layer):
        config.set_device_type(layer, trt.DeviceType.DLA)
    else:
        config.set_device_type(layer, trt.DeviceType.GPU)
```

### Inspecting DLA Layer Assignment

After building the engine, check which layers run on DLA:

```bash
# Use trtexec to build and profile
trtexec --onnx=model.onnx --useDLACore=0 --fp16 --allowGPUFallback --verbose

# Output shows per-layer device assignment:
# Layer: conv1 ... Device: DLA
# Layer: unsupported_op ... Device: GPU (fallback)
```

---

## 8. Supported Layers and Precision

### DLA-Supported Operations

| Operation              | Supported | Notes                                  |
|------------------------|-----------|----------------------------------------|
| Convolution (2D)       | Yes       | All kernel sizes, strides, dilation    |
| Deconvolution          | Yes       | Transposed convolution                  |
| Fully connected        | Yes       | Via 1x1 convolution                     |
| Pooling (max, avg)     | Yes       | Various sizes and strides               |
| ReLU                   | Yes       | Fused with convolution in SDP           |
| PReLU / Leaky ReLU     | Yes       | Fused in SDP                            |
| Sigmoid                | Yes       | Via SDP                                 |
| Tanh                   | Yes       | Via SDP                                 |
| Batch Normalization    | Yes       | Fused with convolution in SDP           |
| Element-wise (add/mul) | Yes       | Via SDP                                 |
| Concatenation          | Yes       | Channel concatenation                   |
| Softmax                | Limited   | May fall back to GPU                    |
| Resize / Upsample      | Limited   | Nearest-neighbor only, some constraints |
| Transpose              | No        | Falls back to GPU                       |
| Attention (QKV)        | No        | Falls back to GPU                       |
| Custom plugins         | No        | DLA only runs compiled operations       |
| Dynamic shapes         | No        | Fixed input dimensions required          |

### Precision Support

| Precision | DLA Support | Notes                                   |
|-----------|-------------|-----------------------------------------|
| FP32      | No          | Must convert to FP16 or INT8            |
| FP16      | Yes         | Default for DLA                          |
| INT8      | Yes         | Requires calibration data                |
| TF32      | No          | GPU-only feature                         |
| BF16      | No          | Not supported on Orin DLA                |

### What Happens With Unsupported Layers

If `GPU_FALLBACK` is enabled:

```
DLA layers → DLA engine
   ↓
Unsupported layer → data transfer → GPU
   ↓
GPU executes layer
   ↓
Next DLA layer → data transfer → back to DLA
```

Each DLA↔GPU transition involves a memory synchronization (not a copy if using zero-copy, but a sync fence). Too many transitions add latency.

### Maximizing DLA Utilization

* Use architectures with DLA-friendly operations (convolution-heavy CNNs)
* Avoid attention mechanisms, dynamic shapes, and custom operations
* Fuse batch normalization into convolution before export
* Use INT8 for maximum DLA throughput
* Minimize DLA↔GPU transitions by grouping unsupported layers

---

## 9. DLA Execution Flow — Step by Step

### Example: Image Classification (ResNet50 on DLA)

```
Step 1: Application submits inference request
        Input: 224x224x3 image tensor (FP16)
             ↓
Step 2: TensorRT runtime selects DLA engine
        Deserializes compiled DLA loadable
             ↓
Step 3: libnvdla programs DLA registers
        Sets up DMA descriptors for input/weights/output
        All addresses are IOVA (mapped through SMMU)
             ↓
Step 4: DLA DMA engine reads input tensor from DRAM
        Streams data into on-chip SRAM buffer
             ↓
Step 5: Convolution core processes layer 1
        MAC array performs conv2d on input
        SDP applies batch norm + ReLU (fused)
        PDP applies max pooling
             ↓
Step 6: Intermediate result written to DRAM (or kept in SRAM)
        Next layer reads from previous output
             ↓
Step 7: Repeat for all DLA-compatible layers
        Conv → BN → ReLU → Pool → Conv → ...
             ↓
Step 8: If unsupported layer encountered:
        DLA writes intermediate to DRAM
        GPU reads same physical pages (zero-copy via SMMU)
        GPU executes unsupported layer(s)
        DLA reads GPU output and continues
             ↓
Step 9: Final output tensor written to DRAM via DMA
             ↓
Step 10: DLA raises completion interrupt
         Kernel driver signals userspace
         Application reads output tensor
```

---

## 10. Multi-Engine Scheduling (DLA + GPU)

### Heterogeneous Execution

On Orin Nano, you can run workloads on DLA and GPU simultaneously:

```
┌─────────────────────────────────────────┐
│               Time →                     │
│                                          │
│  DLA:  [Model A inference ][Model A    ] │
│  GPU:  [Model B inference      ][post] │
│  CPU:  [pre-process][      ][result]    │
│                                          │
└─────────────────────────────────────────┘
```

### Pipeline Architecture

For real-time video inference:

```
Frame N:    CPU pre-process → DLA inference → CPU post-process
Frame N+1:  CPU pre-process → GPU inference → CPU post-process
Frame N+2:  CPU pre-process → DLA inference → CPU post-process

DLA and GPU alternate or run different models in parallel.
```

### TensorRT Multi-Stream Execution

```python
import tensorrt as trt
import pycuda.driver as cuda

# Create two execution contexts
context_dla = engine_dla.create_execution_context()
context_gpu = engine_gpu.create_execution_context()

# Create two CUDA streams
stream_dla = cuda.Stream()
stream_gpu = cuda.Stream()

# Execute in parallel
context_dla.execute_async_v2(bindings_dla, stream_dla.handle)
context_gpu.execute_async_v2(bindings_gpu, stream_gpu.handle)

# Wait for both
stream_dla.synchronize()
stream_gpu.synchronize()
```

### Benefits of DLA + GPU Parallelism

| Metric         | GPU Only     | DLA Only    | DLA + GPU      |
|----------------|-------------|-------------|----------------|
| Throughput     | 1x           | 0.5–0.8x   | 1.3–1.8x       |
| Power          | High         | Low         | Medium          |
| Latency        | Medium       | Low         | Medium (pipelined) |
| GPU availability | 0%         | 100%        | 100% (for other tasks) |

Running inference on DLA frees the GPU for:

* Display rendering
* Video encode/decode (NVENC/NVDEC)
* Additional CUDA workloads
* GStreamer processing

---

## 11. DLA Kernel Driver

### Driver Architecture

The DLA kernel driver is part of the nvhost subsystem:

```
/dev/nvhost-nvdla0        ← DLA engine 0 device node
/dev/nvhost-nvdla1        ← DLA engine 1 (if present)
```

### Driver Responsibilities

| Responsibility        | Details                                         |
|-----------------------|-------------------------------------------------|
| Buffer allocation     | Allocates CMA buffers for DLA DMA               |
| SMMU mapping          | Maps physical pages to DLA's IOVA space          |
| Work submission       | Programs DLA registers, starts execution          |
| Interrupt handling    | Receives completion interrupt, signals userspace  |
| Power management      | Clock gating, power gating when idle              |
| Error handling        | Detects DLA faults, reports to userspace          |

### Module Loading

```bash
# Check if DLA driver is loaded
lsmod | grep nvdla
# nvdla                  12345  0

# Check device nodes
ls /dev/nvhost-nvdla*
# /dev/nvhost-nvdla0

# Check driver messages
dmesg | grep nvdla
# nvdla 15880000.nvdla0: probed
```

### DLA Clock and Power

DLA has its own clock domain managed by BPMP:

```bash
# Check DLA clock rate
cat /sys/kernel/debug/clk/clk_summary | grep dla

# DLA power domain
cat /sys/kernel/debug/bpmp/debug/regulator/*/name | grep dla
```

DLA is power-gated when idle — it consumes near-zero power when not executing inference.

---

## 12. DLA Memory Path — Full Pipeline

### Complete Data Flow

```
1. TensorRT allocates input buffer
   └→ dma_alloc_coherent() → CMA region → physical pages
   └→ SMMU maps pages → IOVA for DLA

2. Application fills input buffer (e.g., camera frame)
   └→ If from camera: DMA-BUF import (zero-copy from ISP)
   └→ If from CPU: memcpy into mapped buffer

3. TensorRT submits inference to DLA
   └→ libnvdla programs DMA descriptors with IOVAs
   └→ ioctl to /dev/nvhost-nvdla0

4. Kernel driver submits work
   └→ Writes to DLA control registers
   └→ DLA starts execution

5. DLA DMA engine reads input tensor
   └→ IOVA → SMMU translation → physical DRAM
   └→ Streams into on-chip SRAM

6. DLA processes layers
   └→ Conv core → SDP → PDP (all on-chip)
   └→ Intermediate results: SRAM or spill to DRAM

7. DLA DMA engine writes output tensor
   └→ IOVA → SMMU → physical DRAM

8. DLA raises IRQ
   └→ Kernel driver handles interrupt
   └→ Signals completion to userspace

9. Application reads output
   └→ Same mapped buffer (zero-copy to CPU)
   └→ Or GPU reads same pages (zero-copy via GPU SMMU)
```

---

## 13. CPU/GPU/DLA/SMMU/CMA Interaction Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        8GB LPDDR5                            │
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │ OS/Kernel │  │ CUDA     │  │ CMA      │  │ Carve-   │    │
│  │ Memory   │  │ Memory   │  │ Region   │  │ outs     │    │
│  │          │  │ (GPU)    │  │          │  │ (BPMP,   │    │
│  │          │  │          │  │          │  │  OP-TEE) │    │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────────┘    │
│       │              │              │                        │
└───────┼──────────────┼──────────────┼────────────────────────┘
        │              │              │
   ┌────┴────┐    ┌────┴────┐    ┌────┴────────────────────┐
   │  CPU    │    │  GPU    │    │  SMMU                    │
   │  MMU    │    │  SMMU   │    │  (shared by DLA, ISP,   │
   │         │    │  context│    │   VI, NVENC, NVDEC)      │
   │  VA→PA  │    │  IOVA→PA│    │  IOVA→PA                │
   └────┬────┘    └────┬────┘    └────┬────────┬───────────┘
        │              │              │        │
   ┌────┴────┐    ┌────┴────┐    ┌────┴───┐ ┌──┴──────┐
   │  CPU    │    │  GPU    │    │  DLA   │ │  ISP    │
   │  A78AE  │    │ Ampere  │    │ Engine │ │  Camera │
   │  cores  │    │ 1024    │    │        │ │  pipe   │
   │         │    │ cores   │    │ MAC +  │ │         │
   │         │    │         │    │ SRAM + │ │         │
   │         │    │         │    │ DMA    │ │         │
   └─────────┘    └─────────┘    └────────┘ └─────────┘
```

### Data Flow Examples

**Camera → DLA inference (zero-copy):**

```
Camera sensor
 → NVCSI → VI → ISP
 → CMA buffer (physical pages)
 → SMMU maps to ISP IOVA (write) and DLA IOVA (read)
 → DLA reads same physical pages — zero copy
 → DLA output to CMA
 → CPU reads result — zero copy
```

**Camera → GPU inference → DLA post-processing:**

```
Camera → CMA buffer
 → GPU SMMU maps buffer → GPU inference
 → GPU output to CUDA memory
 → DLA SMMU maps same pages → DLA post-processing
 → DLA output to CMA → CPU reads result
```

**DLA + GPU parallel inference (two models):**

```
Input → CMA buffer
 ├→ DLA SMMU → DLA runs Model A
 └→ GPU SMMU → GPU runs Model B
Both access DRAM through SMMU, no copies between them
Results available simultaneously
```

---

## 14. Performance Characteristics

### Typical Inference Latency (Orin Nano 8GB)

| Model         | Precision | Engine | Batch | Latency    | Power   |
|---------------|-----------|--------|-------|------------|---------|
| ResNet-50     | INT8      | DLA    | 1     | ~5–8 ms    | ~1.5W   |
| ResNet-50     | INT8      | GPU    | 1     | ~3–5 ms    | ~5W     |
| ResNet-50     | FP16      | DLA    | 1     | ~8–12 ms   | ~2W     |
| MobileNetV2   | INT8      | DLA    | 1     | ~2–4 ms    | ~1W     |
| MobileNetV2   | INT8      | GPU    | 1     | ~1–3 ms    | ~3W     |
| YOLOv5s       | FP16      | DLA    | 1     | ~15–25 ms  | ~2.5W   |
| YOLOv5s       | FP16      | GPU    | 1     | ~8–12 ms   | ~6W     |

### Throughput vs Power Efficiency

| Metric                  | GPU        | DLA         |
|-------------------------|------------|-------------|
| Raw throughput (FPS)    | Higher     | Lower       |
| TOPS per watt           | Lower      | **Higher**  |
| Frames per joule        | Lower      | **Higher**  |

DLA wins on efficiency (TOPS/watt), GPU wins on raw speed. Choose based on your constraint — power budget or throughput target.

### INT8 vs FP16 on DLA

| Precision | Throughput | Accuracy | Calibration Required |
|-----------|-----------|----------|----------------------|
| FP16      | 1x        | Higher   | No                   |
| INT8      | ~2x       | Lower    | Yes (calibration dataset) |

INT8 roughly doubles DLA throughput. Use INT8 when accuracy loss is acceptable (test with your dataset).

---

## 15. Profiling DLA Workloads

### trtexec — Quick Profiling

```bash
# Profile DLA inference
trtexec \
    --onnx=model.onnx \
    --useDLACore=0 \
    --fp16 \
    --allowGPUFallback \
    --verbose \
    --iterations=100

# Key output:
# [DLA] Layer conv1: 1.2ms
# [GPU] Layer unsupported_op: 0.5ms (fallback)
# [DLA] Layer conv2: 0.8ms
# Total: 5.3ms
# DLA utilization: 78%
```

### Nsight Systems — Detailed Timeline

```bash
nsys profile --trace=cuda,nvtx,osrt \
    trtexec --loadEngine=model_dla.engine --iterations=50

# Open in Nsight Systems GUI
# Shows:
# - DLA execution blocks
# - GPU fallback blocks
# - DMA transfers
# - CPU overhead
# - DLA↔GPU sync points
```

### DLA-Specific Metrics

```bash
# Check DLA utilization via tegrastats
tegrastats --interval 1000
# Output includes DLA% utilization

# DLA clock frequency
cat /sys/kernel/debug/clk/clk_summary | grep dla
```

### Identifying Bottlenecks

| Symptom                        | Cause                          | Solution                           |
|--------------------------------|--------------------------------|------------------------------------|
| Low DLA utilization            | Many GPU fallback layers       | Use DLA-friendly architecture      |
| High DLA-GPU transition time   | Frequent engine switches       | Group DLA/GPU layers contiguously  |
| DLA latency higher than GPU    | Model too small for DLA        | Use GPU instead (overhead > benefit)|
| Inconsistent DLA latency       | Memory bandwidth contention    | Reduce concurrent DRAM access      |

---

## 16. DLA Limitations and Fallback Behavior

### Hard Limitations

* **No FP32** — must use FP16 or INT8
* **No dynamic shapes** — input dimensions must be fixed at build time
* **No custom CUDA plugins** — DLA only runs compiled operations
* **Limited layer support** — see Section 8 for full list
* **Single batch only** — batch > 1 is not supported on all layers
* **No in-place operations** — every operation writes to a new buffer

### Fallback Behavior

When `GPU_FALLBACK` is enabled and a layer is not supported on DLA:

1. TensorRT builds a hybrid engine with DLA and GPU sections
2. At runtime, DLA executes its layers, then synchronizes
3. GPU picks up the unsupported layers
4. After GPU finishes, DLA resumes (if more DLA layers follow)

Each DLA→GPU→DLA transition adds:

* Memory synchronization overhead (~0.1–0.5 ms)
* Context switch overhead
* No data copy (zero-copy via SMMU)

### When NOT to Use DLA

* Transformer-heavy models (attention is not supported)
* Models with many custom operations
* Models requiring FP32 precision
* Very small models where DLA setup overhead exceeds computation
* Latency-critical single-model inference where GPU is faster

---

## 17. Production Deployment Patterns

### Pattern 1: DLA-Only Inference (Maximum Power Efficiency)

```
Camera → pre-process (CPU) → DLA inference → post-process (CPU) → output
GPU: idle / display only
```

Best for: battery-powered devices, thermal-constrained systems, always-on monitoring.

### Pattern 2: DLA + GPU Pipeline (Maximum Throughput)

```
Camera → pre-process (CPU)
   ├→ DLA: detection model (lightweight, e.g., MobileNet-SSD)
   └→ GPU: classification model (heavier, e.g., ResNet-50)
Both run simultaneously on alternating frames or different ROIs.
```

Best for: multi-model systems, video analytics with detection + classification.

### Pattern 3: DLA Primary, GPU Fallback (Balanced)

```
Full model compiled for DLA with GPU fallback enabled.
DLA handles convolutions, GPU handles unsupported ops.
TensorRT manages transitions automatically.
```

Best for: single-model deployment where some layers are unsupported.

### Pattern 4: DLA for Always-On + GPU for On-Demand

```
DLA: continuously running lightweight detection (person, vehicle)
GPU: idle, wakes up for heavy processing when DLA detects event
   → GPU runs detailed classification, tracking, or segmentation
   → GPU returns to idle
```

Best for: surveillance, smart cameras, event-driven systems. Minimizes average power.

### Engine Serialization for Production

Always serialize (save) TensorRT engines for production deployment:

```python
# Build once (slow — compile time)
engine = builder.build_engine(network, config)

# Serialize
with open("model_dla_int8.engine", "wb") as f:
    f.write(engine.serialize())

# Deploy: deserialize (fast — load time)
runtime = trt.Runtime(logger)
with open("model_dla_int8.engine", "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())
```

Engine files are device-specific — an engine built for Orin Nano will not work on Orin NX or AGX Orin. Rebuild for each target.

---

## 18. Common DLA Issues and Solutions

### DLA Engine Build Fails

```
[TensorRT] ERROR: Layer X is not supported on DLA
```

**Cause:** Model contains unsupported operations.

**Solution:** Enable `GPU_FALLBACK`, or modify the model to use DLA-friendly operations.

### DLA Inference Slower Than GPU

**Cause:** Too many DLA↔GPU transitions, or model is too small for DLA overhead.

**Solution:** Profile with `trtexec --verbose` to count transitions. If many, consider GPU-only. If model is tiny, GPU is likely faster.

### DLA Accuracy Degradation (INT8)

**Cause:** Poor INT8 calibration or sensitive layers quantized incorrectly.

**Solution:**
* Use a representative calibration dataset (>500 images)
* Use per-channel quantization instead of per-tensor
* Keep sensitive layers (first/last conv, skip connections) in FP16

### DLA Buffer Allocation Failure

```
nvdla: failed to allocate buffer
```

**Cause:** CMA exhaustion or fragmentation.

**Solution:** See [Memory Architecture Guide — CMA](../Orin-Nano-Memory-Architecture/Guide.md#7-cma--contiguous-memory-allocator) for CMA sizing and fragmentation mitigation.

### DLA Not Detected

```
ls /dev/nvhost-nvdla*
# No output
```

**Cause:** DLA device tree node disabled, driver not loaded, or JetPack version mismatch.

**Solution:** Check `dmesg | grep nvdla`, verify DTB has DLA nodes enabled, ensure `nvdla.ko` is loaded.

---

## 19. References

* [NVIDIA TensorRT — DLA Documentation](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla_topic) — official DLA integration guide
* [NVIDIA TensorRT — DLA Supported Layers](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-lay-supp-matrix) — layer support matrix
* [NVIDIA Jetson Linux — DLA](https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/) — Jetson DLA documentation
* [trtexec Reference](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#trtexec) — TensorRT command-line profiling tool
* Main guide: [Nvidia Jetson Platform Guide](../Guide.md)
* Memory deep dive: [Orin Nano Memory Architecture](../Orin-Nano-Memory-Architecture/Guide.md)
* Kernel internals: [Orin Nano Kernel Internals](../Orin-Nano-Kernel-Internals/Guide.md)
