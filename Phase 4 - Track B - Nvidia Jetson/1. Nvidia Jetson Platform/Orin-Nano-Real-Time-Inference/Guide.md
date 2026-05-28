# Jetson Orin Nano 8GB: Real-Time and Deterministic Inference Guide

<div class="course-identity auto-course" style="--course-accent: #ca8a04; --course-accent-rgb: 202, 138, 4;" markdown="1">
<div class="course-identity__icon">JON8</div>
<div markdown="1">
<p class="course-identity__eyebrow">Deep Dive · Jetson Track</p>
<p class="course-identity__title">Specialized course identity for Jetson Orin Nano 8GB: Real-Time and Deterministic Inference Guide.</p>
<p class="course-identity__meta">Artifact: Jetson integration demo · Measure: latency, memory, power, logs</p>
</div>
</div>


## Platform Reference

| Parameter | Value |
|-----------|-------|
| SoC | NVIDIA T234 (Orin) |
| GPU | 1024-core Ampere (SM 8.7) |
| DLA | 1x NVDLA v2.0 |
| CPU | 6-core Arm Cortex-A78AE |
| Memory | 8 GB LPDDR5 (shared CPU/GPU/DLA) |
| JetPack | 6.x (L4T 36.x), CUDA 12.x, TensorRT 10.x |

---

## 1. Introduction

Real-time inference on edge hardware demands a fundamentally different engineering
mindset than throughput-oriented data-center serving. This guide covers every layer
of the stack -- from kernel scheduling to TensorRT engine construction -- required
to deliver deterministic, low-latency AI inference on the Jetson Orin Nano 8GB.

### 1.1 What "Real-Time" Means for Edge AI

A system is real-time when the **correctness of its output depends on when it is
produced**, not only on what it produces. A classification result that arrives after
a robotic arm has already moved is a wrong result, regardless of its accuracy.

There are two categories:

| Property | Hard Real-Time | Soft Real-Time |
|----------|---------------|----------------|
| Deadline miss consequence | System failure / safety hazard | Degraded quality of service |
| Typical domain | Autonomous braking, surgical robots | Video analytics, drone tracking |
| Latency budget | 1-10 ms | 10-100 ms |
| Jitter tolerance | < 100 us | < 5 ms |
| Typical guarantee | Worst-case execution time (WCET) | p99 latency bound |

On the Orin Nano, most practical deployments target **firm real-time**: deadline
misses are tolerated rarely (e.g., < 0.1 % of frames) but each miss is logged and
may trigger fallback behavior.

### 1.2 Latency vs Throughput Tradeoffs

```
Throughput-optimized:           Latency-optimized:

  [B1 B2 B3 B4]  --> GPU          [B1] --> GPU
  large batch, high util          batch=1, lower util
  latency = batch_fill + exec     latency = exec only
```

Key tradeoffs on Orin Nano:

| Strategy | Throughput (FPS) | p99 Latency (ms) | When to use |
|----------|-----------------|-------------------|-------------|
| Batch=8, FP16 | ~120 | ~35 | Offline video |
| Batch=1, FP16 | ~45 | ~8 | Live camera |
| Batch=1, INT8 | ~70 | ~5 | Safety-critical |
| Batch=1, DLA INT8 | ~30 | ~4 (low jitter) | Deterministic path |

### 1.3 End-to-End Latency Anatomy

```
Camera capture  -->  Pre-process  -->  Inference  -->  Post-process  -->  Actuator
    |                    |                |                |                |
  t_cap              t_pre            t_infer          t_post           t_act
  (2-8 ms)          (0.5-2 ms)       (3-15 ms)        (0.3-1 ms)      (0.5-2 ms)

Total end-to-end = t_cap + t_pre + t_infer + t_post + t_act
Typical target:    < 20 ms for 50 Hz control loops
```

The inference time `t_infer` is usually the largest single contributor, but
capture latency and pre-processing are often underestimated.

---

## 2. Jetson Real-Time Capabilities

> **Deep dive:** For kernel-level RT internals — PREEMPT_RT patch architecture, preemption models, interrupt threading, lock primitives under RT, priority inversion and PI mutexes, SCHED_DEADLINE, complete rt-tests suite (cyclictest, hwlatdetect, signaltest, pip_stress), ARM Cortex-A78AE RT specifics, GICv3 tuning, WCET analysis, RT-safe kernel module development, ftrace RT debugging, and production certification — see [**Orin Nano RT Linux Deep Dive**](../Orin-Nano-RT-Linux-Deep-Dive/Guide.md).

### 2.1 PREEMPT_RT Kernel on Jetson

NVIDIA ships a real-time capable kernel starting with JetPack 6.0. The
`PREEMPT_RT` patch converts most interrupt handlers and kernel locks to
preemptible, schedulable entities.

**Checking current kernel preemption model:**

```bash
# Check if PREEMPT_RT is active
uname -a
# Look for "PREEMPT RT" in the output

cat /sys/kernel/realtime
# Returns "1" if PREEMPT_RT is active

# If using stock JetPack without RT:
zcat /proc/config.gz | grep PREEMPT
# CONFIG_PREEMPT_RT=y      <-- full RT
# CONFIG_PREEMPT=y          <-- standard preemption (not RT)
```

**Building the RT kernel from source (JetPack 6.x):**

```bash
# Download L4T kernel source
cd ~/jetson-kernel
tar xf public_sources.tbz2
cd Linux_for_Tegra/source

# Apply PREEMPT_RT patch (included in NVIDIA source)
./generic_rt_build.sh apply

# Configure
export CROSS_COMPILE=aarch64-buildroot-linux-gnu-
export ARCH=arm64
make tegra_defconfig

# Enable RT
scripts/config --enable CONFIG_PREEMPT_RT
scripts/config --disable CONFIG_PREEMPT_VOLUNTARY
scripts/config --enable CONFIG_HIGH_RES_TIMERS
scripts/config --set-val CONFIG_HZ 1000

make -j$(nproc) Image modules dtbs
```

### 2.2 Real-Time Scheduling Classes

Linux provides several scheduling policies. For real-time inference threads:

```
Priority (highest to lowest):

  SCHED_DEADLINE   -- EDF scheduler, specify runtime/deadline/period
  SCHED_FIFO       -- Fixed priority, no time slicing
  SCHED_RR         -- Fixed priority, round-robin time slicing
  SCHED_OTHER      -- CFS (default), not real-time
```

**Setting SCHED_FIFO for an inference thread (C++):**

```cpp
#include <pthread.h>
#include <sched.h>

void set_realtime_priority(pthread_t thread, int priority) {
    struct sched_param param;
    param.sched_priority = priority;  // 1-99, higher = more urgent

    int ret = pthread_setschedparam(thread, SCHED_FIFO, &param);
    if (ret != 0) {
        fprintf(stderr, "Failed to set RT priority: %s\n", strerror(ret));
        fprintf(stderr, "Run with: sudo or set CAP_SYS_NICE\n");
    }
}

// Usage in inference thread setup:
void* inference_thread(void* arg) {
    set_realtime_priority(pthread_self(), 80);
    // ... run inference loop ...
}
```

**Setting SCHED_FIFO from the command line:**

```bash
# Run inference process with FIFO priority 80
sudo chrt -f 80 ./my_inference_app

# Verify scheduling policy of a running process
chrt -p $(pidof my_inference_app)

# Set CPU affinity + RT priority together
sudo taskset -c 4,5 chrt -f 80 ./my_inference_app
```

**Python equivalent using os module:**

```python
import os
import ctypes

# Set SCHED_FIFO priority 80 for current thread
SCHED_FIFO = 1

class SchedParam(ctypes.Structure):
    _fields_ = [("sched_priority", ctypes.c_int)]

libc = ctypes.CDLL("libc.so.6", use_errno=True)

def set_realtime_priority(priority=80):
    param = SchedParam(priority)
    result = libc.sched_setscheduler(0, SCHED_FIFO, ctypes.byref(param))
    if result != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, f"sched_setscheduler failed: {os.strerror(errno)}")
    print(f"Set SCHED_FIFO priority {priority}")
```

### 2.3 Clock Precision and Timekeeping

For latency measurement, clock source matters:

```bash
# Check available clock sources
cat /sys/devices/system/clocksource/clocksource0/available_clocksource
# arch_sys_counter tsc

# Check current clock source
cat /sys/devices/system/clocksource/clocksource0/current_clocksource

# The Arm arch timer provides ~56 ns resolution on Orin Nano
```

**High-resolution timing in C++:**

```cpp
#include <time.h>

// Use CLOCK_MONOTONIC_RAW for jitter-free measurements
// (not adjusted by NTP)
struct timespec get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return ts;
}

double elapsed_ms(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) * 1000.0 +
           (end.tv_nsec - start.tv_nsec) / 1e6;
}

// In inference loop:
auto t0 = get_time();
context->enqueueV3(stream);
cudaStreamSynchronize(stream);
auto t1 = get_time();
printf("Inference: %.3f ms\n", elapsed_ms(t0, t1));
```

**Measuring timer resolution:**

```bash
# cyclictest -- the standard RT latency benchmark
sudo cyclictest -t1 -p 90 -n -i 1000 -l 10000
# T: 0 Min:   3 Act:   5 Avg:   5 Max:  18

# On PREEMPT_RT Orin Nano, expect:
#   Average latency: 5-10 us
#   Max latency: < 50 us (after tuning)
```

---

## 3. TensorRT Engine Optimization

### 3.1 Building Optimized Engines

TensorRT engines are hardware-specific optimized representations of neural
networks. On the Orin Nano, engine building explores the GPU's SM 8.7 capabilities
and available DLA.

**Converting an ONNX model to a TensorRT engine:**

```bash
# Basic FP16 engine build
trtexec \
    --onnx=model.onnx \
    --saveEngine=model_fp16.engine \
    --fp16 \
    --workspace=2048 \
    --minShapes=input:1x3x224x224 \
    --optShapes=input:1x3x224x224 \
    --maxShapes=input:1x3x224x224 \
    --verbose

# INT8 engine with calibration cache
trtexec \
    --onnx=model.onnx \
    --saveEngine=model_int8.engine \
    --int8 \
    --fp16 \
    --calib=calibration_cache.bin \
    --workspace=2048

# DLA engine (Orin Nano has 1 DLA core: dla0)
trtexec \
    --onnx=model.onnx \
    --saveEngine=model_dla.engine \
    --int8 --fp16 \
    --useDLACore=0 \
    --allowGPUFallback
```

### 3.2 Precision Modes

| Precision | Speedup vs FP32 | Accuracy Impact | Memory Reduction | Use Case |
|-----------|-----------------|-----------------|------------------|----------|
| FP32 | 1x (baseline) | None | None | Debugging |
| FP16 | 1.5-2.5x | Negligible | ~50% | Default production |
| INT8 | 2-4x | 0.1-1% mAP drop | ~75% | Latency-critical |
| INT8 on DLA | 1.5-3x | 0.1-1% mAP drop | ~75% + frees GPU | Deterministic path |

### 3.3 Layer Fusion

TensorRT automatically fuses compatible layers to reduce memory bandwidth and
kernel launch overhead. Common fusion patterns on Orin Nano:

```
Before fusion:                    After fusion:
  Conv2D --> BatchNorm --> ReLU     CBR (single kernel)
  Conv2D --> Add --> ReLU           ConvAddReLU
  FC --> ReLU                       FC_ReLU
```

**Inspecting fused layers with trtexec:**

```bash
trtexec --onnx=model.onnx --fp16 --verbose --dumpLayerInfo 2>&1 | grep -i "fused\|fusion"

# Or use the engine inspector in code:
```

```python
import tensorrt as trt

logger = trt.Logger(trt.Logger.VERBOSE)
with open("model_fp16.engine", "rb") as f:
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(f.read())

inspector = engine.create_engine_inspector()
# Print layer-by-layer info showing fusion results
print(inspector.get_engine_information(trt.LayerInformationFormat.ONELINE))
```

### 3.4 INT8 Calibration

INT8 calibration determines per-tensor scale factors by running representative
data through the network. This is critical for accuracy.

**Implementing a calibration class in Python:**

```python
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

class Int8Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, data_loader, cache_file="calibration.cache",
                 batch_size=1):
        super().__init__()
        self.data_loader = data_loader
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.current_index = 0

        # Pre-allocate device buffer for calibration input
        sample = next(iter(data_loader))
        self.device_input = cuda.mem_alloc(sample.nbytes)
        self.batch_allocation = self.device_input

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        try:
            batch = next(self.data_iter)
            cuda.memcpy_htod(self.device_input, batch.ravel())
            return [int(self.device_input)]
        except StopIteration:
            return None

    def read_calibration_cache(self):
        try:
            with open(self.cache_file, "rb") as f:
                return f.read()
        except FileNotFoundError:
            return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

    def reset(self):
        self.data_iter = iter(self.data_loader)


def build_int8_engine(onnx_path, calibrator):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        parser.parse(f.read())

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)  # Allow FP16 fallback
    config.int8_calibrator = calibrator

    serialized = builder.build_serialized_network(network, config)
    return serialized
```

### 3.5 Builder Configuration Best Practices for Orin Nano

```python
import tensorrt as trt

def create_optimized_config(builder):
    config = builder.create_builder_config()

    # Workspace: 2 GB is safe on 8 GB Orin Nano
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)

    # Enable FP16 for all layers that benefit
    config.set_flag(trt.BuilderFlag.FP16)

    # Enable timing cache to speed up subsequent builds
    # (engine building on Orin Nano can take 10-30 minutes)
    try:
        with open("timing.cache", "rb") as f:
            timing_cache = config.create_timing_cache(f.read())
    except FileNotFoundError:
        timing_cache = config.create_timing_cache(b"")
    config.set_timing_cache(timing_cache, ignore_mismatch=False)

    # Prefer precision-reduction for latency
    config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)

    # Limit builder to tactics that run in under 50 ms
    # (avoids slow tactics that hurt latency consistency)
    config.set_flag(trt.BuilderFlag.REJECT_EMPTY_ALGORITHMS)

    return config, timing_cache
```

---

## 4. TensorRT Runtime

### 4.1 Execution Context and CUDA Streams

The execution context holds intermediate activation memory. For real-time
inference, pre-allocate everything before entering the hot loop.

**Complete C++ inference setup:**

```cpp
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <fstream>
#include <vector>
#include <memory>

using namespace nvinfer1;

class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            printf("[TRT] %s\n", msg);
    }
} gLogger;

class RTInferenceEngine {
public:
    bool loadEngine(const std::string& enginePath) {
        // Read serialized engine
        std::ifstream file(enginePath, std::ios::binary);
        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);
        std::vector<char> buffer(size);
        file.read(buffer.data(), size);

        // Deserialize
        auto runtime = std::unique_ptr<IRuntime>(
            createInferRuntime(gLogger));
        m_engine.reset(runtime->deserializeCudaEngine(
            buffer.data(), size));
        if (!m_engine) return false;

        // Create execution context
        m_context.reset(m_engine->createExecutionContext());
        if (!m_context) return false;

        // Create CUDA stream for inference
        cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking);

        // Pre-allocate device buffers
        allocateBuffers();

        return true;
    }

    void allocateBuffers() {
        int numIO = m_engine->getNbIOTensors();
        for (int i = 0; i < numIO; i++) {
            const char* name = m_engine->getIOTensorName(i);
            auto dims = m_engine->getTensorShape(name);
            auto dtype = m_engine->getTensorDataType(name);
            size_t bytes = volume(dims) * dataTypeSize(dtype);

            void* deviceMem;
            cudaMalloc(&deviceMem, bytes);
            m_context->setTensorAddress(name, deviceMem);

            if (m_engine->getTensorIOMode(name) == TensorIOMode::kINPUT) {
                m_inputBuffers[name] = deviceMem;
                m_inputSizes[name] = bytes;
            } else {
                m_outputBuffers[name] = deviceMem;
                m_outputSizes[name] = bytes;
            }
        }
    }

    // Hot path -- zero allocation
    bool infer(const void* inputData, void* outputData) {
        // Copy input H2D
        const char* inputName = m_engine->getIOTensorName(0);
        cudaMemcpyAsync(m_inputBuffers[inputName], inputData,
                        m_inputSizes[inputName],
                        cudaMemcpyHostToDevice, m_stream);

        // Execute
        bool status = m_context->enqueueV3(m_stream);
        if (!status) return false;

        // Copy output D2H
        const char* outputName = m_engine->getIOTensorName(1);
        cudaMemcpyAsync(outputData, m_outputBuffers[outputName],
                        m_outputSizes[outputName],
                        cudaMemcpyDeviceToHost, m_stream);

        cudaStreamSynchronize(m_stream);
        return true;
    }

    void warmup(int iterations = 50) {
        // Allocate dummy input/output on host
        const char* inputName = m_engine->getIOTensorName(0);
        const char* outputName = m_engine->getIOTensorName(1);
        std::vector<uint8_t> dummyIn(m_inputSizes[inputName], 0);
        std::vector<uint8_t> dummyOut(m_outputSizes[outputName], 0);

        for (int i = 0; i < iterations; i++) {
            infer(dummyIn.data(), dummyOut.data());
        }
        printf("Warmup complete: %d iterations\n", iterations);
    }

    ~RTInferenceEngine() {
        cudaStreamDestroy(m_stream);
        for (auto& [name, ptr] : m_inputBuffers) cudaFree(ptr);
        for (auto& [name, ptr] : m_outputBuffers) cudaFree(ptr);
    }

private:
    std::unique_ptr<ICudaEngine> m_engine;
    std::unique_ptr<IExecutionContext> m_context;
    cudaStream_t m_stream;
    std::map<std::string, void*> m_inputBuffers;
    std::map<std::string, void*> m_outputBuffers;
    std::map<std::string, size_t> m_inputSizes;
    std::map<std::string, size_t> m_outputSizes;

    size_t volume(Dims dims) {
        size_t vol = 1;
        for (int i = 0; i < dims.nbDims; i++) vol *= dims.d[i];
        return vol;
    }

    size_t dataTypeSize(DataType dtype) {
        switch (dtype) {
            case DataType::kFLOAT: return 4;
            case DataType::kHALF:  return 2;
            case DataType::kINT8:  return 1;
            case DataType::kINT32: return 4;
            default: return 0;
        }
    }
};
```

### 4.2 Engine Deserialization Optimization

Engine deserialization on Orin Nano can take 500 ms to 3 seconds for large models.
For real-time systems that must start quickly:

```cpp
// Strategy 1: Memory-mapped engine file (avoids copy during load)
#include <sys/mman.h>
#include <fcntl.h>

ICudaEngine* loadEngineMmap(const std::string& path, IRuntime* runtime) {
    int fd = open(path.c_str(), O_RDONLY);
    struct stat st;
    fstat(fd, &st);

    void* mapped = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);

    // Advise kernel to read ahead
    madvise(mapped, st.st_size, MADV_SEQUENTIAL);

    ICudaEngine* engine = runtime->deserializeCudaEngine(mapped, st.st_size);

    munmap(mapped, st.st_size);
    close(fd);
    return engine;
}

// Strategy 2: Pre-load engine into pinned memory
ICudaEngine* loadEnginePinned(const std::string& path, IRuntime* runtime) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    size_t size = file.tellg();
    file.seekg(0);

    void* pinnedMem;
    cudaMallocHost(&pinnedMem, size);
    file.read(static_cast<char*>(pinnedMem), size);

    ICudaEngine* engine = runtime->deserializeCudaEngine(pinnedMem, size);

    cudaFreeHost(pinnedMem);
    return engine;
}
```

### 4.3 Warm-Up Strategies

The first several inferences after engine load exhibit higher latency due to:
- CUDA context initialization
- JIT compilation of any remaining PTX
- GPU clock ramp-up (DVFS)
- Memory page faults on first access

```python
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import time

def warmup_engine(context, stream, d_input, d_output, h_input,
                  iterations=100):
    """
    Run dummy inferences until latency stabilizes.
    Returns the stable latency baseline.
    """
    latencies = []

    for i in range(iterations):
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async_v3(stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(np.empty_like(h_input), d_output, stream)
        stream.synchronize()

    # Measure the last 20 iterations for baseline
    for i in range(20):
        start = time.perf_counter_ns()
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async_v3(stream_handle=stream.handle)
        stream.synchronize()
        end = time.perf_counter_ns()
        latencies.append((end - start) / 1e6)

    avg = np.mean(latencies)
    std = np.std(latencies)
    print(f"Warmup baseline: {avg:.2f} +/- {std:.2f} ms")
    return avg
```

---

## 5. DLA for Deterministic Inference

### 5.1 DLA Fixed-Function Advantage

The Orin Nano contains one NVDLA v2.0 core. DLA is a fixed-function accelerator --
unlike the GPU, it does not share execution resources with other workloads. This
makes DLA inference latency highly predictable.

```
GPU inference jitter:           DLA inference jitter:

  |  *                            |
  | ***  *                        | ****
  |*****  *  *                    |******
  |****** ** **    *              |*******
  +-------------------->          +-------------------->
   4   6   8  10  12 ms            5   5.5  6   6.5 ms

  GPU: mean=6.2, std=1.8 ms      DLA: mean=5.5, std=0.2 ms
```

### 5.2 DLA-Compatible Layers

Not all TensorRT layers run on DLA. Unsupported layers fall back to GPU:

| Layer Type | DLA Support | Notes |
|-----------|-------------|-------|
| Convolution | Yes | 3x3, 5x5, 7x7; groups supported |
| Deconvolution | Yes | Limited kernel sizes |
| Pooling (Max/Avg) | Yes | -- |
| ElementWise | Yes | Add, Sub, Mul, Min, Max |
| Activation (ReLU) | Yes | -- |
| Activation (Sigmoid) | Partial | May fall back |
| BatchNormalization | Yes | Fused with Conv |
| Softmax | No | Falls back to GPU |
| Resize/Upsample | Partial | Nearest-neighbor only |
| Transformer attention | No | Falls back to GPU |

### 5.3 Building a DLA Engine

```python
import tensorrt as trt

def build_dla_engine(onnx_path, output_path, allow_gpu_fallback=True):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            return None

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    # Enable INT8 + FP16 (DLA requires INT8 or FP16)
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)

    # Assign to DLA core 0 (Orin Nano has 1 DLA)
    config.default_device_type = trt.DeviceType.DLA
    config.DLA_core = 0

    if allow_gpu_fallback:
        config.set_flag(trt.BuilderFlag.GPU_FALLBACK)

    serialized = builder.build_serialized_network(network, config)
    if serialized:
        with open(output_path, "wb") as f:
            f.write(serialized)
        print(f"DLA engine saved to {output_path}")
    return serialized
```

### 5.4 Measuring DLA vs GPU Jitter

```bash
# GPU-only engine benchmark
trtexec --loadEngine=model_fp16.engine \
    --iterations=1000 --warmUp=5000 \
    --percentile=50,95,99 \
    --useSpinWait

# DLA engine benchmark
trtexec --loadEngine=model_dla.engine \
    --iterations=1000 --warmUp=5000 \
    --percentile=50,95,99 \
    --useDLACore=0 \
    --useSpinWait
```

Typical results on Orin Nano (ResNet-50 INT8, batch=1):

| Metric | GPU | DLA |
|--------|-----|-----|
| p50 latency | 3.8 ms | 5.1 ms |
| p95 latency | 4.5 ms | 5.3 ms |
| p99 latency | 6.2 ms | 5.4 ms |
| Std deviation | 0.9 ms | 0.15 ms |
| Max latency | 12.1 ms | 5.8 ms |

DLA is slower on average but dramatically more consistent. The p99/p50 ratio
is 1.06 for DLA vs 1.63 for GPU.

---

## 6. Multi-Engine Scheduling

### 6.1 GPU + DLA Concurrent Execution

The Orin Nano can run GPU and DLA inference simultaneously because they are
independent hardware blocks. This enables pipeline parallelism.

```
Timeline:
  GPU:  [---Model A---][---Model A---][---Model A---]
  DLA:       [--Model B--]  [--Model B--]  [--Model B--]

  Total throughput = GPU throughput + DLA throughput
```

**C++ concurrent GPU + DLA execution:**

```cpp
#include <thread>
#include <atomic>

class DualEngineScheduler {
public:
    struct EngineSlot {
        ICudaEngine* engine;
        IExecutionContext* context;
        cudaStream_t stream;
        void* inputBuffer;
        void* outputBuffer;
        size_t inputSize;
        size_t outputSize;
    };

    EngineSlot gpuSlot;   // FP16 engine on GPU
    EngineSlot dlaSlot;   // INT8 engine on DLA

    void init(const std::string& gpuEnginePath,
              const std::string& dlaEnginePath) {
        loadSlot(gpuSlot, gpuEnginePath);
        loadSlot(dlaSlot, dlaEnginePath);
    }

    // Run both engines concurrently on different inputs
    void inferConcurrent(
        const void* gpuInput, void* gpuOutput,
        const void* dlaInput, void* dlaOutput)
    {
        // Launch GPU inference
        cudaMemcpyAsync(gpuSlot.inputBuffer, gpuInput,
                        gpuSlot.inputSize,
                        cudaMemcpyHostToDevice, gpuSlot.stream);
        gpuSlot.context->enqueueV3(gpuSlot.stream);
        cudaMemcpyAsync(gpuOutput, gpuSlot.outputBuffer,
                        gpuSlot.outputSize,
                        cudaMemcpyDeviceToHost, gpuSlot.stream);

        // Launch DLA inference (independent stream)
        cudaMemcpyAsync(dlaSlot.inputBuffer, dlaInput,
                        dlaSlot.inputSize,
                        cudaMemcpyHostToDevice, dlaSlot.stream);
        dlaSlot.context->enqueueV3(dlaSlot.stream);
        cudaMemcpyAsync(dlaOutput, dlaSlot.outputBuffer,
                        dlaSlot.outputSize,
                        cudaMemcpyDeviceToHost, dlaSlot.stream);

        // Wait for both
        cudaStreamSynchronize(gpuSlot.stream);
        cudaStreamSynchronize(dlaSlot.stream);
    }

private:
    void loadSlot(EngineSlot& slot, const std::string& path) {
        // ... load engine, create context, allocate buffers ...
        cudaStreamCreateWithFlags(&slot.stream, cudaStreamNonBlocking);
    }
};
```

### 6.2 Pipeline Parallelism

For multi-stage pipelines (e.g., detection then classification), assign stages
to different accelerators:

```
Frame N:    [GPU: Detect]---->[DLA: Classify]
Frame N+1:       [GPU: Detect]---->[DLA: Classify]
Frame N+2:            [GPU: Detect]---->[DLA: Classify]

Throughput limited by slower stage, latency = sum of stages
```

```python
import threading
import queue
import time

class PipelineScheduler:
    def __init__(self, detector_engine, classifier_engine):
        self.detect_queue = queue.Queue(maxsize=2)
        self.classify_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)

        self.detector = detector_engine    # GPU engine
        self.classifier = classifier_engine  # DLA engine

    def detection_worker(self):
        """Runs on GPU -- detects objects in each frame."""
        while True:
            frame = self.detect_queue.get()
            if frame is None:
                break
            detections = self.detector.infer(frame)
            self.classify_queue.put((frame, detections))

    def classification_worker(self):
        """Runs on DLA -- classifies each detected region."""
        while True:
            item = self.classify_queue.get()
            if item is None:
                break
            frame, detections = item
            results = []
            for det in detections:
                crop = extract_crop(frame, det)
                cls = self.classifier.infer(crop)
                results.append((det, cls))
            self.result_queue.put(results)

    def start(self):
        self.det_thread = threading.Thread(
            target=self.detection_worker, daemon=True)
        self.cls_thread = threading.Thread(
            target=self.classification_worker, daemon=True)
        self.det_thread.start()
        self.cls_thread.start()

    def submit(self, frame):
        self.detect_queue.put(frame)

    def get_result(self, timeout=0.1):
        return self.result_queue.get(timeout=timeout)
```

### 6.3 Engine Assignment Strategies

| Strategy | When to Use | Pros | Cons |
|----------|-------------|------|------|
| GPU-only | Single model, max throughput | Highest raw speed | Jitter under contention |
| DLA-only | Deterministic latency required | Lowest jitter | Slower, limited layers |
| GPU primary + DLA secondary | Two-model pipeline | Parallelism | Complex scheduling |
| DLA primary + GPU fallback | DLA for latency, GPU for unsupported ops | Best of both | Some layers on GPU add jitter |

---

## 7. Latency Optimization Techniques

### 7.1 Batching Strategies for Real-Time

For real-time applications, batch size = 1 is almost always optimal. Larger
batches trade latency for throughput:

```
Batch=1:  [frame] --> [infer 4ms] --> result    total: 4 ms
Batch=4:  [frame][frame][frame][frame] --> [infer 10ms] --> results
          wait 3 frames (60ms @50fps) + 10ms = 70ms per first frame
```

**Dynamic batching with timeout (for variable-rate inputs):**

```python
import time
import threading

class LatencyAwareBatcher:
    def __init__(self, engine, max_batch=4, max_wait_ms=5.0):
        self.engine = engine
        self.max_batch = max_batch
        self.max_wait_s = max_wait_ms / 1000.0
        self.pending = []
        self.lock = threading.Lock()

    def submit(self, input_data):
        """Submit input; returns a Future-like event."""
        event = threading.Event()
        result_holder = [None]

        with self.lock:
            self.pending.append((input_data, event, result_holder))
            if len(self.pending) >= self.max_batch:
                self._flush()

        # If not flushed yet, wait for timeout-based flush
        event.wait(timeout=self.max_wait_s * 2)
        return result_holder[0]

    def _flush(self):
        """Execute batch inference on accumulated inputs."""
        batch = self.pending[:self.max_batch]
        self.pending = self.pending[self.max_batch:]

        inputs = np.stack([item[0] for item in batch])
        outputs = self.engine.infer_batch(inputs)

        for i, (_, event, holder) in enumerate(batch):
            holder[0] = outputs[i]
            event.set()

    def timeout_flusher(self):
        """Background thread that flushes on timeout."""
        while True:
            time.sleep(self.max_wait_s)
            with self.lock:
                if self.pending:
                    self._flush()
```

### 7.2 Input Pipeline Optimization

Pre-processing on CPU is a common bottleneck. Move it to GPU:

```python
import cv2
import numpy as np
import pycuda.driver as cuda

# BAD: CPU preprocessing (5-10 ms on Orin Nano A78AE)
def preprocess_cpu(frame):
    resized = cv2.resize(frame, (224, 224))          # CPU
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)   # CPU
    normalized = resized.astype(np.float32) / 255.0  # CPU
    chw = np.transpose(normalized, (2, 0, 1))        # CPU
    return np.ascontiguousarray(chw)

# GOOD: GPU preprocessing with CUDA (0.3-0.5 ms)
# Use cv2.cuda or custom CUDA kernels (see Section 11)
def preprocess_gpu(frame, d_input, stream):
    gpu_frame = cv2.cuda_GpuMat()
    gpu_frame.upload(frame, stream=stream)
    gpu_resized = cv2.cuda.resize(gpu_frame, (224, 224), stream=stream)
    # Continue with GPU-based normalize + transpose
    # (custom CUDA kernel, see Section 11)
    return gpu_resized
```

### 7.3 End-to-End Latency Measurement

```python
import time
import numpy as np

class LatencyTracker:
    def __init__(self, window_size=1000):
        self.latencies = []
        self.window_size = window_size

    def record(self, latency_ms):
        self.latencies.append(latency_ms)
        if len(self.latencies) > self.window_size:
            self.latencies.pop(0)

    def report(self):
        arr = np.array(self.latencies)
        return {
            "count": len(arr),
            "mean": np.mean(arr),
            "std": np.std(arr),
            "min": np.min(arr),
            "max": np.max(arr),
            "p50": np.percentile(arr, 50),
            "p95": np.percentile(arr, 95),
            "p99": np.percentile(arr, 99),
            "p999": np.percentile(arr, 99.9),
        }

# Usage in inference loop:
tracker = LatencyTracker()
for frame in camera_stream:
    t0 = time.perf_counter_ns()

    preprocessed = preprocess(frame)
    output = engine.infer(preprocessed)
    result = postprocess(output)

    t1 = time.perf_counter_ns()
    tracker.record((t1 - t0) / 1e6)

    if frame_count % 100 == 0:
        stats = tracker.report()
        print(f"p50={stats['p50']:.2f} p99={stats['p99']:.2f} "
              f"max={stats['max']:.2f} ms")
```

---

## 8. Memory Optimization for Inference

### 8.1 Understanding the 8 GB Shared Memory

The Orin Nano's 8 GB LPDDR5 is shared between CPU, GPU, and DLA. Typical
memory budget for a real-time inference application:

```
Total: 8192 MB
  - OS + services:        ~1500 MB
  - CUDA runtime:         ~300 MB
  - TensorRT engine(s):   200-800 MB (depends on model)
  - Workspace:            512-2048 MB
  - Input/output buffers: 10-50 MB
  - Application code:     100-500 MB
  ----------------------------------------
  Available headroom:     ~3000-5000 MB
```

### 8.2 Workspace Size Tuning

TensorRT workspace is scratch memory used during inference for tactics that
need temporary storage. Too small limits optimization; too large wastes memory.

```python
# Query actual workspace usage after build
inspector = engine.create_engine_inspector()
info = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
# Parse JSON to find max workspace per layer

# Recommended approach: start large, then tighten
# Build with 2 GB workspace
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)

# After building, check actual usage via verbose build log:
# [TRT] [V] Total Activation Memory: 12345678 bytes
# Set workspace to 1.5x that value for safety margin
```

### 8.3 Engine Memory Footprint Reduction

```bash
# Check engine file size (approximates device memory footprint)
ls -lh model_fp16.engine
# -rw-r--r-- 1 user user 45M model_fp16.engine

# INT8 engines are smaller
ls -lh model_int8.engine
# -rw-r--r-- 1 user user 24M model_int8.engine
```

**Strategies to reduce engine memory:**

```python
# 1. Use INT8 precision (halves weight memory vs FP16)
config.set_flag(trt.BuilderFlag.INT8)

# 2. Enable weight streaming for very large models
# (streams weights from host memory on demand)
config.set_flag(trt.BuilderFlag.WEIGHT_STREAMING)

# 3. Use strongly-typed network to prevent unnecessary FP32 copies
config.set_flag(trt.BuilderFlag.STRONGLY_TYPED)

# 4. Share execution context memory across engines
#    that do NOT run concurrently
context_a = engine_a.create_execution_context()
context_b = engine_b.create_execution_context()
# Both can share the same device memory if only one runs at a time
```

### 8.4 Avoiding Allocation During Inference

**Critical rule:** Never call `cudaMalloc`, `new`, `malloc`, or any allocating
function in the inference hot path.

```cpp
// BAD: allocating in the hot loop
void infer_bad(const float* input, int size) {
    float* d_input;
    cudaMalloc(&d_input, size);           // ALLOCATION IN HOT PATH
    cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);
    context->enqueueV3(stream);
    cudaFree(d_input);                     // DEALLOCATION IN HOT PATH
}

// GOOD: pre-allocated in constructor, reused every call
class ZeroAllocInference {
    void* d_input_;
    void* d_output_;
    size_t input_size_;
    cudaStream_t stream_;
    IExecutionContext* context_;

public:
    ZeroAllocInference(ICudaEngine* engine) {
        // All allocation happens once at construction
        cudaStreamCreate(&stream_);
        context_ = engine->createExecutionContext();

        const char* inName = engine->getIOTensorName(0);
        const char* outName = engine->getIOTensorName(1);
        input_size_ = /* compute from dims */;

        cudaMalloc(&d_input_, input_size_);
        cudaMalloc(&d_output_, /* output size */);

        context_->setTensorAddress(inName, d_input_);
        context_->setTensorAddress(outName, d_output_);
    }

    // Hot path: zero allocations
    void infer(const void* input, void* output) {
        cudaMemcpyAsync(d_input_, input, input_size_,
                        cudaMemcpyHostToDevice, stream_);
        context_->enqueueV3(stream_);
        cudaMemcpyAsync(output, d_output_, /* size */,
                        cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);
    }
};
```

### 8.5 Memory Pools

Use CUDA memory pools to avoid repeated allocation/deallocation overhead:

```cpp
// Create a memory pool for inference buffers
cudaMemPool_t memPool;
cudaDeviceGetDefaultMemPool(&memPool, 0);

// Set pool to retain allocations (avoid returning memory to OS)
uint64_t threshold = UINT64_MAX;  // Never release
cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrReleaseThreshold,
                        &threshold);

// Allocations from pool are fast (no syscall)
void* ptr;
cudaMallocAsync(&ptr, size, stream);
// ... use ptr ...
cudaFreeAsync(ptr, stream);  // Returns to pool, not OS
```

---

## 9. CPU Real-Time Configuration

### 9.1 CPU Isolation (isolcpus)

The Orin Nano has 6 Cortex-A78AE cores (cores 0-5). Isolate cores for
dedicated inference threads:

```bash
# Edit /boot/extlinux/extlinux.conf on Orin Nano
# Add to APPEND line:
#   isolcpus=4,5 nohz_full=4,5 rcu_nocbs=4,5

# Example extlinux.conf entry:
# APPEND ${cbootargs} root=/dev/mmcblk0p1 rw rootwait
#   isolcpus=4,5 nohz_full=4,5 rcu_nocbs=4,5
#   irqaffinity=0,1,2,3

# After reboot, verify isolation:
cat /sys/devices/system/cpu/isolated
# 4,5

# Isolated cores will not run any tasks unless explicitly pinned
taskset -p 1  # PID 1 (init) should show mask without cores 4,5
```

### 9.2 Thread Affinity

Pin inference threads to isolated cores:

```cpp
#include <pthread.h>

void pin_to_core(pthread_t thread, int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);

    int ret = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
    if (ret != 0) {
        fprintf(stderr, "Failed to set CPU affinity: %s\n", strerror(ret));
    }
}

// Pin inference thread to core 4, post-processing to core 5
void* inference_thread(void* arg) {
    pin_to_core(pthread_self(), 4);
    set_realtime_priority(pthread_self(), 80);
    // ... inference loop ...
}

void* postprocess_thread(void* arg) {
    pin_to_core(pthread_self(), 5);
    set_realtime_priority(pthread_self(), 70);
    // ... post-processing loop ...
}
```

**Python equivalent:**

```python
import os

def pin_to_cores(cores):
    """Pin current process to specific CPU cores."""
    os.sched_setaffinity(0, cores)
    actual = os.sched_getaffinity(0)
    print(f"Pinned to cores: {actual}")

# Pin inference process to isolated cores
pin_to_cores({4, 5})
```

### 9.3 IRQ Affinity

Move hardware interrupts away from inference cores:

```bash
# List current IRQ affinity
for irq in /proc/irq/*/smp_affinity_list; do
    echo "$irq: $(cat $irq)"
done

# Move all IRQs to cores 0-3 (away from inference cores 4-5)
for irq in $(ls /proc/irq/ | grep -E '^[0-9]+$'); do
    echo 0-3 > /proc/irq/$irq/smp_affinity_list 2>/dev/null
done

# Verify GPU interrupt is on non-inference core
cat /proc/interrupts | grep gpu
# The GPU interrupt should fire on core 0-3
```

### 9.4 Reducing Jitter Sources

```bash
# 1. Disable CPU frequency scaling (lock to max)
echo performance > /sys/devices/system/cpu/cpu4/cpufreq/scaling_governor
echo performance > /sys/devices/system/cpu/cpu5/cpufreq/scaling_governor

# Or use jetson_clocks to lock all clocks
sudo jetson_clocks

# 2. Disable kernel tick on isolated cores (tickless)
# (Already enabled via nohz_full=4,5 boot parameter)

# 3. Minimize kernel background work on isolated cores
echo 1 > /sys/bus/workqueue/devices/writeback/cpumask  # cores 0 only

# 4. Disable transparent huge pages (THP causes latency spikes)
echo never > /sys/kernel/mm/transparent_hugepage/enabled

# 5. Lock memory to prevent page faults in RT threads
```

```cpp
#include <sys/mman.h>

// Lock all current and future memory to prevent page faults
void lock_memory() {
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
        perror("mlockall failed");
    }
}
```

### 9.5 Complete CPU RT Setup Script

```bash
#!/bin/bash
# rt_setup.sh -- Run once after boot on Orin Nano
set -e

echo "=== Orin Nano Real-Time Setup ==="

# Lock CPU frequency to maximum
sudo jetson_clocks

# Verify RT kernel
if [ "$(cat /sys/kernel/realtime 2>/dev/null)" = "1" ]; then
    echo "[OK] PREEMPT_RT kernel active"
else
    echo "[WARN] Not running PREEMPT_RT kernel"
fi

# Move IRQs away from isolated cores
ISOLATED=$(cat /sys/devices/system/cpu/isolated)
echo "Isolated cores: $ISOLATED"
for irq in $(ls /proc/irq/ | grep -E '^[0-9]+$'); do
    echo 0-3 > /proc/irq/$irq/smp_affinity_list 2>/dev/null || true
done
echo "[OK] IRQs moved to cores 0-3"

# Disable THP
echo never | sudo tee /sys/kernel/mm/transparent_hugepage/enabled > /dev/null
echo "[OK] Transparent huge pages disabled"

# Set GPU and memory clocks to maximum
sudo bash -c 'echo 1 > /sys/devices/gpu.0/force_idle'
sleep 0.1
sudo bash -c 'echo 0 > /sys/devices/gpu.0/force_idle'

echo "=== RT setup complete ==="
```

---

## 10. CUDA Real-Time Patterns

### 10.1 Persistent Kernels

Standard CUDA kernels launch, execute, and exit. Persistent kernels stay
resident on the GPU, polling for work. This eliminates kernel launch latency.

```cpp
__global__ void persistent_inference_kernel(
    volatile int* work_ready,
    volatile int* work_done,
    float* input_buffer,
    float* output_buffer,
    int input_size)
{
    // Each block stays resident and polls for work
    while (true) {
        // Spin-wait for work (low latency)
        while (atomicAdd((int*)work_ready, 0) == 0) {
            __nanosleep(100);  // Brief pause to save power
        }

        // Process input
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < input_size) {
            output_buffer[idx] = some_computation(input_buffer[idx]);
        }
        __threadfence();

        // Signal completion
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            atomicExch((int*)work_done, 1);
            atomicExch((int*)work_ready, 0);
        }
        __syncthreads();
    }
}
```

Note: Persistent kernels are an advanced pattern. For most TensorRT-based
inference, CUDA Graphs (Section 10.2) are the recommended approach.

### 10.2 CUDA Graphs for Reduced Launch Overhead

CUDA Graphs capture a sequence of operations (memcpy, kernel launches) into a
single launchable unit. This eliminates per-operation CPU overhead.

**Capturing TensorRT inference as a CUDA Graph:**

```cpp
#include <cuda_runtime.h>

class GraphInference {
    cudaGraph_t graph_;
    cudaGraphExec_t graphExec_;
    cudaStream_t captureStream_;
    cudaStream_t execStream_;
    IExecutionContext* context_;
    void* d_input_;
    void* d_output_;
    size_t inputSize_;
    size_t outputSize_;

public:
    void captureGraph(const void* sampleInput) {
        cudaStreamCreate(&captureStream_);
        cudaStreamCreate(&execStream_);

        // Run a few warmup iterations first (required before capture)
        for (int i = 0; i < 10; i++) {
            cudaMemcpyAsync(d_input_, sampleInput, inputSize_,
                            cudaMemcpyHostToDevice, captureStream_);
            context_->enqueueV3(captureStream_);
            cudaStreamSynchronize(captureStream_);
        }

        // Begin graph capture
        cudaStreamBeginCapture(captureStream_,
                               cudaStreamCaptureModeGlobal);

        // Record the operations we want to capture
        cudaMemcpyAsync(d_input_, sampleInput, inputSize_,
                        cudaMemcpyHostToDevice, captureStream_);
        context_->enqueueV3(captureStream_);

        // End capture
        cudaStreamEndCapture(captureStream_, &graph_);

        // Instantiate executable graph
        cudaGraphInstantiate(&graphExec_, graph_, nullptr, nullptr, 0);

        printf("CUDA Graph captured successfully\n");
    }

    // Ultra-low-overhead inference: single API call
    void infer(const void* input, void* output) {
        // Update input data (graph uses same device pointers)
        cudaMemcpy(d_input_, input, inputSize_, cudaMemcpyHostToDevice);

        // Launch entire graph
        cudaGraphLaunch(graphExec_, execStream_);
        cudaStreamSynchronize(execStream_);

        // Copy output
        cudaMemcpy(output, d_output_, outputSize_, cudaMemcpyDeviceToHost);
    }
};
```

**Python CUDA Graph capture with TensorRT:**

```python
import tensorrt as trt
import pycuda.driver as cuda
import numpy as np

class CudaGraphInference:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.graph = None
        self.graph_exec = None

        # Allocate buffers
        self._alloc_buffers()

    def _alloc_buffers(self):
        self.d_input = cuda.mem_alloc(self.input_size)
        self.d_output = cuda.mem_alloc(self.output_size)
        self.h_output = np.empty(self.output_shape, dtype=np.float32)
        self.context.set_tensor_address("input", int(self.d_input))
        self.context.set_tensor_address("output", int(self.d_output))

    def capture_graph(self, sample_input):
        """Capture inference as CUDA graph."""
        # Warmup
        for _ in range(10):
            cuda.memcpy_htod_async(self.d_input, sample_input, self.stream)
            self.context.execute_async_v3(self.stream.handle)
            self.stream.synchronize()

        # Capture
        self.stream.begin_capture()
        cuda.memcpy_htod_async(self.d_input, sample_input, self.stream)
        self.context.execute_async_v3(self.stream.handle)
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.graph = self.stream.end_capture()
        self.graph_exec = self.graph.instantiate()
        print("CUDA Graph captured")

    def infer(self, input_data):
        """Launch pre-captured graph."""
        cuda.memcpy_htod(self.d_input, input_data)
        self.graph_exec.launch(self.stream)
        self.stream.synchronize()
        return self.h_output.copy()
```

### 10.3 Pre-Allocated Buffers and Pinned Memory

Pinned (page-locked) memory enables faster H2D/D2H transfers and is required
for asynchronous memcpy:

```cpp
// Allocate pinned host memory for input/output
float* h_input;
float* h_output;
cudaMallocHost(&h_input, input_size);    // Pinned
cudaMallocHost(&h_output, output_size);  // Pinned

// Transfer is ~2x faster with pinned memory on Orin Nano
// Pinned: ~12 GB/s effective bandwidth
// Pageable: ~6 GB/s effective bandwidth

// IMPORTANT: Don't over-allocate pinned memory
// It reduces available memory for the OS page cache
// Rule of thumb: < 500 MB pinned on 8 GB Orin Nano
```

### 10.4 Stream Callbacks for Async Processing

```cpp
// Callback triggered when inference completes
void CUDART_CB inferenceComplete(cudaStream_t stream,
                                  cudaError_t status,
                                  void* userData) {
    auto* result = static_cast<InferenceResult*>(userData);
    result->timestamp = std::chrono::steady_clock::now();
    result->ready.store(true, std::memory_order_release);
    // Post-processing can start immediately
}

// Register callback after inference
cudaMemcpyAsync(d_input, h_input, size, cudaMemcpyHostToDevice, stream);
context->enqueueV3(stream);
cudaMemcpyAsync(h_output, d_output, size, cudaMemcpyDeviceToHost, stream);
cudaLaunchHostFunc(stream, inferenceComplete, &result);
// CPU thread can do other work while GPU executes
```

---

## 11. Pre/Post Processing Pipeline

### 11.1 GPU-Accelerated Image Preprocessing

A custom CUDA kernel for the common resize+normalize+HWC-to-CHW pipeline:

```cpp
// preprocess.cu
__global__ void preprocess_kernel(
    const uint8_t* __restrict__ input,    // HWC, BGR, uint8
    float* __restrict__ output,           // CHW, RGB, float32
    int src_w, int src_h,
    int dst_w, int dst_h,
    float mean_r, float mean_g, float mean_b,
    float std_r, float std_g, float std_b)
{
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;

    if (dx >= dst_w || dy >= dst_h) return;

    // Bilinear interpolation coordinates
    float sx = (dx + 0.5f) * src_w / dst_w - 0.5f;
    float sy = (dy + 0.5f) * src_h / dst_h - 0.5f;

    int x0 = (int)floorf(sx), y0 = (int)floorf(sy);
    int x1 = x0 + 1, y1 = y0 + 1;
    float fx = sx - x0, fy = sy - y0;

    x0 = max(0, min(x0, src_w - 1));
    x1 = max(0, min(x1, src_w - 1));
    y0 = max(0, min(y0, src_h - 1));
    y1 = max(0, min(y1, src_h - 1));

    // Interpolate each channel (input is BGR)
    for (int c = 0; c < 3; c++) {
        float v00 = input[(y0 * src_w + x0) * 3 + c];
        float v01 = input[(y0 * src_w + x1) * 3 + c];
        float v10 = input[(y1 * src_w + x0) * 3 + c];
        float v11 = input[(y1 * src_w + x1) * 3 + c];

        float val = (1 - fx) * (1 - fy) * v00 +
                    fx * (1 - fy) * v01 +
                    (1 - fx) * fy * v10 +
                    fx * fy * v11;

        // BGR to RGB: channel 0->2, 1->1, 2->0
        int out_c = 2 - c;

        // Normalize
        float mean = (out_c == 0) ? mean_r : (out_c == 1) ? mean_g : mean_b;
        float std  = (out_c == 0) ? std_r  : (out_c == 1) ? std_g  : std_b;
        val = (val / 255.0f - mean) / std;

        // Write CHW format
        output[out_c * dst_h * dst_w + dy * dst_w + dx] = val;
    }
}

void launchPreprocess(
    const uint8_t* d_input, float* d_output,
    int src_w, int src_h, int dst_w, int dst_h,
    cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((dst_w + 15) / 16, (dst_h + 15) / 16);

    // ImageNet normalization values
    preprocess_kernel<<<grid, block, 0, stream>>>(
        d_input, d_output, src_w, src_h, dst_w, dst_h,
        0.485f, 0.456f, 0.406f,   // mean RGB
        0.229f, 0.224f, 0.225f);  // std RGB
}
```

### 11.2 NMS (Non-Maximum Suppression) on GPU

For object detection post-processing, NMS on CPU can take 1-5 ms.
A GPU-accelerated version:

```cpp
// Simplified GPU NMS for YOLO-style detections
__global__ void nms_kernel(
    const float* boxes,     // [N, 4] x1,y1,x2,y2
    const float* scores,    // [N]
    int* keep,              // [N] output mask
    int num_boxes,
    float iou_threshold)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_boxes) return;

    if (scores[i] < 0.01f) {
        keep[i] = 0;
        return;
    }

    keep[i] = 1;  // Assume keeping

    for (int j = 0; j < i; j++) {
        if (keep[j] == 0) continue;
        if (scores[j] <= scores[i]) continue;

        // Compute IoU
        float x1 = fmaxf(boxes[i*4+0], boxes[j*4+0]);
        float y1 = fmaxf(boxes[i*4+1], boxes[j*4+1]);
        float x2 = fminf(boxes[i*4+2], boxes[j*4+2]);
        float y2 = fminf(boxes[i*4+3], boxes[j*4+3]);

        float intersection = fmaxf(0, x2-x1) * fmaxf(0, y2-y1);
        float area_i = (boxes[i*4+2]-boxes[i*4+0]) *
                        (boxes[i*4+3]-boxes[i*4+1]);
        float area_j = (boxes[j*4+2]-boxes[j*4+0]) *
                        (boxes[j*4+3]-boxes[j*4+1]);
        float iou = intersection / (area_i + area_j - intersection);

        if (iou > iou_threshold) {
            keep[i] = 0;
            return;
        }
    }
}
```

### 11.3 Complete Pre/Post Pipeline Timing

```
Typical pipeline timing on Orin Nano (1080p camera, YOLOv8s):

  CPU preprocess:   6.2 ms   |  GPU preprocess:  0.4 ms
  H2D copy:         0.8 ms   |  (already on GPU)  0.0 ms
  Inference (FP16): 7.5 ms   |  Inference (FP16): 7.5 ms
  D2H copy:         0.3 ms   |  D2H copy:         0.3 ms
  CPU NMS:          2.1 ms   |  GPU NMS:          0.2 ms
  -----------------------------|-------------------------------
  Total:           16.9 ms   |  Total:            8.4 ms
                               |  Savings:          8.5 ms (50%)
```

---

## 12. Multi-Model Inference

### 12.1 Running Multiple Models Concurrently

On the Orin Nano, you can run up to 3 inference tasks simultaneously:
- 1 on GPU (via TensorRT)
- 1 on DLA (via TensorRT)
- 1 on CPU (lightweight model via ONNX Runtime or PyTorch)

```python
import threading
import tensorrt as trt
import numpy as np

class MultiModelManager:
    def __init__(self):
        self.engines = {}
        self.contexts = {}
        self.streams = {}
        self.lock = threading.Lock()

    def load_model(self, name, engine_path, device="gpu"):
        """Load a TensorRT engine."""
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())

        context = engine.create_execution_context()
        stream = cuda.Stream()

        self.engines[name] = engine
        self.contexts[name] = context
        self.streams[name] = stream

    def infer(self, name, input_data):
        """Run inference on a specific model."""
        context = self.contexts[name]
        stream = self.streams[name]

        # Each model has its own stream -- can run concurrently
        cuda.memcpy_htod_async(self.d_inputs[name], input_data, stream)
        context.execute_async_v3(stream.handle)
        cuda.memcpy_dtoh_async(self.h_outputs[name],
                               self.d_outputs[name], stream)
        stream.synchronize()
        return self.h_outputs[name].copy()

    def infer_concurrent(self, tasks):
        """
        Run multiple models concurrently.
        tasks: list of (model_name, input_data) tuples
        """
        threads = []
        results = {}

        def worker(name, data):
            results[name] = self.infer(name, data)

        for name, data in tasks:
            t = threading.Thread(target=worker, args=(name, data))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        return results
```

### 12.2 Resource Partitioning

```
Orin Nano resource partitioning for 3-model pipeline:

  Model A (Detection):     GPU, FP16, Priority HIGH
  Model B (Segmentation):  DLA, INT8, Priority MEDIUM
  Model C (Classification): GPU, INT8, Priority LOW

  GPU timeline:
    [----Model A (high)----]  [--C (low)--]  [----Model A----] ...

  DLA timeline:
    [------Model B------]  [------Model B------] ...
```

### 12.3 Priority-Based Scheduling

```cpp
#include <queue>
#include <mutex>
#include <condition_variable>

struct InferenceRequest {
    int priority;        // Higher = more urgent
    std::string model;
    void* input;
    void* output;
    std::function<void()> callback;

    bool operator<(const InferenceRequest& other) const {
        return priority < other.priority;  // Max-heap
    }
};

class PriorityInferenceScheduler {
    std::priority_queue<InferenceRequest> queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::map<std::string, RTInferenceEngine*> engines_;
    bool running_ = true;

public:
    void submit(InferenceRequest req) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push(std::move(req));
        }
        cv_.notify_one();
    }

    void worker() {
        while (running_) {
            InferenceRequest req;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [&] {
                    return !queue_.empty() || !running_;
                });
                if (!running_) break;
                req = queue_.top();
                queue_.pop();
            }

            auto* engine = engines_[req.model];
            engine->infer(req.input, req.output);
            if (req.callback) req.callback();
        }
    }
};
```

---

## 13. Inference Serving

### 13.1 Triton Inference Server on Jetson

NVIDIA provides Triton Inference Server containers for Jetson (JetPack 6.x).

**Installation:**

```bash
# Pull the Jetson-optimized Triton container
# (Uses L4T base, includes TensorRT backend)
sudo docker pull nvcr.io/nvidia/tritonserver:24.08-jetpack-py3

# Create model repository structure
mkdir -p ~/triton-models/yolov8/1
cp model.engine ~/triton-models/yolov8/1/model.plan

# Create model configuration
cat > ~/triton-models/yolov8/config.pbtxt << 'PBTXT'
name: "yolov8"
platform: "tensorrt_plan"
max_batch_size: 4
input [
  {
    name: "images"
    data_type: TYPE_FP32
    dims: [ 3, 640, 640 ]
  }
]
output [
  {
    name: "output0"
    data_type: TYPE_FP32
    dims: [ 84, 8400 ]
  }
]
instance_group [
  {
    count: 1
    kind: KIND_GPU
  }
]
dynamic_batching {
  preferred_batch_size: [ 1, 2, 4 ]
  max_queue_delay_microseconds: 5000
}
PBTXT
```

**Running Triton on Orin Nano:**

```bash
sudo docker run --rm --runtime=nvidia \
    --network=host \
    -v ~/triton-models:/models \
    nvcr.io/nvidia/tritonserver:24.08-jetpack-py3 \
    tritonserver \
        --model-repository=/models \
        --strict-model-config=false \
        --min-supported-compute-capability=8.7 \
        --log-verbose=0 \
        --exit-on-error=false
```

### 13.2 Client Inference Request

```python
import tritonclient.grpc as grpcclient
import numpy as np

def triton_infer(image_data):
    client = grpcclient.InferenceServerClient(url="localhost:8001")

    # Prepare input
    input_tensor = grpcclient.InferInput("images", [1, 3, 640, 640], "FP32")
    input_tensor.set_data_from_numpy(image_data)

    # Prepare output
    output = grpcclient.InferRequestedOutput("output0")

    # Inference with deadline
    result = client.infer(
        model_name="yolov8",
        inputs=[input_tensor],
        outputs=[output],
        client_timeout=50.0,   # 50 ms timeout
    )

    return result.as_numpy("output0")
```

### 13.3 Triton Performance Tuning for Real-Time

```bash
# Key Triton settings for low-latency on Orin Nano:

# 1. Disable dynamic batching for lowest latency
#    (Remove dynamic_batching section from config.pbtxt)

# 2. Use gRPC with shared memory for zero-copy
#    (Avoids serialization overhead)

# 3. Pin Triton to specific cores
sudo taskset -c 0,1,2,3 docker run ...

# 4. Benchmark with perf_analyzer
perf_analyzer \
    -m yolov8 \
    -u localhost:8001 \
    -i grpc \
    --concurrency-range 1:1 \
    --measurement-interval 10000 \
    --percentile=99 \
    -b 1

# Expected output on Orin Nano (YOLOv8s FP16):
# Concurrency: 1, throughput: 42 infer/sec
# p50 latency: 7.8 ms, p99 latency: 12.3 ms
```

---

## 14. Benchmarking and Profiling

### 14.1 trtexec Usage

`trtexec` is the primary benchmarking tool for TensorRT engines.

```bash
# Basic latency benchmark
trtexec --loadEngine=model_fp16.engine \
    --iterations=1000 \
    --warmUp=5000 \
    --duration=0 \
    --useSpinWait \
    --percentile=50,95,99

# Key flags for real-time benchmarking:
#   --useSpinWait       Busy-wait instead of sleep (lower jitter)
#   --noDataTransfers   Measure compute only (no H2D/D2H)
#   --streams=1         Single stream (matches RT deployment)
#   --exposeDMA         Show H2D/D2H timing separately
#   --dumpProfile       Per-layer timing breakdown

# Compare GPU vs DLA
trtexec --loadEngine=model_gpu_int8.engine --useSpinWait --percentile=50,95,99
trtexec --loadEngine=model_dla_int8.engine --useDLACore=0 --useSpinWait \
    --percentile=50,95,99

# Profile with layer-level timing
trtexec --loadEngine=model_fp16.engine \
    --dumpProfile --separateProfileRun \
    --iterations=100
```

### 14.2 End-to-End Latency Measurement Framework

```python
import time
import json
import numpy as np
from dataclasses import dataclass, asdict
from typing import List

@dataclass
class LatencyBreakdown:
    preprocess_ms: float
    h2d_ms: float
    inference_ms: float
    d2h_ms: float
    postprocess_ms: float
    total_ms: float

class E2EBenchmark:
    def __init__(self):
        self.records: List[LatencyBreakdown] = []

    def measure(self, preprocess_fn, infer_fn, postprocess_fn, input_data):
        """Measure each stage of the pipeline."""
        # Preprocess
        t0 = time.perf_counter_ns()
        preprocessed = preprocess_fn(input_data)
        t1 = time.perf_counter_ns()

        # H2D + Inference + D2H (measured inside infer_fn)
        h2d_start = time.perf_counter_ns()
        cuda.memcpy_htod(d_input, preprocessed)
        h2d_end = time.perf_counter_ns()

        infer_start = time.perf_counter_ns()
        context.execute_async_v3(stream.handle)
        stream.synchronize()
        infer_end = time.perf_counter_ns()

        d2h_start = time.perf_counter_ns()
        cuda.memcpy_dtoh(h_output, d_output)
        d2h_end = time.perf_counter_ns()

        # Postprocess
        t4 = time.perf_counter_ns()
        result = postprocess_fn(h_output)
        t5 = time.perf_counter_ns()

        record = LatencyBreakdown(
            preprocess_ms=(t1 - t0) / 1e6,
            h2d_ms=(h2d_end - h2d_start) / 1e6,
            inference_ms=(infer_end - infer_start) / 1e6,
            d2h_ms=(d2h_end - d2h_start) / 1e6,
            postprocess_ms=(t5 - t4) / 1e6,
            total_ms=(t5 - t0) / 1e6,
        )
        self.records.append(record)
        return result

    def report(self):
        """Generate percentile report for each stage."""
        fields = ["preprocess_ms", "h2d_ms", "inference_ms",
                  "d2h_ms", "postprocess_ms", "total_ms"]
        report = {}
        for field in fields:
            values = [getattr(r, field) for r in self.records]
            arr = np.array(values)
            report[field] = {
                "mean": f"{np.mean(arr):.2f}",
                "p50":  f"{np.percentile(arr, 50):.2f}",
                "p95":  f"{np.percentile(arr, 95):.2f}",
                "p99":  f"{np.percentile(arr, 99):.2f}",
                "max":  f"{np.max(arr):.2f}",
            }
        return report

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.report(), f, indent=2)
```

### 14.3 Nsight Systems Profiling

```bash
# Profile inference application with nsys
nsys profile \
    --trace=cuda,nvtx,osrt \
    --output=inference_profile \
    --duration=10 \
    --sample=cpu \
    ./my_inference_app

# Generate summary report
nsys stats inference_profile.nsys-rep

# Key things to look for in the profile:
# 1. Gap between kernel launches (CPU overhead)
# 2. Time spent in cudaMemcpy (data transfer bottleneck)
# 3. Long gaps between inference iterations (scheduling delay)
# 4. Unexpected CPU activity during GPU execution

# Add NVTX markers in code for fine-grained profiling:
```

```cpp
#include <nvtx3/nvToolsExt.h>

void inference_loop() {
    while (running) {
        nvtxRangePush("Frame");

        nvtxRangePush("Preprocess");
        preprocess(frame);
        nvtxRangePop();

        nvtxRangePush("H2D");
        cudaMemcpyAsync(d_in, h_in, size, cudaMemcpyHostToDevice, stream);
        nvtxRangePop();

        nvtxRangePush("Inference");
        context->enqueueV3(stream);
        nvtxRangePop();

        nvtxRangePush("D2H");
        cudaMemcpyAsync(h_out, d_out, size, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        nvtxRangePop();

        nvtxRangePush("Postprocess");
        postprocess(h_out);
        nvtxRangePop();

        nvtxRangePop();  // Frame
    }
}
```

### 14.4 Interpreting p50/p95/p99 Latency

```
Latency distribution for GPU FP16 inference (Orin Nano, 10000 samples):

Count
  |
  |    *
  |   ***
  |  *****
  | *******
  |*********                *     *
  +---------------------------------->  Latency (ms)
  3    5    7    9    11   13   15

  p50 =  5.2 ms   (half of inferences faster than this)
  p95 =  7.8 ms   (only 5% slower than this)
  p99 = 11.3 ms   (only 1% slower -- tail latency)
  max = 15.1 ms   (worst case observed)

  For a 20 ms deadline:
    p50 meets deadline:  YES
    p99 meets deadline:  YES
    max meets deadline:  YES
    --> Safe for soft real-time at 50 Hz

  For a 10 ms deadline:
    p50 meets deadline:  YES
    p99 meets deadline:  NO (11.3 > 10)
    --> Need INT8 or DLA to meet this deadline at p99
```

---

## 15. Real-Time Video Inference

### 15.1 Frame-Accurate Processing

For real-time video, every frame has a timestamp. The system must process
frames within their validity window.

```python
import cv2
import time
import threading
from collections import deque

class RTVideoProcessor:
    def __init__(self, camera_id=0, target_fps=30):
        self.cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FPS, target_fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffering

        self.target_fps = target_fps
        self.frame_deadline_ms = 1000.0 / target_fps
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.stats = {"processed": 0, "dropped": 0, "deadline_miss": 0}

    def capture_thread(self):
        """Continuously grab latest frame (drop old ones)."""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            with self.frame_lock:
                self.latest_frame = (frame, time.perf_counter_ns())

    def process_loop(self, engine):
        """Process frames with deadline awareness."""
        self.running = True
        cap_thread = threading.Thread(target=self.capture_thread,
                                      daemon=True)
        cap_thread.start()

        while self.running:
            # Grab latest frame (skip stale frames)
            with self.frame_lock:
                if self.latest_frame is None:
                    continue
                frame, capture_time = self.latest_frame
                self.latest_frame = None

            # Check if frame is already stale
            age_ms = (time.perf_counter_ns() - capture_time) / 1e6
            if age_ms > self.frame_deadline_ms:
                self.stats["dropped"] += 1
                continue

            # Process
            t0 = time.perf_counter_ns()
            preprocessed = preprocess(frame)
            result = engine.infer(preprocessed)
            detections = postprocess(result)
            t1 = time.perf_counter_ns()

            latency_ms = (t1 - t0) / 1e6
            total_age_ms = (t1 - capture_time) / 1e6

            if total_age_ms > self.frame_deadline_ms * 2:
                self.stats["deadline_miss"] += 1

            self.stats["processed"] += 1

            # Report periodically
            if self.stats["processed"] % 100 == 0:
                total = (self.stats["processed"] +
                         self.stats["dropped"])
                drop_rate = self.stats["dropped"] / max(total, 1) * 100
                miss_rate = (self.stats["deadline_miss"] /
                             max(self.stats["processed"], 1) * 100)
                print(f"Processed: {self.stats['processed']}, "
                      f"Dropped: {drop_rate:.1f}%, "
                      f"Deadline miss: {miss_rate:.1f}%, "
                      f"Latency: {latency_ms:.1f} ms")
```

### 15.2 GStreamer Pipeline Integration

For production video pipelines on Jetson, GStreamer with NVIDIA plugins
provides hardware-accelerated capture and decode:

```bash
# GStreamer pipeline: CSI camera -> hardware ISP -> inference
# Use nvarguscamerasrc for CSI cameras on Orin Nano

gst-launch-1.0 \
    nvarguscamerasrc sensor-id=0 ! \
    'video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1' ! \
    nvvidconv ! \
    'video/x-raw(memory:NVMM),width=640,height=640,format=RGBA' ! \
    queue max-size-buffers=1 leaky=downstream ! \
    nvvidconv ! \
    'video/x-raw,width=640,height=640,format=RGBA' ! \
    appsink max-buffers=1 drop=true
```

```python
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

class GstInferencePipeline:
    def __init__(self, engine):
        Gst.init(None)
        self.engine = engine

        pipeline_str = (
            "nvarguscamerasrc sensor-id=0 ! "
            "video/x-raw(memory:NVMM),width=1920,height=1080,"
            "framerate=30/1 ! "
            "nvvidconv ! "
            "video/x-raw,width=640,height=640,format=BGR ! "
            "queue max-size-buffers=1 leaky=downstream ! "
            "appsink name=sink emit-signals=true max-buffers=1 drop=true"
        )
        self.pipeline = Gst.parse_launch(pipeline_str)
        sink = self.pipeline.get_by_name("sink")
        sink.connect("new-sample", self.on_new_sample)

    def on_new_sample(self, sink):
        sample = sink.emit("pull-sample")
        buf = sample.get_buffer()
        caps = sample.get_caps()

        # Extract frame data
        success, map_info = buf.map(Gst.MapFlags.READ)
        if success:
            frame_data = np.frombuffer(map_info.data, dtype=np.uint8)
            frame = frame_data.reshape((640, 640, 3))

            # Run inference
            result = self.engine.infer(preprocess(frame))
            detections = postprocess(result)

            buf.unmap(map_info)
        return Gst.FlowReturn.OK

    def start(self):
        self.pipeline.set_state(Gst.State.PLAYING)

    def stop(self):
        self.pipeline.set_state(Gst.State.NULL)
```

### 15.3 Pipeline Buffering Strategies

```
Strategy 1: Drop-oldest (for latest-frame applications)
  Camera: [F1][F2][F3][F4][F5]...
  Buffer: [F5]  (always keep only latest)
  Inference gets the most recent frame

Strategy 2: Drop-newest (for sequential processing)
  Camera: [F1][F2][F3][F4][F5]...
  Buffer: [F1][F2][F3]  (process in order, drop if full)
  Inference processes frames sequentially

Strategy 3: Skip-N (for fixed-rate processing)
  Camera: [F1][F2][F3][F4][F5][F6]...
  Process: [F1]      [F4]      [F7]...
  Process every Nth frame (N = camera_fps / inference_fps)
```

```python
class FramePolicy:
    """Frame selection policies for real-time video inference."""

    @staticmethod
    def latest_only(buffer):
        """Drop all but the most recent frame."""
        while len(buffer) > 1:
            buffer.popleft()
        return buffer.popleft() if buffer else None

    @staticmethod
    def skip_n(buffer, n=2):
        """Process every Nth frame."""
        frame = None
        for _ in range(min(n, len(buffer))):
            frame = buffer.popleft()
        return frame

    @staticmethod
    def deadline_aware(buffer, max_age_ms=30.0):
        """Drop frames older than max_age_ms."""
        now = time.perf_counter_ns()
        while buffer:
            frame, timestamp = buffer[0]
            age_ms = (now - timestamp) / 1e6
            if age_ms > max_age_ms:
                buffer.popleft()  # Too old, drop
            else:
                return buffer.popleft()
        return None
```

---

## 16. Production Real-Time Patterns

### 16.1 Watchdog Timers

A watchdog ensures the inference loop does not hang:

```cpp
#include <atomic>
#include <thread>
#include <chrono>

class InferenceWatchdog {
    std::atomic<bool> alive_{true};
    std::atomic<uint64_t> last_heartbeat_{0};
    std::thread watchdog_thread_;
    uint64_t timeout_ms_;
    std::function<void()> on_timeout_;

public:
    InferenceWatchdog(uint64_t timeout_ms,
                      std::function<void()> on_timeout)
        : timeout_ms_(timeout_ms), on_timeout_(on_timeout)
    {
        last_heartbeat_.store(now_ms());
        watchdog_thread_ = std::thread([this]() {
            while (alive_.load()) {
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(timeout_ms_ / 2));
                uint64_t elapsed = now_ms() - last_heartbeat_.load();
                if (elapsed > timeout_ms_) {
                    fprintf(stderr, "WATCHDOG: Inference timeout "
                            "(%lu ms without heartbeat)\n", elapsed);
                    on_timeout_();
                }
            }
        });
    }

    void heartbeat() {
        last_heartbeat_.store(now_ms());
    }

    ~InferenceWatchdog() {
        alive_.store(false);
        if (watchdog_thread_.joinable())
            watchdog_thread_.join();
    }

private:
    uint64_t now_ms() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()
        ).count();
    }
};

// Usage:
InferenceWatchdog watchdog(100, []() {  // 100 ms timeout
    // Emergency action: restart inference, switch to fallback model
    fprintf(stderr, "Restarting inference engine...\n");
    // ... reinitialize engine ...
});

while (running) {
    auto result = engine.infer(input);
    watchdog.heartbeat();
    process(result);
}
```

### 16.2 Deadline Monitoring

```python
import time
import logging

class DeadlineMonitor:
    def __init__(self, deadline_ms, model_name="default"):
        self.deadline_ms = deadline_ms
        self.model_name = model_name
        self.violations = 0
        self.total = 0
        self.logger = logging.getLogger(f"deadline.{model_name}")

    def check(self, latency_ms):
        self.total += 1
        if latency_ms > self.deadline_ms:
            self.violations += 1
            violation_rate = self.violations / self.total * 100
            self.logger.warning(
                f"Deadline violation: {latency_ms:.2f} ms > "
                f"{self.deadline_ms} ms "
                f"(rate: {violation_rate:.2f}%)"
            )
            return False
        return True

    def sla_compliant(self, target_rate=99.9):
        """Check if meeting SLA (e.g., 99.9% within deadline)."""
        if self.total == 0:
            return True
        success_rate = (self.total - self.violations) / self.total * 100
        return success_rate >= target_rate

    def report(self):
        success_rate = ((self.total - self.violations)
                        / max(self.total, 1) * 100)
        return {
            "model": self.model_name,
            "deadline_ms": self.deadline_ms,
            "total_inferences": self.total,
            "violations": self.violations,
            "success_rate_pct": f"{success_rate:.3f}",
        }
```

### 16.3 Graceful Degradation Under Load

```python
class AdaptiveInference:
    """
    Switch between quality tiers based on latency budget.

    Tier 0: Full model (FP16, 640x640) -- highest accuracy
    Tier 1: Reduced resolution (FP16, 320x320) -- faster
    Tier 2: INT8 reduced model (INT8, 320x320) -- fastest
    Tier 3: Skip inference, use last result -- emergency
    """

    def __init__(self, engines, deadline_ms=20.0):
        self.engines = engines  # List of (engine, resolution) by tier
        self.deadline_ms = deadline_ms
        self.current_tier = 0
        self.consecutive_violations = 0
        self.consecutive_ok = 0
        self.last_result = None

    def infer(self, frame):
        if self.current_tier >= len(self.engines):
            # Emergency: return last known result
            return self.last_result

        engine, resolution = self.engines[self.current_tier]

        t0 = time.perf_counter_ns()
        preprocessed = preprocess(frame, resolution)
        result = engine.infer(preprocessed)
        t1 = time.perf_counter_ns()
        latency_ms = (t1 - t0) / 1e6

        self.last_result = result

        # Adapt tier based on latency
        if latency_ms > self.deadline_ms * 0.9:
            self.consecutive_violations += 1
            self.consecutive_ok = 0
            if self.consecutive_violations >= 3:
                self._downgrade()
        else:
            self.consecutive_ok += 1
            self.consecutive_violations = 0
            if self.consecutive_ok >= 50:
                self._upgrade()

        return result

    def _downgrade(self):
        if self.current_tier < len(self.engines):
            self.current_tier += 1
            self.consecutive_violations = 0
            print(f"Degrading to tier {self.current_tier}")

    def _upgrade(self):
        if self.current_tier > 0:
            self.current_tier -= 1
            self.consecutive_ok = 0
            print(f"Upgrading to tier {self.current_tier}")
```

### 16.4 SLA Compliance Framework

```python
class SLATracker:
    """
    Track SLA compliance over sliding windows.

    Example SLA: 99.9% of inferences under 15 ms,
                 measured over 1-minute windows.
    """

    def __init__(self, deadline_ms, target_pct=99.9,
                 window_seconds=60):
        self.deadline_ms = deadline_ms
        self.target_pct = target_pct
        self.window_seconds = window_seconds
        self.measurements = deque()  # (timestamp, latency_ms)

    def record(self, latency_ms):
        now = time.time()
        self.measurements.append((now, latency_ms))
        self._evict_old(now)

    def _evict_old(self, now):
        cutoff = now - self.window_seconds
        while self.measurements and self.measurements[0][0] < cutoff:
            self.measurements.popleft()

    def is_compliant(self):
        if len(self.measurements) < 10:
            return True  # Not enough data
        within = sum(1 for _, lat in self.measurements
                     if lat <= self.deadline_ms)
        pct = within / len(self.measurements) * 100
        return pct >= self.target_pct

    def current_percentile(self):
        if not self.measurements:
            return 0.0
        within = sum(1 for _, lat in self.measurements
                     if lat <= self.deadline_ms)
        return within / len(self.measurements) * 100

    def alert_if_noncompliant(self):
        if not self.is_compliant():
            pct = self.current_percentile()
            print(f"SLA VIOLATION: {pct:.2f}% within {self.deadline_ms}ms "
                  f"(target: {self.target_pct}%)")
            return True
        return False
```

---

## 17. Common Issues and Debugging

### 17.1 Latency Spikes

**Symptom:** Occasional inference takes 2-5x longer than normal.

**Common causes and solutions:**

```bash
# Cause 1: GPU clock scaling (DVFS)
# The GPU dynamically adjusts frequency based on load.
# Fix: Lock clocks with jetson_clocks
sudo jetson_clocks
sudo jetson_clocks --show

# Verify GPU frequency is locked:
cat /sys/devices/gpu.0/devfreq/17000000.ga10b/cur_freq
# Should show maximum frequency

# Cause 2: CPU throttling due to thermals
# Check current CPU frequency:
for i in $(seq 0 5); do
    echo "CPU$i: $(cat /sys/devices/system/cpu/cpu$i/cpufreq/scaling_cur_freq) kHz"
done

# Check thermal zones:
for zone in /sys/class/thermal/thermal_zone*/; do
    echo "$(cat ${zone}type): $(cat ${zone}temp) m-degC"
done

# Cause 3: Memory pressure causing swapping
free -h
cat /proc/meminfo | grep -E "MemAvail|SwapUsed"
# If SwapUsed > 0, you are swapping -- reduce memory usage
```

**Diagnostic script for latency spikes:**

```python
import subprocess
import time

class JetsonHealthMonitor:
    def __init__(self):
        self.warnings = []

    def check_gpu_clock(self):
        try:
            freq = int(open(
                "/sys/devices/gpu.0/devfreq/17000000.ga10b/cur_freq"
            ).read().strip())
            max_freq = int(open(
                "/sys/devices/gpu.0/devfreq/17000000.ga10b/max_freq"
            ).read().strip())
            if freq < max_freq * 0.95:
                self.warnings.append(
                    f"GPU clock below max: {freq/1e6:.0f} / "
                    f"{max_freq/1e6:.0f} MHz")
        except Exception as e:
            self.warnings.append(f"Cannot read GPU clock: {e}")

    def check_thermals(self):
        import glob
        for zone in glob.glob("/sys/class/thermal/thermal_zone*"):
            try:
                temp = int(open(f"{zone}/temp").read().strip()) / 1000.0
                zone_type = open(f"{zone}/type").read().strip()
                if temp > 85.0:
                    self.warnings.append(
                        f"THERMAL WARNING: {zone_type} = {temp:.1f} C")
            except Exception:
                pass

    def check_memory(self):
        with open("/proc/meminfo") as f:
            lines = f.readlines()
        info = {}
        for line in lines:
            parts = line.split()
            info[parts[0].rstrip(":")] = int(parts[1])
        avail_mb = info.get("MemAvailable", 0) / 1024
        swap_used = (info.get("SwapTotal", 0)
                     - info.get("SwapFree", 0)) / 1024
        if avail_mb < 500:
            self.warnings.append(
                f"Low memory: {avail_mb:.0f} MB available")
        if swap_used > 10:
            self.warnings.append(
                f"Swapping active: {swap_used:.0f} MB used")

    def full_check(self):
        self.warnings = []
        self.check_gpu_clock()
        self.check_thermals()
        self.check_memory()
        return self.warnings
```

### 17.2 GPU Scheduling Interference

**Symptom:** Inference latency increases when other GPU workloads are running
(e.g., display compositor, video decode).

```bash
# Check what is using the GPU
sudo cat /sys/kernel/debug/gpu.0/load
# Shows GPU utilization percentage

# Identify processes using GPU memory
nvidia-smi  # (if available on Jetson)
# Or check /proc/*/status for gpu_mem entries

# Mitigation 1: Use DLA for deterministic inference
# (DLA is not affected by GPU contention)

# Mitigation 2: Disable the desktop compositor
sudo systemctl stop gdm3
sudo systemctl disable gdm3
# This frees ~200 MB GPU memory and removes display-driven GPU work

# Mitigation 3: Use separate CUDA streams with priorities
```

```cpp
// Create a high-priority CUDA stream for inference
cudaStream_t inferenceStream;
int leastPriority, greatestPriority;
cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
// greatestPriority is the highest (most urgent) priority
cudaStreamCreateWithPriority(&inferenceStream,
                              cudaStreamNonBlocking,
                              greatestPriority);

// Other GPU work uses lower priority
cudaStream_t backgroundStream;
cudaStreamCreateWithPriority(&backgroundStream,
                              cudaStreamNonBlocking,
                              leastPriority);
```

### 17.3 Memory Fragmentation

**Symptom:** `cudaMalloc` fails or becomes slow after hours of operation, even
though total free memory appears sufficient.

```cpp
// Diagnostic: check fragmented vs total free memory
size_t free_bytes, total_bytes;
cudaMemGetInfo(&free_bytes, &total_bytes);
printf("GPU memory: %zu MB free / %zu MB total\n",
       free_bytes / (1 << 20), total_bytes / (1 << 20));

// Prevention: allocate all GPU memory at startup, never free
// Use a custom memory pool:
class InferenceMemoryPool {
    void* pool_;
    size_t pool_size_;
    size_t offset_ = 0;

public:
    InferenceMemoryPool(size_t size_mb) {
        pool_size_ = size_mb << 20;
        cudaMalloc(&pool_, pool_size_);
    }

    void* alloc(size_t bytes) {
        // Align to 256 bytes (GPU requirement)
        size_t aligned = (bytes + 255) & ~255;
        if (offset_ + aligned > pool_size_) {
            fprintf(stderr, "Pool exhausted!\n");
            return nullptr;
        }
        void* ptr = static_cast<char*>(pool_) + offset_;
        offset_ += aligned;
        return ptr;
    }

    void reset() { offset_ = 0; }

    ~InferenceMemoryPool() { cudaFree(pool_); }
};
```

### 17.4 Thermal-Induced Slowdowns

The Orin Nano throttles CPU and GPU clocks when temperature exceeds limits:

```
Thermal zones on Orin Nano (T234):

  Zone          Warning    Throttle    Shutdown
  CPU           85 C       95 C        105 C
  GPU           85 C       95 C        105 C
  SOC           85 C       95 C        105 C
  TBoard        80 C       90 C        100 C
```

```bash
# Monitor thermals in real time
watch -n 1 'paste <(cat /sys/class/thermal/thermal_zone*/type) \
                   <(cat /sys/class/thermal/thermal_zone*/temp) | \
             awk "{printf \"%20s: %5.1f C\n\", \$1, \$2/1000}"'

# Check if throttling is active
cat /sys/devices/gpu.0/devfreq/17000000.ga10b/trans_stat
# Shows frequency transition history -- frequent changes = throttling

# Mitigation:
# 1. Use an active cooling fan
sudo bash -c 'echo 255 > /sys/devices/pwm-fan/target_pwm'

# 2. Use a heat sink with proper thermal compound

# 3. Set power mode to reduce heat generation
sudo nvpmodel -m 2    # 15W mode (balanced)
sudo nvpmodel -m 0    # MAXN (maximum performance, most heat)

# 4. Monitor and adapt -- reduce model complexity when hot
```

**Thermal-aware inference loop:**

```python
class ThermalAwareInference:
    THROTTLE_TEMP = 82.0  # Start reducing load before HW throttle

    def __init__(self, full_engine, lite_engine):
        self.full_engine = full_engine
        self.lite_engine = lite_engine
        self.use_lite = False

    def get_max_temp(self):
        import glob
        max_temp = 0
        for zone in glob.glob("/sys/class/thermal/thermal_zone*"):
            try:
                temp = int(open(f"{zone}/temp").read()) / 1000.0
                max_temp = max(max_temp, temp)
            except Exception:
                pass
        return max_temp

    def infer(self, input_data):
        # Check temperature every 100 inferences
        if self.inference_count % 100 == 0:
            temp = self.get_max_temp()
            if temp > self.THROTTLE_TEMP and not self.use_lite:
                print(f"Temperature {temp:.1f}C > {self.THROTTLE_TEMP}C, "
                      f"switching to lite model")
                self.use_lite = True
            elif temp < self.THROTTLE_TEMP - 5.0 and self.use_lite:
                print(f"Temperature {temp:.1f}C cooled down, "
                      f"switching to full model")
                self.use_lite = False

        engine = self.lite_engine if self.use_lite else self.full_engine
        self.inference_count += 1
        return engine.infer(input_data)
```

### 17.5 Clock Scaling Surprises

**Symptom:** First inference after idle is 3-10x slower.

The GPU enters a low-power state during idle, and the first operation triggers
a clock ramp-up that can take 5-20 ms.

```bash
# Check current GPU power state
cat /sys/devices/gpu.0/power/runtime_status
# "active" or "suspended"

# Prevent GPU from entering low-power state
# Method 1: jetson_clocks (locks all clocks)
sudo jetson_clocks

# Method 2: Disable GPU runtime PM
echo on > /sys/devices/gpu.0/power/control

# Method 3: Keep a lightweight "heartbeat" kernel running
```

```cpp
// Heartbeat kernel to prevent GPU idle
__global__ void heartbeat_kernel() {
    // Minimal work -- just enough to keep GPU "active"
    if (threadIdx.x == 0) {
        volatile int x = 1;
    }
}

void gpu_heartbeat_thread(cudaStream_t stream) {
    while (running) {
        heartbeat_kernel<<<1, 1, 0, stream>>>();
        cudaStreamSynchronize(stream);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}
```

### 17.6 Debugging Checklist

When inference latency is not meeting requirements, work through this checklist:

```
STEP  CHECK                                   COMMAND / ACTION
----  -----                                   ----------------
 1    Is PREEMPT_RT kernel active?             cat /sys/kernel/realtime
 2    Are clocks locked to maximum?            sudo jetson_clocks --show
 3    Is thermal throttling occurring?         Check thermal zones
 4    Is the GPU being shared?                 Disable desktop, check load
 5    Is the right precision mode used?        trtexec --best --verbose
 6    Are CPU cores isolated?                  cat /sys/devices/system/cpu/isolated
 7    Is the inference thread RT-scheduled?    chrt -p <pid>
 8    Is memory being allocated in hot path?   Profile with nsys
 9    Are there page faults?                   perf stat -e page-faults ./app
10    Is swap being used?                      free -h
11    Is the input pipeline on GPU?            Profile preprocess stage
12    Is CUDA graph being used?                Check launch overhead in nsys
13    Are IRQs on inference cores?             Check /proc/interrupts
14    Is the engine built with timing cache?   Rebuild with --timingCacheFile
15    Is warmup sufficient?                    Run 100+ warmup iterations
```

### 17.7 Complete Production Startup Sequence

```bash
#!/bin/bash
# production_start.sh -- Full RT inference startup on Orin Nano
set -e

echo "=== Production RT Inference Startup ==="

# 1. Lock clocks
echo "[1/8] Locking clocks..."
sudo jetson_clocks

# 2. Set power mode (MAXN for maximum performance)
echo "[2/8] Setting power mode..."
sudo nvpmodel -m 0

# 3. Set fan to maximum
echo "[3/8] Setting fan speed..."
sudo bash -c 'echo 255 > /sys/devices/pwm-fan/target_pwm' 2>/dev/null || \
    echo "  (no fan control available)"

# 4. Disable desktop if running
echo "[4/8] Disabling desktop compositor..."
sudo systemctl stop gdm3 2>/dev/null || true

# 5. Set up CPU isolation and IRQ affinity
echo "[5/8] Configuring CPU affinity..."
for irq in $(ls /proc/irq/ | grep -E '^[0-9]+$'); do
    echo 0-3 > /proc/irq/$irq/smp_affinity_list 2>/dev/null || true
done

# 6. Disable THP
echo "[6/8] Disabling transparent huge pages..."
echo never | sudo tee /sys/kernel/mm/transparent_hugepage/enabled > /dev/null

# 7. Drop filesystem caches
echo "[7/8] Dropping caches..."
sudo bash -c 'echo 3 > /proc/sys/vm/drop_caches'

# 8. Launch inference with RT priority on isolated cores
echo "[8/8] Launching inference application..."
sudo taskset -c 4,5 chrt -f 80 ./inference_app \
    --engine model_fp16.engine \
    --warmup 100 \
    --deadline-ms 15 \
    --camera /dev/video0

echo "=== Startup complete ==="
```

---

## Appendix A: Quick Reference -- Orin Nano Real-Time Tuning Parameters

| Parameter | Default | Real-Time Setting | How to Set |
|-----------|---------|-------------------|------------|
| Kernel preemption | PREEMPT | PREEMPT_RT | Rebuild kernel |
| CPU governor | schedutil | performance | cpufreq sysfs |
| GPU clock | Dynamic | Locked max | jetson_clocks |
| CPU isolation | None | isolcpus=4,5 | Boot param |
| IRQ affinity | All cores | Cores 0-3 only | /proc/irq/*/smp_affinity |
| THP | always | never | sysfs |
| Scheduler | SCHED_OTHER | SCHED_FIFO (80) | chrt / sched_setscheduler |
| CUDA stream | Default | High-priority | cudaStreamCreateWithPriority |
| TRT precision | FP32 | FP16 or INT8 | Builder flags |
| Batch size | Model-dependent | 1 | Engine build shapes |
| Memory locking | Disabled | mlockall | mlockall(MCL_CURRENT) |
| Swap | Enabled | Disabled | swapoff -a |
| Desktop | Enabled | Disabled | systemctl stop gdm3 |

## Appendix B: Latency Budget Template

```
Application: ___________________________
Target FPS:  _____ (frame deadline: _____ ms)
SLA:         _____ % within deadline

Stage               Budget (ms)    Measured p50   Measured p99
-----------------------------------------------------------------
Camera capture       ________       ________       ________
Preprocessing        ________       ________       ________
H2D transfer         ________       ________       ________
Inference            ________       ________       ________
D2H transfer         ________       ________       ________
Postprocessing       ________       ________       ________
Application logic    ________       ________       ________
-----------------------------------------------------------------
Total                ________       ________       ________

Meets SLA?  [ ] Yes  [ ] No
Action items: ________________________________________________
```

## Appendix C: TensorRT Version Compatibility

| JetPack | L4T | CUDA | TensorRT | DLA Compiler | Notes |
|---------|-----|------|----------|--------------|-------|
| 5.1.2 | 35.4.1 | 11.4 | 8.5.2 | 3.12.0 | Last JP5 |
| 6.0 | 36.3.0 | 12.2 | 8.6.2 | 3.16.0 | First JP6 |
| 6.1 | 36.4.0 | 12.6 | 10.3 | 3.17.0 | Current stable |

Engines built with one TensorRT version are **not compatible** with another.
Always rebuild engines when upgrading JetPack.

---

*End of guide. For updates, refer to the NVIDIA Jetson documentation at
https://docs.nvidia.com/jetson/ and the TensorRT documentation at
https://docs.nvidia.com/deeplearning/tensorrt/.*
