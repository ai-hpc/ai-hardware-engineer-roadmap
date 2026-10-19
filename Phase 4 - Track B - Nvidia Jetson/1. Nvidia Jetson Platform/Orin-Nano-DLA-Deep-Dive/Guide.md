# DLA capability and accelerator partitioning — Nano, NX, and AGX

<div class="course-identity jetson-platform" markdown="1">
<div class="course-identity__icon">DLA</div>
<div markdown="1">
<p class="course-identity__eyebrow">Jetson · Accelerator comparison</p>
<p class="course-identity__title">Verify the hardware, partition a graph, and measure the cost of crossing engines.</p>
<p class="course-identity__meta">Artifact: capability and placement report · Measure: parity, latency, energy, transitions</p>
</div>
</div>

**Orin Nano has no DLA.** This course has a Nano-compatible capability lab and an optional execution lab on DLA-equipped Orin NX/AGX hardware. The directory is retained so existing links continue to work. NVIDIA's [Orin power tables](https://docs.nvidia.com/jetson/archives/r36.5/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html) distinguish SKU capabilities and mode-dependent active cores.

**Before:** [memory ownership and mappings](../Orin-Nano-Memory-Architecture/Guide.md), [CUDA programming](../Orin-Nano-CUDA-Programming/Guide.md), and graph optimization. **After:** take the placement report into [runtime and driver design](../../8.%20Runtime%20and%20Driver%20Development/Guide.md), then the custom accelerator project.

## 1. Start from a capability contract

A SoC family may contain blocks that are unavailable on a particular module. A compiler must reject unsupported placements before launch. Check the module label, BSP, power mode, TensorRT version, and runtime capability together:

```python
# Run with the TensorRT package installed for the target's BSP.
import tensorrt as trt
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
print({"tensorrt": trt.__version__, "dla_cores": builder.num_DLA_cores})
```

On Nano, zero DLA cores is the expected outcome, not a missing driver to fix. Save that result and run the GPU baseline. On other modules, reject a requested core index unless `0 <= core < builder.num_DLA_cores`. Keep the capability check in the deployment manifest so a plan compiled for one SKU cannot silently ship to another.

## 2. What a dedicated inference engine changes

A DLA implements a constrained set of tensor operations with its own execution resources. A GPU provides a more programmable kernel path. Dedicated compute can free GPU capacity, but shared DRAM, host queues, thermal limits, and format conversions remain. Neither a fixed-function block nor a low observed p99 establishes deterministic worst-case latency.

The [TensorRT DLA documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-with-dla.html) defines the supported layer, shape, precision, and memory-format restrictions. As checked on 2026-10-18, the current TensorRT 11.4 page explicitly excludes JetPack DLA support and points to supported earlier configurations. Select the archive and runtime supported by the installed JetPack; do not treat a generic latest package as a valid Jetson upgrade. An operator name alone is not enough: a supported convolution with an unsupported parameter combination can still be rejected. GPU fallback must be explicitly enabled where supported; a strict DLA build instead exposes unsupported regions.

For a VLA, do not assume that supporting its image encoder implies support for attention, normalization, autoregressive decoding, or a diffusion/flow action head. Partition the actual exported graph.

## 3. Work through a partitioning decision

Suppose a candidate graph has `conv → activation → unsupported transform → conv`. A GPU-only plan costs 5.0 ms. A hypothetical mixed plan spends 2.8 ms on DLA operations, 0.9 ms on the GPU transform, and crosses engines twice at 0.8 ms each. The mixed total is **5.3 ms** before any overlap. These are teaching numbers, not hardware results.

This explains why maximizing the percentage of DLA layers is a poor objective. Better objectives are sensor-to-result latency, throughput at a quality target, or joules per completed request. Count layout conversions, copies, and fences; then measure the whole path. If overlapping independent models, measure their concurrent performance rather than adding their isolated throughputs.

## 4. Optional execution lab on supported hardware

Use a small, fixed-shape CNN first. Record the ONNX hash and input shape. The following command pattern is for a compatible TensorRT 10.x installation; check that release's `trtexec --help` before running:

```bash
# GPU reference
trtexec --onnx=model.onnx --fp16 --saveEngine=gpu.plan --verbose
# Strict DLA compilation: no GPU fallback, so restrictions are visible
trtexec --onnx=model.onnx --fp16 --useDLACore=0 --saveEngine=dla.plan --verbose
# Mixed placement, if the graph requires GPU work
trtexec --onnx=model.onnx --fp16 --useDLACore=0 --allowGPUFallback \
  --saveEngine=mixed.plan --verbose
```

Do not treat `--fp16` as a guarantee that every intermediate uses the same precision. Save build logs, inspect engine layers, and compare real outputs using a separate inference harness; `trtexec` timing by itself does not prove task accuracy. An INT8 experiment also needs a valid quantization/calibration workflow, not just an INT8 flag.

Run GPU-only and mixed plans on identical inputs under the same power and thermal conditions. Include cold start, warmed execution, and execution while a second workload is active. Report sample count, p50/p95/p99, maximum observed latency, peak memory, and task quality. Whole-board energy requires a specified measurement boundary; `tegrastats` rails are not automatically total system energy.

## 5. Memory, synchronization, and driver lessons

A DLA-equipped Jetson still uses system DRAM for model and tensor allocations. An imported buffer needs a supported layout, appropriate mapping, a producer-completion dependency, and a known lifetime. Shared DRAM is not proof of a zero-copy or zero-conversion handoff. Releasing a buffer after enqueue but before completion is a runtime defect on any accelerator.

For the custom physical-AI SoC, write the contract first:

| Interface | Required decision |
|---|---|
| Compiler → accelerator | Supported operators, dtypes, layouts, alignment, dynamic-shape policy |
| Runtime → driver | Command format, buffer handles, queue depth, cancellation semantics |
| Driver → memory system | DMA mapping, cache ownership, bounds, completion signaling |
| Inference → controller | Timestamp, action frame/units, validity horizon, timeout behavior |

The open [NVDLA hardware documentation](https://nvdla.org/hw/contents.html) is an architecture study resource. It is not a promise that NVIDIA's shipped Jetson DLA firmware or hardware is identical to the open reference implementation.

## Practice and assessment

**Nano path:** submit a capability manifest, a validated GPU baseline, and an annotated graph identifying candidate offload regions. Mark the DLA timing column **not run — no DLA hardware**. This is a complete capability/partitioning exercise, not a DLA performance result.

**NX/AGX path:** also submit strict-build failures, final placement, output comparisons on held-out inputs, and simultaneous-load measurements. The lab passes when the deployment rejects an unavailable core, numerical/task error stays within a declared budget, and every claimed improvement includes transition costs. A slower mixed plan is a valid engineering finding.

**Handoff:** give the operator support matrix and one expensive fallback boundary to the compiler course. Propose either a legal graph rewrite, an accelerator instruction, or a deliberate GPU fallback, with its memory and verification cost.
