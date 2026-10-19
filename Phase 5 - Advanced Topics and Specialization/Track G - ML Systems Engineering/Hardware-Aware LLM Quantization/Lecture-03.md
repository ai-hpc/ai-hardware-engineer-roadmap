# Module 03 — Blackwell Hardware: Match the Format to the Kernel

**Collection:** [Hardware-Aware LLM Quantization](README.md) | **Previous:** [Previous lesson](Lecture-02.md) | **Next:** [Next lesson](Lecture-04.md)

A format on disk, an instruction's operand type, and a runtime's supported model
are three different contracts. This lesson connects them without treating a model
name or GPU family name as a complete compatibility statement.

## 1. Establish the device and units

The RTX 5090 is a consumer Blackwell device with 32 GB GDDR7 and advertised
1792 GB/s bandwidth. NVIDIA lists 3352 AI TOPS; distinguish advertised precision
and structured sparsity assumptions before using that number in a dense roofline.
It is not a measured token rate. [NVIDIA specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/).

The historical course estimates use approximately 419 dense BF16, 838 FP8 and
1676 FP4 TFLOP/s at the advertised operating point. They are idealized peaks,
not a promise for arbitrary shape, accumulation mode, clocks or runtime.
Compare measured operators to their exact supported arithmetic path.

## 2. Two Blackwell instruction families

| Target | Typical architecture-specific path | Consequence |
|---|---|---|
| Datacenter SM100 | `tcgen05`, Tensor Memory | its kernel schedules/layouts are architecture-specific |
| Consumer SM120 | warp-level `mma.sync` block-scaled operations | use an SM120-supported operator and build target |

The [CUTLASS 4.8 warp-level guide](https://docs.nvidia.com/cutlass/4.8.0/media/docs/pythonDSL/guides/mma/wmma_programming.html)
and [official SM120 NVFP4 example](https://github.com/NVIDIA/cutlass/blob/main/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu)
provide concrete supported paths. The latter is a moving source link: pin its
repository revision, CUDA compiler and architecture target when building it.
“Supports Blackwell” alone does not establish SM100↔SM120 binary or performance
portability. An `a` architecture target denotes architecture-specific features;
consult that example's build requirements instead of copying another GPU's flags.

Native block-scaled FP4 operations need supported operand types, scale layouts,
orientations and tiles. A W4A16 weight-only path generally dequantizes weights for
wider arithmetic; it must not be assigned the native W4A4 FP4 peak automatically.
Likewise an exporter advertising NVFP4 does not prove that its scale arrangement
matches a particular kernel.

## 3. A non-native format can still win

Packing fewer bits saves traffic but adds unpacking, address calculations or
conversion. A tuned fused kernel can hide some of that work; cache, instruction
throughput and operator shape determine the outcome. Non-native is a cost to
measure, not a proof that every INT3/INT4 implementation is slower.

Under a **bandwidth-only illustration**, NVFP4 payload+scales use 0.5625 bytes per
weight; INT3 with one FP16 scale per 128 uses `3/8+2/128=0.390625` bytes. If the
baseline achieves an assumed 92% of peak, the candidate must achieve:

```text
0.92 × 0.390625/0.5625 = 63.89% of peak
```

to tie in that model. At 55%, predicted throughput is `0.55/0.6389=0.861×` baseline.
Those percentages are hypothetical inputs, not measured kernel efficiencies.
Extra padding, zero points, metadata and conversion traffic change the calculation.
At larger batches, the compute path becomes more important as weights are reused.

## 4. Profile the operator, not its label

First collect kernel names and durations, then inspect expensive operators. Names
containing “FP4” are hints, not proof. Inspect the compiled instructions/dispatch,
shapes, scale tensors and timing; compare output accuracy before benchmarking.

The commands below are **upstream-tool workflows, not executed GPU results here**:

```bash
nsys profile -o decode_trace ./your_decode_runner
nsys stats --report cuda_gpu_kern_sum decode_trace.nsys-rep
ncu --query-metrics
```

Query supported metric names on your installed Nsight Compute/device before
constructing a selected-metrics command. A tensor-pipe counter need not be high
for a tiny bandwidth-limited GEMV, and 72% estimated byte-rate does not diagnose
a specific scheduling failure. Profiling replay also perturbs timings; use a
separate end-to-end benchmark for latency.

## 5. Version boundaries

TensorRT `latest` currently describes **11.4**, while JetPack-supported TensorRT
is a different release line. TensorRT 11.4 does not support JetPack. Its strongly
typed/QDQ workflow and supported schemes do not validate historical 10.x builder
flags or this community model's custom SGLang loader.
[Platform boundary](https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-10x-to-11x-jetson.html).

The [TensorRT quantization scheme documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-schemes.html)
expresses graph Q/DQ scales and restrictions. Those graph scale tensors need not
have the same storage dtype as packed runtime block-scale metadata. Do not equate
an ONNX Q/DQ graph with a hardware-ready packed checkpoint.

## Exercise and completion

For one candidate, fill in device capability, compiler, library revision, weight
format, activation format, scale types/layout, accumulator and tested shapes.
Classify each field as documented, observed or unknown. Run the CPU reference lab
to establish numerical expectations; it cannot validate a CUDA dispatch path.

**Answer guidance:** CPU dequantize-to-FP32 inference proves the reference numerical
policy, not native FP4 acceleration. A device performance claim requires the
matching packed kernel, outputs, timing and hardware identity.
