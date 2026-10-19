# Tensor Cores: matrix work, numeric formats and measured benefit

**Goal:** explain what Tensor Cores accelerate, calculate useful matrix work, and evaluate a lower-precision model without confusing a format name with a performance result. **Pace:** three 60–90-minute sessions. Start after [CUDA programming](../Orin-Nano-CUDA-Programming/Guide.md) and basic matrix multiplication.

Orin Nano 8 GB has 32 Tensor Cores within eight Ampere SMs. They accelerate supported matrix operations alongside the GPU's general-purpose execution units. A complete inference pipeline also needs memory movement, normalization, activation functions, indexing, scheduling and host work. No single hardware peak describes all of that work.

## 1. Start with the actual operation

A matrix multiply-accumulate computes `D = A × B + C`. If A is M×K and B is K×N, each output sums K products. The common dense operation-count convention is approximately `2MNK` floating-point operations, counting a multiply and an add separately. State the convention when reporting FLOPs.

**Hand example:** A=`[[1,2],[3,4]]`, B=`[[2,0],[1,2]]`, C=zeros. The output is `[[4,4],[10,8]]`. The top-left value is `1×2 + 2×1 = 4`. This small example establishes layout and arithmetic before tiling or quantization changes the implementation.

For M=N=K=256, the convention gives 33,554,432 FLOPs. If a hypothetical correctly measured execution interval is 0.20 ms, achieved throughput is approximately 167.8 GFLOP/s. The time is illustrative; it is not an Orin result. A different interval that includes allocation/copies answers a different performance question.

## 2. Understand what a matrix instruction promises

At the CUDA programming level, a warp cooperates on matrix fragments and collective matrix operations. Fragment layout across lanes is an interface contract; do not infer it from a diagram of a logical matrix. Correct participation, data layout, alignment and supported shapes matter.

A WMMA operation is not a promise that a complete logical 16×16×16 multiplication takes one clock cycle or maps to one machine instruction. Compilation and architecture determine the actual instruction sequence and scheduling. Tensor Cores do not remove the time to load operands, handle boundary tiles or write results. Use the [PTX matrix-operation specification](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-wmma) for exact instruction constraints.

Use a compatible cuBLAS/cuDNN/TensorRT path before writing a custom matrix kernel. Libraries can select different tactics for dimensions, layout, precision and workspace. Their internal implementation is not necessarily the public WMMA C++ API. A requested low-precision mode also does not establish that every operation uses a Tensor Core.

## 3. Separate storage type, multiply precision and accumulation

| Format/concept | What to remember |
| --- | --- |
| FP32 | 32-bit storage with a wider significand than FP16/BF16; a library may permit reduced-precision multiply behavior under its policy |
| FP16 | 16-bit floating-point storage; limited exponent range can overflow on values that fit FP32 |
| BF16 | 16-bit storage with FP32-like exponent range but fewer significand bits than FP16 |
| TF32 computation | Reduced-significand matrix multiplication using FP32 inputs/outputs under supported policies; not a general 19-bit tensor storage dtype |
| INT8 | Integer values plus quantization scale/zero-point/layout semantics; accumulator and requantization behavior matter |
| INT4 or weight-only formats | Packed representation and kernel-specific decoding; four-bit weights do not imply that every multiply executes as native four-bit matrix arithmetic |

FP16's largest finite value is 65,504. BF16 retains the broad FP32 exponent range but rounds many neighboring values to the same representation. A model can therefore encounter range problems in one format and precision problems in another. Accumulating products in FP32 can reduce accumulation error while still retaining errors introduced when operands were rounded earlier.

A float32 tensor may use Tensor Core matrix arithmetic through a TF32-enabled library path. Consequently, “FP32 always means CUDA cores only” is incorrect without specifying the compute policy. Conversely, changing storage to FP16 does not make an arbitrary elementwise kernel a matrix operation.

For accuracy testing, compare against a specified reference, retain absolute/relative error statistics and examine sensitive operations such as reductions, normalization and softmax. Use the installed runtime's precision controls and inspect the selected graph/tactics. [TensorRT accuracy considerations](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/accuracy-considerations.html).

## 4. Calculate one quantized dot product

For a simple symmetric quantizer, approximate a real value by `x ≈ scale_x × q_x`, with integer q and a positive scale. Suppose x=`[0.2,−0.4]`, w=`[0.3,0.1]`, and both scales are 0.1. Integer vectors are qx=`[2,−4]` and qw=`[3,1]`.

The integer dot product is `2×3 + (−4)×1 = 2`. Multiplying by the scale product, `0.1×0.1`, gives 0.02, matching the unquantized result for these deliberately representable values. With nonzero zero-points, subtract them before the product according to the quantization scheme. Per-channel scales introduce another indexing contract.

Now choose a scale too small for an outlier: the integer value may exceed the representable range and saturate. Choose a scale too large: nearby small values collapse into the same integer. Calibration and clipping choices trade those errors; a lower bit width is not automatically accurate for a specific deployment distribution.

**Exercise:** quantize 0.26 with scale 0.1 under a specified round-to-nearest policy. The stored integer is 3 and reconstructed value is 0.3, an error of 0.04. State the tie policy before testing an exact halfway value. Different rounding and saturation rules can make otherwise similar CPU/hardware implementations disagree.

## 5. Work through a real export and calibration experiment

Complete the [CPU PyTorch→ONNX→INT8 lab](../../5.%20Application%20Development/5.%20ML%20and%20AI/cpu-export-quantization/Guide.md). It trains/loads a small synthetic stop-decision model, exports it and executes ONNX Runtime on the CPU. The review actually ran the framework integration tests.

In the recorded exercise, representative calibration retained stop recall around 99.24%, while repeatedly calibrating on one row reduced it to about 64.89%. A separate sensor-shift test also reduced recall. Those are results for the synthetic dataset and model, not field-robot safety guarantees. The tiny quantized artifact was larger because quantization metadata/node overhead outweighed its weight savings.

This experiment establishes why “INT8 means only a slight accuracy loss and at least 2× speedup” is not a valid course rule. Quality, size and latency are separate measurements. CPU ONNX Runtime results do not prove TensorRT/Orin speed, and an equivalent exported graph still needs its target implementation checked.

Use held-out data by the appropriate session/scene/task split. Keep calibration data separate from evaluation data. Report false negatives or other task-relevant failures, not only aggregate accuracy. Preserve the exact model, preprocessing, calibration and runtime identities with the result.

## 6. Count memory reuse as well as operations

For an idealized FP16 square GEMM with M=N=K=256, A and B each contain 65,536 elements or 131,072 bytes. Writing an FP32 output adds 262,144 bytes. Ignoring a separate C read and all repeated loads, requested traffic is 524,288 bytes; arithmetic intensity is `33,554,432 / 524,288 = 64 FLOP/byte`.

That is an optimistic traffic model, not measured DRAM traffic. Actual tiling, caches, workspace, output type, C/bias/activation reads and split reductions change it. State what you counted. A tiny batch or a matrix-vector operation can have very different reuse from the square matrix example.

Dense FP16 FLOPs, dense INT8 operations and sparse INT8 operations are different quantities. Structured-sparse peak specifications assume an eligible sparsity pattern and execution path; ordinary zeros in arbitrary weights are insufficient. Compare achieved work to a relevant peak and supported mode, not to the largest TOPS number on the product page.

Shared LPDDR bandwidth is also used by CPU, cameras and other engines. A kernel measured alone may compete differently in the full robot pipeline. Connect this calculation to the [memory architecture lesson](../Orin-Nano-Memory-Architecture/Guide.md).

## 7. Run a target experiment when hardware is available

Pin the BSP, TensorRT/library versions, model hash, input shapes, precision controls and power mode. Begin with a compatible reference implementation and numerical comparisons. Check unsupported operators, dynamic-shape profiles, layout conversions and fallback behavior.

Then measure cold initialization, warmed inference, queue delay, peak memory and useful end-to-end throughput. Compare the same input workload and quality requirement. Include concurrent camera/CPU activity if the deployed robot uses them. Record clocks/temperature rather than treating an advertised peak as sustained operation.

Inspect the engine/kernel evidence to determine what executed. A `--fp16` flag alone does not prove every layer's precision or which hardware units were used. A profile alone does not establish task accuracy. The [real-time inference module](../Orin-Nano-Real-Time-Inference/Guide.md) connects those checks to frame freshness and deadline behavior.

## 8. Answer before continuing

1. **Does a matrix instruction imply one clock cycle?** No; API shape and instruction scheduling are different levels.
2. **Can float32 tensors use Tensor Core arithmetic?** Yes, under a supported library compute policy such as TF32; specify that policy when comparing accuracy.
3. **Why can a quantized tiny model be larger?** Metadata and graph overhead can exceed the saved weight bytes.
4. **Why does FP32 accumulation not eliminate operand rounding error?** The products already use rounded input representations.
5. **Why compare task recall as well as numerical error?** Small numerical changes can cross decision thresholds and affect the task unevenly.
6. **Why can fewer bytes fail to improve latency?** Conversion overhead, unsupported operations, poor tiling or another pipeline bottleneck may dominate.

Submit the hand matrix/dot-product calculations, the real CPU calibration report and a target-test plan. If you execute on hardware, add numerical and timing evidence for that exact configuration; keep the CPU and target results labeled separately.
