# 05 — Optimize the bottleneck you can demonstrate

**Outcome:** turn a profile into one numerical prediction, change one factor, and recheck correctness and whole-workload performance. Source review: 2026-10-18. GPU profiling/optimization is an unexecuted hardware exercise here.

## Roofline is a bound, not a diagnosis

For a chosen arithmetic mode, attainable throughput is bounded by `min(compute_peak, memory_bandwidth × arithmetic_intensity)`. Arithmetic intensity counts useful FLOPs divided by the bytes crossing the **named memory boundary**. H200 SXM dense BF16 uses about 989.5 TFLOP/s and 4.8 TB/s, giving a ridge point near **206.15 FLOP/byte**. The 412 figure belongs to the doubled structured-sparse compute ceiling, not dense BF16. [H200 specifications](https://www.nvidia.com/en-us/data-center/h200/).

For square dense GEMM of size n with BF16 A/B/output and no rereads, approximate work is `2n³` FLOPs and minimum traffic `6n²` bytes, so intensity is n/3 FLOP/byte. n=2048 gives about 682.7 FLOP/byte. This optimistic bound suggests compute could dominate; small tiles, cache reuse, extra epilogue traffic and kernel dispatch can change actual behavior. A low measured rate does not by itself prove the operation is memory-bound.

For one vector addition with three BF16 arrays (read x/y, write z), there is one FLOP per six bytes: about **0.167 FLOP/byte** at that boundary. A fused epilogue can remove intermediate traffic, but still reads its operands and writes output. An AllReduce performs reductions and moves data; small-message launch/latency and large-message bandwidth require separate analysis.

## Profile the whole operation before a kernel

Begin with the [bounded correctness/measurement script](06-Benchmarks-and-Validation.md), then trace a representative application interval. Record startup/compile separately from steady-state work. Check the installed tool's options and permissions:

```bash
nsys profile --trace=cuda,nvtx --stats=true --output=course_timeline \
    python inference.py
ncu --list-sets
ncu --list-sections
```

Use a supported roofline section from the installed Nsight Compute version, and select a kernel observed in your own trace. A hard-coded Ampere kernel name may never occur on H200. Profilers replay/instrument work and can change timing; collect ordinary end-to-end timing separately. L1 throughput is not L1 hit rate. [Nsight Systems](https://docs.nvidia.com/nsight-systems/), [Nsight Compute](https://docs.nvidia.com/nsight-compute/).

## CUDA Graphs require a stable replay contract

A graph reduces repeated launch overhead; replay is not instantaneous or free. Stable device addresses, supported capture operations, shapes and stream dependencies matter. Autoregressive inference has variable lengths, active requests and cache state, so the graph contract must be designed explicitly.

Use the dedicated [CUDA Graphs lesson](../CUDA-Advanced-Optimization/01-CUDA-Graphs.md) for its complete example. Warmup on another stream needs explicit dependencies before capture. Keep static buffers alive, copy new inputs, replay, and respect completion before reading/reusing results. Bucketing needs correct padding masks, position IDs, output slicing and a fallback when inputs exceed the largest bucket. Zero-padding token IDs alone can change model semantics.

**Prediction exercise:** a hypothetical inference spends 2 ms in CPU launch gaps and 8 ms in irreducible device work. Even removing every launch gap gives at most `10/8=1.25×`, before graph replay/copy overhead. A 3× claim contradicts that particular profile.

## Fusion and compilation must preserve the contract

Fusion can save intermediate writes/reads and launches, but can increase registers, spills or reduce occupancy. A matrix bias of shape `[H]` needs broadcasting; indexing `bias_ptr+flat_offset` over `[B,H]` reads out of bounds. Compare tail shapes and non-contiguous/aliased inputs before timing custom kernels. Use [kernel fusion](../CUDA-Advanced-Optimization/04-Kernel-Fusion.md) for the detailed implementation path.

`torch.compile` modes choose different tradeoffs, not universal optimization levels. `dynamic=True` requests dynamic-shape handling; it does **not** disable compilation or guarantee lower memory. Compilation can create multiple graphs, graph breaks, caches and warmup costs. Preserve one eager reference and compare steady-state latency, compile cost, memory and numerical differences over the shape distribution. [torch.compile API](https://docs.pytorch.org/docs/2.14/generated/torch.compile.html).

## Communication overlap must respect dependencies

DDP can reduce ready gradient buckets while later backward work executes. Complete required reductions before the optimizer updates parameters. FSDP prefetch can overlap gathers while increasing transient memory; current FSDP2 and older FSDP1 have different configuration APIs. See [framework communication](../NCCL-Deep-Dive/03-Framework-Integration.md).

Leave NCCL algorithm/transport tuning automatic for a baseline, record logs and isolate any override. Numeric `NCCL_NET_GDR_LEVEL=5`, forced Tree, fixed channel counts or hard-coded RoCE GIDs are not universal H200 recipes. [Configuration and controlled experiments](../NCCL-Deep-Dive/04-Configuration-and-Tuning.md).

PowerSGD and other compression methods approximate gradients; quality and convergence must be checked at equal training objectives/budgets. A low-rank storage ratio is not a guaranteed network or wall-time speedup.

## Precision is a separate quality experiment

BF16 autocast can improve throughput while retaining selected operations/state at higher precision. It does not guarantee stable numerics for all models. FP8 requires a supported runtime/Transformer Engine recipe, scaling and eligible operators. A generic float8 cast or enclosing an arbitrary model in a context does not prove its operations use valid FP8 kernels. Compare held-out quality, losses and outlier behavior with a higher-precision baseline.

## Diagnose with competing explanations

| Observation | Two hypotheses | Small distinguishing experiment |
|---|---|---|
| Low GPU utilization | Input starvation or little demand | Replay preloaded fixed inputs at controlled request rate |
| Large collective time | Network bottleneck or delayed rank arrival | Compare isolated collective sweep with all-rank application timeline |
| Low GEMM throughput | Poor shape/kernel or memory/launch bound | Sweep representative sizes and inspect selected kernel/traffic |
| Compile-time OOM | Transient compilation memory or excessive captured variants | Compare eager, one fixed shape, then shape buckets with peak memory logs |
| Larger batch improves tokens/s | Better reuse or more padded useless work | Count valid output tokens and deadline-qualified completions |

**Exit artifact:** one baseline trace, one falsifiable explanation, one controlled change, a numerical/quality regression check and an end-to-end result. Mark GPU results unexecuted if you only completed the CPU reference.
