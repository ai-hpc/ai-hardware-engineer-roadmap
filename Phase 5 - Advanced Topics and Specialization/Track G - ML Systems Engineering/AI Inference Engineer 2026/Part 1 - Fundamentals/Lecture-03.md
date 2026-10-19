# Part 1 · Lecture 03 — Use a roofline as a bound

A roofline answers: “Given this amount of work and data movement, what resource could limit performance?” It does not predict a model's measured latency by itself. The [CPU accounting lab](labs/README.md) checks units and boundary cases without pretending to simulate a GPU.

## Count at one boundary

Let `F` be FLOPs and `Q` be bytes transferred across a stated memory boundary, such as HBM↔processor. Arithmetic intensity is `I=F/Q` FLOPs/byte. With compatible compute ceiling `P` FLOP/s and bandwidth ceiling `W` byte/s:

```text
attainable FLOP/s <= min(P, W * I)
execution time >= max(F/P, Q/W)
ridge point I* = P/W
```

These are optimistic bounds under the chosen ceilings. Launches, dependencies, synchronization, insufficient parallel work, quantization conversions, communication, and queueing can increase time. If stages cannot overlap, add their durations instead of taking one global maximum.

Use dense versus structured-sparse rates correctly. Match datatype, accumulator mode, hardware SKU, and operation. A sparse FP8 Tensor Core peak is not a ceiling you can directly substitute for a dense BF16 elementwise kernel. GB/s means `10^9` bytes/s; GiB means `2^30` bytes. Convert units before division.

## Work a dense decode example

Suppose a hypothetical dense model has exactly 70 billion parameters, all stored as BF16, and one decode step must read each weight once from HBM. Weight bytes alone are 140 billion. On an H100 SXM quoted at 3.35 TB/s, the weight-transfer bound is:

```text
140e9 bytes / 3.35e12 bytes/s = 0.04179 s = 41.79 ms
```

This is a **single-device, weights-only bound under those assumptions**; the weights do not fit in an 80 GB H100. It is useful arithmetic, not a feasible deployment or a benchmark. Sharding over devices changes both capacity and communication, while batching reuses weights across multiple output tokens. Weight compression changes payload and adds scale/dequantization work.

If one matrix has `K×N` weights and serves `M` token rows, GEMM work is about `2MKN`. Ignoring activations/output and assuming each weight is read once, intensity is approximately `2M / weight_element_bytes`. For BF16 this is `M` FLOPs/byte. Including all reads and writes lowers that estimate; reuse in caches can change HBM traffic. Small `M` does not prove every kernel is bandwidth-bound.

## Separate memory levels

Registers, shared memory, caches, and HBM have different capacities and access rules. Registers are per-thread values; excessive live values can spill to memory. Shared memory is explicitly managed per block/cluster and requires correct synchronization. L1/L2 behavior depends on access pattern. A measured HBM roofline does not explain a shared-memory bank conflict or instruction dependency by itself.

For Hopper, the [tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/) documents 64K 32-bit registers per SM, a maximum 255 registers per thread, a combined 256 KB L1/texture/shared pool, and up to 228 KB shared memory per SM. These are distinct quantities; assigning the whole pool to one block is incorrect. [H100 specifications](https://www.nvidia.com/en-us/data-center/h100/) distinguish SXM/NVL capacities and sparse-rate footnotes. Record the actual SKU.

## Design a diagnostic experiment

Sweep batch size while holding context length fixed, then sweep context while holding batch fixed. Measure end-to-end latency and use a profiler to attribute kernel and communication time. Compare a bandwidth-sensitive copy/stream benchmark with relevant GEMM shapes; peak specification rates are useful outer bounds, not expected sustained rates.

For each important kernel record shape, dtype, counted FLOPs, actual or estimated traffic, duration, and the boundary used. An arithmetic estimate can say “consistent with bandwidth pressure.” A profiler can test whether memory pipelines are busy and whether other causes dominate. Low bandwidth utilization may mean dependency/launch limitations rather than unused easy bandwidth.

**Exercise.** A kernel performs 1 billion FLOPs and transfers 2 billion bytes. Assume 10 TFLOP/s and 100 GB/s. **Answer:** intensity 0.5 FLOP/byte; compute bound 0.1 ms; transfer bound 20 ms; roofline lower bound 20 ms. A measured 30 ms is plausible. A measured 2 ms means a count, boundary, unit, or assumption is wrong.

**Exercise.** Removing 80% of a stage that was 10% of total latency improves total latency from 100 to 92 units: only `100/92 ≈ 1.087×`. This is why isolated kernel wins need whole-service measurement.

Next: [precision and quantization](Lecture-04.md).
