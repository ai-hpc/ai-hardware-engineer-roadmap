# Lecture 3 — FA1 traversal and the meaning of its IO bound

**Course:** [FlashAttention](Guide.md) · **Reviewed:** 18 October 2026

**Goal:** relate an algorithm's loop order to memory traffic without confusing words, bytes or hardware counters. **Artifact:** an annotated traversal and a tested traffic estimate.

## 1. Read the historical algorithm accurately

The original FA1 Algorithm 1 has an **outer loop over K/V column blocks** and an inner loop over Q row blocks. It loads a K/V block, revisits Q/output/statistic blocks, and updates the running result. The [paper's Algorithm 1 and IO analysis](https://arxiv.org/html/2205.14135v2#S3.SS2) define the specific schedule and memory model.

A query-outer implementation is also a valid tiled attention schedule. Our NumPy lab keeps one query block's numerator/statistics while streaming all K/V blocks, then stores that output. Do not label this loop order a verbatim transcription of FA1 Algorithm 1. Later implementations change scheduling and parallelism while preserving the attention function.

## 2. Derive the bound with consistent units

In the theoretical model, SRAM capacity `M` is measured in stored scalar elements. A K/V tile has width on the order of `M/D`; there are roughly `ND/M` K/V tiles. Revisiting `ND` query/output elements for each tile gives the term:

```text
(ND/M) × ND = N²D²/M element accesses
```

This reasoning applies under the paper's stated capacity regime. Converting to bytes requires the relevant element widths; real kernels mix low-precision operands with wider accumulators and statistics. Substituting “100 KB” directly as an element count and reporting a precise byte total is dimensionally wrong.

## 3. Analyze the lab's different, explicit schedule

For query length L, key length S, query/key dimension D, value dimension Dv, element size b and query row tile Br:

```text
query_tiles = ceil(L/Br)
Q and O bytes = L × (D + Dv) × b
K and V bytes = query_tiles × S × (D + Dv) × b
LSE bytes = L × stats_bytes
```

This counts logical reads/writes under a no-cache model. Tail tiles count only valid elements. The column tile changes scratch size and scheduling, but partitioning the same full K/V pass into smaller pieces does not by itself change this formula.

Run `attention_contract.py` at sequence lengths 4096 and 8192 with Br fixed at 64. The dominant term nearly quadruples. The tests require this behavior. There is no promise of linear HBM traffic or a byte-counter-to-profiler agreement within an invented percentage.

## 4. Budget the live tile

A useful first sketch includes Q, K, V, output accumulators, score/probability tiles, per-row statistics and pipeline buffers. For one query tile, storage includes terms like `Br×D`, `Bc×D`, `Bc×Dv`, `Br×Dv`, and `Br×Bc`. Multiply each by its actual storage width and account for overlapping lifetimes before adding them.

Registers, shared memory and compiler temporaries have different constraints. A tile that fits an abstract SRAM sum may still exceed register limits or reduce occupancy too far. A second buffering stage increases storage while potentially hiding latency.

## 5. Connect the estimate to a profiler

For an optional GPU run, query the installed profiler's available metrics first:

```bash
ncu --query-metrics
ncu --set basic --target-processes all -o attention-profile python your_selected_gpu_experiment.py
```

Replace the script with a tested selected path and distinguish its warmups/reference calls from the target range before interpreting results. Metric names and availability vary by architecture. Kernel DRAM traffic, L2 traffic and logical tensor bytes are different quantities; caches can make the measured DRAM total smaller than the logical model.

**Exercise:** doubling Br halves the repeated K/V term approximately. Does it guarantee half the runtime? **No.** It can increase live storage, change residency, padding and scheduling, or move the limiting resource. Submit the model's prediction and a separate outcome field marked measured or not run.
