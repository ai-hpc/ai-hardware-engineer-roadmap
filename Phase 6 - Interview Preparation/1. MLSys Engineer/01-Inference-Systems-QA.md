# MLSys Engineer — Inference Systems Q&A

**Collection:** [MLSys Engineer](README.md) · **Up:** [Interview Preparation](../README.md)

These are reasoning exercises, not accounts of work the reader has shipped. Numerical examples below are calculations with explicit assumptions, not measured product benchmarks. In an interview, separate your own measured results from estimates and published results. Reconstruct each answer, then defend the assumptions against one changed constraint.

## Q1. How does PagedAttention work?

It stores a sequence’s KV cache in separately allocated blocks and uses a block table to recover logical token order inside the attention kernel. This reduces the need for a maximum-length contiguous allocation and enables prefix sharing with appropriate reference counting and copy-on-write. The idea is inspired by virtual memory; software KV blocks are not OS page-table mappings. See the [PagedAttention paper](https://arxiv.org/abs/2309.06180).

**Derive an example:** With block size 16, a 35-token sequence uses three blocks with capacity 48. Thirteen slots are unused in the last block. That bound applies to one sequence’s final partially filled block; allocator metadata, multiple cache groups, reserved capacity, and sharing policy still affect the total.

**Implementation question:** Which thread owns each reference count, and what prevents an in-flight attention kernel from reading a block after eviction? Draw allocation → publication → use → completion → release. A memory-saving scheme is incomplete without that lifetime proof.

**Transfer to a VLA:** Cache reuse must include image history, language prompt, model revision, action-conditioning state, and positional semantics in its validity conditions. Reusing a text prefix does not make visual observations interchangeable.

## Q2. How would you integrate EAGLE-3 speculative decoding?

First distinguish versions. EAGLE-3 uses multi-layer target features and direct token prediction; describing it only as predicting the target’s next top-layer hidden state confuses it with earlier feature-prediction approaches. The target, trained drafter, feature taps, tokenizer, and vocabulary mapping must match. Use the [EAGLE-3 paper and linked implementation](https://arxiv.org/abs/2503.01840) for those contracts.

An integration needs feature extraction, draft generation, target verification, token acceptance, and KV rollback/retention. A tree verifier must enforce causal ancestry and correct positions; arbitrary attention-mask overrides are not guaranteed by every runtime backend. Pin the backend release and read its supported path before choosing a plugin boundary.

Greedy verification and stochastic verification are different. Greedy acceptance compares the draft path with target choices. Exact stochastic speculative sampling uses the target and proposal probabilities with an acceptance/rejection correction, not merely “keep the longest matching random sample.” See the [speculative sampling derivation](https://arxiv.org/abs/2211.17192).

**Lab:** On a tiny categorical model, test that a deliberately poor drafter changes cost and acceptance but preserves the target distribution under the exact sampler. For a deployed model, record accepted tokens, target calls, rollback bytes, and wall-clock time. A draft model that sounds accurate can still be too expensive.

## Q3. How do you budget and optimize KV cache?

Start with the architecture and supported runtime layout. For uniform full-attention layers:

```text
KV bytes = 2 × L × B × T × Hkv × D × s
           K/V  layers batch tokens KV-heads head-dim bytes/element
```

**Worked calculation:** For `L=24, B=1, T=4096, Hkv=8, D=128, s=2`, payload is 402,653,184 bytes = 384 MiB. INT8 would make the raw payload 192 MiB, before scales, alignment, and metadata. If K/V dimensions or cache ownership differ by layer, sum each actual allocation; do not force a uniform formula onto it.

Quantization, shorter retained context, and prefix reuse change different contracts. Quantization changes numerical error; context eviction changes visible history; reuse requires identity and lifetime checks. GQA is a model architecture choice, not a runtime switch that can be enabled on arbitrary pretrained MHA weights.

**Lab:** Predict allocation from configuration, compare it with runtime counters, and explain the difference. Evaluate representative long contexts and VLA action outputs after quantization. INT8 error is not inherently bounded by an FP16 ULP: scale, outliers, clipping, and downstream sensitivity matter.

## Q4. How would you write a fused attention kernel?

Write the reference first: shapes, scaling, mask, bias, GQA mapping, accumulator dtype, and all-masked-row behavior. Tile scores and update a numerically stable softmax without writing the full score matrix to external memory. Track a running row maximum, normalization denominator, and weighted value accumulator; rescale prior contributions when the maximum grows.

Work partitioning matters after the algorithm is correct: assign rows/tiles to blocks and warps, choose registers versus shared storage, and inspect occupancy and synchronization. [FlashAttention-2](https://arxiv.org/abs/2307.08691) specifically studies work partitioning and non-matmul overhead; a familiar algorithm name is not enough to predict performance.

**Validation matrix:** Test ragged lengths, unequal query/key lengths, causal offsets, sliding windows, large logits, all-masked rows, GQA, and supported head dimensions. Compare with a high-precision reference using absolute and relative error, then the end-to-end model metric. Test asynchronous buffer reuse and race detection separately from numerical error.

**Design review:** Increasing tile size may reduce re-reads while increasing registers and shared memory enough to reduce parallelism. Present measured counters for both choices, with the same input and output tolerance.

## Q5. Why does FlashAttention reduce external-memory traffic?

A naive implementation materializes scores and probabilities. For one head and `N=4096`, an FP16 `N×N` matrix occupies 32 MiB. Four whole-matrix passes—write scores, read scores, write probabilities, read probabilities—account for 128 MiB. Multiply by heads and batch explicitly; this is not automatically 4 GB per layer.

FlashAttention keeps score/probability tiles on chip and avoids those full intermediates. It still re-reads tiles when finite SRAM prevents retaining everything. Linear auxiliary storage does not imply linear total I/O. In the original paper’s model, with SRAM capacity `M` elements, the tiled algorithm has an I/O term proportional to `N²d²/M` over the analyzed range. Exact dense attention also retains quadratic arithmetic. See [FlashAttention](https://arxiv.org/abs/2205.14135).

**Lab:** Vary sequence length while holding head dimension and dtype fixed; collect allocated memory, measured DRAM bytes, and latency. Explain why those three curves differ. Repeat with another tile configuration before claiming a compute or bandwidth limit. On Jetson the external memory is shared LPDDR, not discrete-GPU HBM.

## Q6. When is speculative decoding worthwhile on an edge device?

Let `A` be the expected number of emitted tokens per iteration, including any bonus token under the chosen algorithm. Let `D`, `V`, and `O` be draft, verify, and scheduling/rollback time, and `t` the ordinary target time per token:

```text
speedup estimate = A × t / (D + V + O)
beneficial when  = A × t > D + V + O
```

For an illustrative `A=3, t=10 ms, D=5 ms, V=12 ms, O=1 ms`, speedup is `30/18 ≈ 1.67`. If acceptance falls to `A=1.5`, the same system is slower (`15/18 ≈ 0.83`). These are arithmetic examples, not hardware measurements.

On a shared-memory SoC, draft and target contend with camera, CPU, and other device traffic. Measure sustained performance, memory peak, power mode, temperature, and task quality at realistic concurrency. Datacenter deployments can also share memory resources; an interconnect does not automatically isolate draft-model bandwidth.

**VLA boundary:** A continuous flow/diffusion action head is not an autoregressive text decoder. Do not apply token speculation to it without an algorithm and correctness argument for that action representation. Optimize the measured action path.

## Q7. How would you reduce time to first token or first action?

Define start and end events before optimizing. Warm request TTFT can include ingress, queueing, tokenization, prefill, first-token decoding, and delivery. Cold process startup additionally includes runtime initialization, model loading, and possibly compilation. Overlapped stages require a timeline, not a blind sum.

For robotics, measure observation-capture → first usable action and observation age at actuation. A faster text TTFT may leave the camera encoder or action head unchanged.

**Experiment order:** Measure one warm baseline; measure a fresh-process baseline; break out queue and device time; then optimize the dominant stage. Candidates include keeping a process resident, supported prefix reuse, batched/chunked prefill, removing redundant copies, and compiling stable shapes. Do not encode changing instructions into an adapter and call that lossless prompt compression.

A mapped file plus `cudaHostRegister` is not a universal fast path: platform support, page faults, cacheability, and access patterns matter. Check [CUDA for Tegra memory guidance](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html) and compare against explicit/managed allocations on the target.

**Pass condition:** Your change improves the chosen percentile under the same workload, quality threshold, thermal state, and concurrency; report any startup/memory tradeoff.

## Q8. How would you port a model to a new inference backend?

Build a contract from the checkpoint configuration, reference implementation, and preprocessing/postprocessing. List every operator, shape range, layout, dtype, positional rule, cache update, weight transformation, and action normalization. Pin source and checkpoint revisions; a family name is not an executable specification.

Implement in stages:

1. Load and validate configuration and weight shapes; reject unknown architectural features.
2. Reproduce one block against the reference before exporting a whole graph.
3. Map supported operators; add a plugin only for a documented unsupported operation.
4. Test dynamic shapes, cache/state transitions, and resets in the runtime.
5. Compare intermediates to locate numerical regressions; assess task quality and action validity.
6. Profile after correctness, retaining a reproducible baseline and rollback path.

TensorRT, TensorRT-LLM, and a separate edge runtime have different APIs and support boundaries. Use the [TensorRT-LLM source/documentation for the pinned release](https://github.com/NVIDIA/TensorRT-LLM) rather than assuming a generic attention plugin accepts every architecture.

**Custom-chip extension:** Turn the supported operator subset into a compiler legality table, define the runtime command/buffer ABI, and compare a software model of the accelerator with the same golden tensors. For VLA deployment, retain closed-loop or replay evaluation in addition to a few greedy text prompts.

## Quick calculations to defend at a whiteboard

**Decode matrix-vector intensity.** For `W[K,N]` and FP16 operands, an idealized single-use calculation gives `2KN / [2(KN+K+N)]` FLOP/byte, approaching **1**, not 2, for large `K,N`. Different weight/activation/output dtypes require separate byte terms; cached reuse changes the denominator. A roofline is a ceiling, not proof of the observed bottleneck.

**GQA cache saving.** Reducing KV heads from 32 to 8 cuts that cache payload by four. It does not guarantee four times faster end-to-end inference because weights, compute, and other traffic remain.

**Q8_0 storage.** In ggml, one Q8_0 block stores 32 signed 8-bit values plus one 16-bit scale: `(32+2)×8/32 = 8.5` bits/value before file-level overhead. Q4_K_M is a mixed quantization policy, not one uniform 4-bit tensor type. Inspect the actual tensor types and sizes in the checkpoint. Source: [ggml block definitions](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h).

**Final defense:** Bring your own benchmark manifest, one failed optimization, a corrected equation, and the tensor/buffer contract used by the next engineer. That evidence connects this interview course back to the [physical AI roadmap](../../README.md).
