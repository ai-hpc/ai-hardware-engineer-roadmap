# Part 1 · Lecture 02 — Transformer Execution, From Tokens to Bits

## Overview

Inference engineering rests on one mechanical insight: a modern decoder-only transformer running inference does **two completely different jobs** in two completely different cost regimes, joined by one data structure (the KV cache) that grows monotonically and dominates everything at long context.

The two jobs are **prefill** and **decode**. They have the same forward pass on paper. They have entirely different bottlenecks in practice. Every optimization in Parts 2 and 3 of this course either reduces the cost of one of them, or trades one against the other.

This lecture builds the model from the bottom:

1. The two-phase forward pass of a decoder-only transformer.
2. The KV cache anatomy — shape, growth, bytes-per-token math.
3. Where the FLOPs go vs where the bytes go — the compute / bandwidth split.
4. The three regimes — compute-bound, memory-bound, scheduler-bound — and how to diagnose which one you are in.
5. A lab that puts numbers on all of the above for one specific model + GPU.

By the end you should be able to derive, from a model card alone, why a 70B model decodes at ~30 tokens/sec on H200 — and what each lever (batch size, precision, speculation) would do to that number.

---

## 1. The two phases — prefill and decode

A decoder-only transformer generating text autoregressively does two operations:

### 1.1 Prefill — process the prompt

```text
input: prompt of P tokens
output: hidden states for all P positions
        + populated KV cache (K and V tensors for every layer, every position)
```

Concretely, for each of the L transformer layers:

> **RMSNorm** rescales each token vector to a stable magnitude before every sub-layer (pre-norm). It is elementwise, bandwidth-bound, and costs ~0.5 FLOP/B — cheap individually but called 160 times across 80 layers per token. For the full intuition (what RMS measures, why it replaces LayerNorm, and what ε does) see [Part 2 → Lecture 01 §1.1](../Part%202%20-%20Dense%20at%20Hopper/Lecture-01.md#11-rmsnorm-what-it-actually-does).

```text
x ── RMSNorm ─► QKV projection ──► (Q, K, V each of shape [P, h_q × head_dim] / [P, h_kv × head_dim])
                                    │
                                    ├── K and V stored into KV cache at positions 0..P-1
                                    └── attention(Q, K_cache, V_cache) → output projection → residual
            ── RMSNorm ─► gate/up projections → SiLU(gate) * up → down projection → residual
```

Key cost shape:

* **All P positions of the prompt are processed in one matrix multiplication per layer per projection.**
* The Q/K/V matmul is `(P, d) @ (d, h × head_dim)` — a real GEMM, with arithmetic intensity that scales with P.
* The attention is a `(P, P)` softmax with a `(P, head_dim)` output — quadratic in P.
* The FFN gate/up/down matmuls are `(P, d) @ (d, d_ff)` and `(P, d_ff) @ (d_ff, d)` — by far the FLOP-heaviest stage at long P.

Prefill is **compute-bound** for any prompt longer than a few hundred tokens. Tensor cores hit close to peak. This is the regime that benefits from **FP8 / FP4 throughput** on Hopper / Blackwell.

### 1.2 Decode — emit one token at a time

```text
loop while not done:
    take previous output token
    for each layer:
        x ── RMSNorm ─► QKV projection (only for *current* token, single row)
                        ├── K and V appended to KV cache at position P+t
                        └── attention(Q, K_cache_so_far, V_cache_so_far) → single row out
        FFN on single row
    sample next token
```

Key cost shape:

* Every projection is now `(1, d) @ (d, h × head_dim)` — a **GEMV** (matrix-vector), not a GEMM.
* The weight matrices are large (gigabytes) and *fully re-read from HBM* for each token. There is essentially no FLOP-per-byte to keep tensor cores busy.
* Attention reads the *entire* KV cache (length P+t) for each new token. **KV-read bandwidth** grows linearly with context.

Decode is **bandwidth-bound** at batch=1 — almost all of the time is **HBM read** of the weights + KV. This is why decode TPOT tracks HBM bandwidth, not FLOPs.

### 1.3 The mental picture

```text
prefill:                         decode:
┌──────────────────────────┐     ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐
│   P×d × d×4d FFN GEMM    │     │1│ │1│ │1│ │1│ │1│  ← batch=1 GEMV per step
│  compute-bound, fp8 wins │     └─┘ └─┘ └─┘ └─┘ └─┘
└──────────────────────────┘      ▲   ▲   ▲   ▲   ▲
                                  │   │   │   │   │
                                 bandwidth-bound: HBM read of W
                                 every step + growing KV cache
```

Two completely different optimization problems wearing the same code path.

### 1.4 Why batching helps decode

If you decode for `B` concurrent requests at the same step, the weight read is **shared**:

* Without batching: B × (read W + 1 matmul) = B weight reads
* With batching: 1 × read W + B-row matmul (still bandwidth-bound but amortized)

Effective throughput scales **nearly linearly with B** until the matmul reaches the compute ceiling or the KV cache fills HBM. This is the entire reason **continuous batching** (vLLM's headline feature) was the breakthrough of 2023: it raises decode throughput by 10–50× without changing the model.

---

## 2. The KV cache — anatomy

The KV cache is the **load-bearing data structure** of decoder-only inference. Understand it cold.

### 2.1 What it stores

For each layer, the K and V tensors produced by every past token. Shape:

```text
K_cache: [batch, num_kv_heads, seq_len, head_dim]
V_cache: [batch, num_kv_heads, seq_len, head_dim]
```

Stored *across all L layers*.

### 2.2 Bytes per token, per request

The formula a senior can derive on a whiteboard:

```text
kv_bytes_per_token = 2 (K and V)
                   × L (layers)
                   × num_kv_heads
                   × head_dim
                   × bytes_per_element
```

Worked examples:

**Llama 3.3 70B** — L=80, num_kv_heads=8, head_dim=128, FP16 (2 bytes):

```text
2 × 80 × 8 × 128 × 2 = 327,680 bytes/token ≈ 320 KB/token
```

**Qwen 2.5 72B** — L=80, num_kv_heads=8, head_dim=128, FP16 — **identical** to Llama 3.3:

```text
2 × 80 × 8 × 128 × 2 ≈ 320 KB/token
```

Both share **GQA** with 8 KV heads and the same depth → identical KV cost per token. This is *not* a coincidence; it is the GQA configuration both teams converged on. We will return to this in [Part 2 Lecture 01](../Part%202%20-%20Dense%20at%20Hopper/Lecture-01.md).

**DeepSeek V3.1 with MLA** — Multi-head Latent Attention compresses the KV cache by storing a small *latent* vector instead of full K/V. The bytes-per-token math is **dramatically lower**. We will compute it in Part 3.

### 2.3 KV cache cost at scale

At long context the KV cache becomes the dominant HBM consumer:

| Context length | Llama 3.3 70B FP16 KV | FP8 KV | INT4 KV |
|----------------|------------------------|---------|----------|
| 4,096 | 1.3 GB | 0.7 GB | 0.3 GB |
| 16,384 | 5.2 GB | 2.6 GB | 1.3 GB |
| 65,536 | 21 GB | 10.5 GB | 5.3 GB |
| 131,072 | 42 GB | 21 GB | 10.5 GB |

**Per request.** At batch=16 with 32K context, the KV cache alone is ~168 GB at FP16 — more than double H100 80 GB capacity; even batch=8 (~84 GB) already blows past it, forcing FP8 KV or smaller batch. This is why **long-context serving forces precision drops on KV** before it forces them on weights.

### 2.4 The bandwidth half of the KV cache

Beyond capacity, the KV cache is *read* at every decode step:

```text
KV bandwidth per decode step = 2 (K and V) × L × h_kv × head_dim × seq_len × bytes_per_element
                             = kv_bytes_per_token × seq_len
```

For Llama 3.3 70B at 64K context, FP16 KV: 320 KB × 65,536 ≈ **20 GB per decode step**. On H200 (4.8 TB/s HBM3e) that read alone is **~4 ms** — before any weight matmul, before any sampling. This is why **decode TPOT grows with context length** even though FLOPs do not.

The fix is either KV precision drop (FP8 cuts the read time in half) or sparse / sliding-window attention (read fewer KV tokens).

---

## 3. Where the FLOPs go vs where the bytes go

For a transformer step, you can decompose cost into three columns:

| Stage | FLOPs scaling (decode batch=1) | HBM bytes read scaling |
|-------|--------------------------------|------------------------|
| QKV projection | O(d²) | O(d²) (read W) |
| Attention | O(seq × d) | O(seq × kv_size_per_token) |
| Output projection | O(d²) | O(d²) |
| FFN gate + up | O(d × d_ff) | O(d × d_ff) (read W) |
| FFN down | O(d_ff × d) | O(d_ff × d) (read W) |

The total cost per decode step at batch=1:

* **FLOPs ≈ 2 × P** where P is parameter count.
* **HBM bytes ≈ model_size_in_bytes + kv_read_bytes** ≈ P × bytes_per_param + seq × kv_bytes_per_token.

The ratio (FLOPs / bytes) is the **arithmetic intensity**. For **decode at batch=1** it is approximately:

```text
arithmetic_intensity = 2 × P / (P × bytes_per_param) = 2 / bytes_per_param
```

So at FP16 (2 bytes): **1 FLOP per byte read**. At FP8 (1 byte): **2 FLOPs per byte**. At FP4 (0.5 bytes): **4 FLOPs per byte**.

Compare to the hardware ridge point (FLOPs/byte ceiling above which you become compute-bound):

| GPU | Peak FP16/BF16 TFLOPs | HBM bandwidth (TB/s) | Ridge point (FLOPs/byte) |
|-----|------------------------|----------------------|--------------------------|
| H100 SXM | 989 | 3.35 | ~295 |
| H200 SXM | 989 | 4.80 | ~206 |
| B200 SXM | 2,250 | 8.0 | ~280 |

Decode arithmetic intensity (1–4 FLOPs/byte) is **two orders of magnitude below** the ridge point. Decode is **bandwidth-bound. Always. Until you batch.**

**With batching B**, the FFN matmul becomes a `(B, d_ff)` × `(d_ff, d)` operation: arithmetic intensity scales roughly linearly with B until the GEMM is large enough to saturate tensor cores. This is the lever that **moves decode from bandwidth-bound to compute-bound** — typically around B = 64–256 for 70B-class models on Hopper.

---

## 4. The three regimes — diagnosis

Given a slow inference workload, **the first job is to identify which regime you are in.**

### 4.1 Compute-bound

* Tensor cores at high utilization (>70% in Nsight Compute).
* HBM bandwidth at moderate utilization (<50%).
* Improves with: lower precision (FP8 → FP4), larger tensor core kernels, kernel fusion.
* Does *not* improve with: more HBM, faster HBM.

**Smell test:** prefill at sequence ≥ 512. Embedding inference. Heavily batched decode (B ≥ 128).

### 4.2 Memory-bound (bandwidth-bound)

* Tensor cores at low utilization (<30%).
* HBM bandwidth at high utilization (>80% of peak).
* Improves with: precision drop on the *read-heavy* tensors (weights, KV), batching (amortize the read), prefix cache (skip reads).
* Does *not* improve with: more FLOPs.

**Smell test:** decode at low concurrency. Memory-bound regime is where most chat traffic lives by default.

### 4.3 Scheduler-bound

* Tensor cores **and** HBM both at low utilization.
* GPU shows lots of bubbles in the timeline.
* Improves with: better batching, lower per-request overhead, fewer kernel launches, CUDA Graphs, faster Python-side path.
* Does *not* improve with: precision drops, faster HBM, more tensor cores.

**Smell test:** very short decodes (1–10 tokens), small batches, agent loops with rapid request churn. Often dominated by Python overhead.

### 4.4 Quick diagnostic table

| Symptom | Likely regime | First experiment |
|---------|---------------|------------------|
| Decode TPOT scales linearly with model size | Memory-bound | Try precision drop on weights |
| Decode TPOT scales linearly with context | Memory-bound on KV | Try FP8 KV cache |
| Decode TPOT does not improve with batch | Already compute-bound *or* scheduler-bound | Profile to disambiguate |
| Prefill TFLOPs/s low even with long prompt | Suspect kernel selection or precision | Switch FP8 / check kernel |
| Both bandwidth and FLOPs low | Scheduler / Python overhead | Try CUDA Graphs, larger batches |

---

## 5. Lab — put numbers on this for one model + one GPU

Goal: extend the benchmark harness from Lecture 01 with phase-aware measurements.

1. **Pick one model** (continue with Qwen3-4B from Lecture 01 if you have it).
2. **Add `--measure prefill` and `--measure decode` modes** to your harness:
   * **Prefill mode:** feed prompts of varying length (128, 512, 2048, 8192 tokens) with `max_new_tokens=1` and measure GPU-time per stage with CUDA events. Report TFLOPs/s achieved.
   * **Decode mode:** feed a 128-token prompt with `max_new_tokens=128`, batch sizes (1, 4, 16, 64), and measure per-token decode latency.
3. **Compute the bandwidth ceiling** from the model card + GPU spec. Plot achieved decode tokens/sec against the ceiling.
4. **Profile one decode step** with Nsight Systems. Identify (a) % time spent in attention, (b) % in FFN, (c) % in Python overhead between kernels.
5. **Compute and plot arithmetic intensity** for each kernel against the GPU ridge point. Annotate which regime each kernel is in.

Pass criterion: you can show a chart of decode throughput (tokens/sec) vs batch size with the bandwidth ceiling overlaid, and explain why the curve flattens at the batch size where it does.

---

## Self-check

1. A 70B-class dense model decoding at batch=1 on H200 hits 32 tokens/sec at FP16. You quantize weights to FP8 (1 byte/param). Without re-running, predict the new TPOT. Why might the actual measurement undershoot your prediction by 20–30%?
2. Llama 3.3 70B has 80 layers, 8 KV heads, head_dim 128. You are serving at 64K context, FP8 KV, batch=8. How many GB does the KV cache occupy? Will this fit alongside an INT4-quantized weight matrix on an H200 141 GB?
3. An agent product has p99 TPOT spiking to 200 ms at random. p50 is stable at 30 ms. Tensor cores show <10% utilization on Nsight. Which regime, and what is your first experiment?
4. You batch 32 decode requests together and TPOT *increases* from 35 ms to 70 ms per token. What two things are most likely wrong?
5. For DeepSeek V3.1 (MoE with 37B active params, MLA attention), how does the bandwidth-bound decode ceiling differ from a 70B dense model on the same GPU? Sketch the math.

---

## References

* "Reducing Activation Recomputation in Large Transformer Models" — [arXiv:2205.05198](https://arxiv.org/abs/2205.05198) — activation memory math
* "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision" — [arXiv:2407.08608](https://arxiv.org/abs/2407.08608)
* "Efficient Memory Management for Large Language Model Serving with PagedAttention" — [arXiv:2309.06180](https://arxiv.org/abs/2309.06180) — the KV cache as the bottleneck
* NVIDIA Nsight Systems documentation — [docs.nvidia.com/nsight-systems/](https://docs.nvidia.com/nsight-systems/)
* NVIDIA Nsight Compute documentation — [docs.nvidia.com/nsight-compute/](https://docs.nvidia.com/nsight-compute/)

Cross-references:

* [Phase 5 → Edge AI → Edge LLM Inference Internals](../../../3.%20Edge%20AI/Edge%20LLM%20Inference%20Internals/Lecture-01.md) — GEMV vs GEMM
* [Phase 5 → Edge AI → Qwen Inference Optimization → Lecture 03 — Decode Optimization on Jetson](../../../3.%20Edge%20AI/Qwen%20Inference%20Optimization/Lecture-03.md) — bandwidth-bound decode on the edge end

---

## Current as of 2026-06

Math pinned at FP16 / FP8 / INT4 / FP4 with the precision sizes given; ridge-point numbers from NVIDIA's published H100 / H200 / B200 datasheets. Update if NVIDIA publishes corrected peak numbers or if a new precision lands (FP6, FP3).

---

## Next

* Next: [Lecture 03 — Roofline, bandwidth, and the memory hierarchy](Lecture-03.md)
* Previous: [Lecture 01 — The 2026 inference engineer's mental model](Lecture-01.md)
* Up: [Part 1 — Fundamentals](README.md)
