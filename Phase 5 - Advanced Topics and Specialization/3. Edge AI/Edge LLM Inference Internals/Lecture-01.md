# Lecture 1: Edge LLM Inference Internals — GEMV Decode, QKV Projections, and the Memory-Bandwidth Wall

## Overview

When a Jetson-class edge device generates tokens from an LLM, almost everything you care about — latency, power, thermal — is decided by a single, deeply unglamorous operation: **GEMV** (general matrix–vector multiply) over **quantized weights**. Datacenter folklore tells you to think in TFLOPS and tensor cores; that folklore actively lies about what happens on an Orin Nano. Edge LLM decode is **memory-bandwidth-bound**, not compute-bound, and the silicon engineer's job is to **move fewer bytes per token**, not to add more MACs.

This lecture takes a single observation — a runtime log line like

```
[GEMV #0] type=12 M=2560 K=2560
[embd]   Q6_K dequant token 9707
```

— and unpacks it into the full mental model: what GEMV is, why it dominates decode but not prefill, what the `M=K=2560` shape says about the model, where QKV projections sit inside that arithmetic, why quantization formats like Q4_K / Q6_K exist, and what concrete Jetson knobs determine whether you get 0.04 tok/s or 40 tok/s.

By the end you should be able to:

* Read a llama.cpp / mlc-llm / ggml GEMV trace and identify which transformer block each line belongs to.
* Compute the per-token byte traffic for a given model and explain why decode is bandwidth-bound.
* Place QKV projections, FFN gate/up/down, and the LM head on a roofline plot for Orin Nano vs Orin AGX vs H100.
* Diagnose "GPU at 0 MHz" failures with `nvpmodel`, `jetson_clocks`, `tegrastats`, and CUDA profiling.
* Choose between Q4_K_M, Q5_K_M, Q6_K, AWQ, GPTQ, and FP16 *based on bandwidth*, not just perplexity.

---

## 1. GEMM vs GEMV — Why Decode Is a Different Animal

A transformer forward pass is a chain of linear projections sandwiched between attention and FFN. Mathematically the same op runs in both prefill and decode:

```
y = W · x
```

What changes is **`x`**:

| Phase | Shape of `x`        | Op type | Arithmetic intensity (flops/byte) |
|---|---|---|---|
| Prefill (process prompt) | `[seq_len × d]`     | **GEMM** | High — ~`2·seq_len` |
| Decode (generate 1 token) | `[1 × d]`          | **GEMV** | Low — ~`2`             |

Arithmetic intensity is the deciding number. On Orin Nano (Ampere, 1024 CUDA cores, ~50 GB/s LPDDR5 bandwidth), the **roofline knee** sits around 30–40 flops/byte. GEMM during prefill clears that bar by an order of magnitude and lives in the compute-bound regime. GEMV during decode is stuck below 2 flops/byte — it lives in the **bandwidth-bound** regime, where adding tensor cores does literally nothing.

The practical consequence: a Jetson Orin Nano with 50 GB/s of DRAM bandwidth, decoding a 7 B-parameter model at INT4 (~3.5 GB of weights), can in principle do

```
50 GB/s / 3.5 GB/token = ~14 tokens/s
```

That is a **hard ceiling**. You only beat it by moving fewer bytes per token (smaller model, lower-bit quant, KV-cache compression, weight reuse across decoded tokens via speculative/parallel decoding).

---

## 2. Reading a GEMV Trace Line by Line

A line like:

```
[GEMV #0] type=12 M=2560 K=2560
```

decodes as:

| Field      | Meaning                              | What you do with it |
|---|---|---|
| `GEMV`     | matrix × vector kernel               | confirms decode path, not prefill |
| `type=12`  | quantization kernel id (Q4_K, Q5_K, Q6_K…) | tells you weight bits-per-element |
| `M=2560`   | output dim (rows of W, length of y)  | model's hidden size / projection out |
| `K=2560`   | input dim (cols of W, length of x)   | model's hidden size in |

For a `d_model = 2560` decoder-only LLM (the shape used by Phi-2, Stable LM 3B variants, etc.), each transformer block emits this sequence of GEMVs per generated token:

```
hidden ─► RMSNorm ─► [GEMV] Q  : 2560×2560
                   ├► [GEMV] K  : 2560×2560     (× num_kv_heads / num_heads if GQA)
                   └► [GEMV] V  : 2560×2560     (same)
                              │
                     Rotary embed + KV cache append
                              │
                       attention(Q,K,V)
                              │
                   [GEMV] O proj : 2560×2560
                              │
              residual add ─► RMSNorm
                              │
                ├► [GEMV] FFN gate : 2560×~6912
                └► [GEMV] FFN up   : 2560×~6912
                              │
                        SwiGLU activation
                              │
                   [GEMV] FFN down  : 6912×2560
                              │
                      residual add
```

Multiply by `n_layers` (≈ 32 for a 3 B-class model). Then add one final **LM head** GEMV — usually `2560 × vocab_size`, which on a 32 k vocab can be the *single largest* op of the entire forward pass.

So the *count* of GEMVs per token is roughly `7 × n_layers + 1`. For a 32-layer model, that's **225 GEMVs per token**. Every one of them re-reads its weight matrix from DRAM. That is why bandwidth dominates.

---

## 3. QKV Projections in Detail

The first three GEMVs of every layer are the QKV projections. Conceptually:

* **Query (Q)** — "what am I looking for?"
* **Key (K)** — "what do I offer to be matched against?"
* **Value (V)** — "what content do I carry if matched?"

Mathematically:

```
Q = x · W_Q
K = x · W_K
V = x · W_V
```

Then attention:

```
attn = softmax( Q · Kᵀ / √d_k ) · V
```

### 3.1 Fused QKV — the standard runtime trick

Three separate GEMVs over the same input `x` read `x` three times and pay three kernel-launch latencies. Real runtimes fuse them:

```
[W_Q | W_K | W_V]    # concatenated along output dim
        │
        ▼
    QKV = x · W_QKV    # single GEMV, output split into Q,K,V slices
```

On Jetson GPUs this is a clear win — one global-memory read of `x`, one launch, contiguous output. llama.cpp does this when the model export packs the matrices; mlc-llm does it by default via its Relax IR fusion pass.

The trace gives this away: if you see one GEMV of shape `M = 3·d, K = d` instead of three of shape `M = d, K = d`, your runtime fused. That is what you want.

### 3.2 GQA, MQA — when K and V get smaller

Modern LLMs (Llama 3, Mistral, Gemma, Phi-3) use **Grouped-Query Attention (GQA)** or **Multi-Query Attention (MQA)** to shrink the K and V projections:

| Variant | Q heads | K/V heads | Output dim of W_K, W_V |
|---|---|---|---|
| MHA (vanilla) | `n_heads` | `n_heads`           | `d` |
| GQA           | `n_heads` | `n_heads / g` (e.g. 8) | `d · n_kv / n_heads` |
| MQA           | `n_heads` | `1`                 | `d_head` |

This is purely a **bandwidth** trick. The Q projection stays full size; K and V shrink by the GQA ratio. The KV cache (per token, per layer) shrinks correspondingly — which matters more for long-context decode than the projection itself.

### 3.3 Shared / removed projections — research, not deployment

The "KV Transformer" (drop Q, reuse K) and "K Transformer" (single projection used as Q, K, and V) variants exist in research. They cut bandwidth further at the cost of expressivity. As of ICLR 2026 they are not in production deployments — but they are exactly the kind of change a hardware engineer should track, because each removed projection is one fewer GEMV per layer per token.

### 3.4 PAMM and training-time compression

Recent work like **Projected Attention Memory Mapping (PAMM)** compresses QKV activations during *training* by up to 512×. This is a **training-memory** story, not an inference one — your inference traces are unaffected. But it indirectly enables larger models trained on the same hardware, which then *do* land on your edge device with more parameters per byte of training cost.

---

## 4. Quantization Formats — What `type=12` Actually Means

Edge LLM runtimes use heavily quantized weights. The kernel id you see in the trace maps to a specific block format. For ggml / llama.cpp:

| `type` id | Name      | Bits/weight (effective) | Block size | Notes |
|---|---|---|---|---|
| 0  | F32       | 32 | — | reference only |
| 1  | F16       | 16 | — | "FP16 baseline" |
| 8  | Q8_0      | 8.5 | 32 | per-block FP16 scale |
| 10 | Q4_0      | 4.5 | 32 | symmetric INT4 |
| 12 | Q4_K      | ~4.5 | 256 | K-quant family — superblocks w/ shared scale |
| 13 | Q5_K      | ~5.5 | 256 |  |
| 14 | Q6_K      | ~6.5 | 256 | near-lossless for many models |
| 8+ | IQ-family | 1.5–4 | varied | "i-quants", non-uniform codebook |

The K-quants (Q4_K, Q5_K, Q6_K) use a two-level **superblock + sub-block** layout: one FP16 scale per superblock of 256 weights, plus smaller per-sub-block scales packed into a few bytes. The compute kernel does:

```
for each block of 256 weights:
   load packed weights + scales
   dequant on-the-fly into FP16 or FP32 registers
   FMA into accumulator
```

Dequant happens **in registers**, not in DRAM. That is the whole point — you read 4.5 bits per weight from DRAM and synthesize 16-bit values on chip. Bandwidth wins.

**Why this matters for your trace:** `type=12` (Q4_K) reads `~0.56 bytes/weight`; `type=14` (Q6_K) reads `~0.81 bytes/weight`. The same GEMV will be ~44% slower at Q6_K vs Q4_K on a bandwidth-bound device, perplexity be damned. Choose the lowest-bit format that meets your quality bar — usually Q4_K_M for 7B-class, Q5_K_M for 3B-class — and only escalate to Q6_K if you actually see regression on your eval set.

---

## 5. The KV Cache — The Other Memory Story

Per generated token, attention re-reads **the entire KV cache so far**:

```
KV bytes per layer per token = 2 · n_kv_heads · d_head · bytes_per_elem
```

For a Llama-3-8B-class model (n_layers=32, n_kv_heads=8, d_head=128) at FP16:

```
2 · 8 · 128 · 2 = 4096 bytes/layer/token
× 32 layers     = 131 072 bytes per cache slot
× context len   (e.g. 4096) = 537 MB
```

That is **on top** of the ~5 GB of weights. On an 8 GB Orin Nano, long-context decode runs you out of LPDDR very quickly, and the runtime falls back to slower paths. KV cache compression — INT8/INT4 KV, paged attention, sliding-window, sparse retrieval — is the second axis of bandwidth optimization.

In your trace, you do *not* see explicit "KV read" lines. They are folded into the attention kernel. But they show up clearly when you compare measured bandwidth against the "weights only" prediction — the gap is KV traffic.

---

## 6. Roofline for Edge LLM Decode

A back-of-envelope roofline for three platforms decoding a 7B Q4_K_M model (~3.5 GB weights, ~14 GFLOPS/token of useful compute):

| Platform | Peak FP16 TFLOPS | DRAM BW (GB/s) | Knee (flops/byte) | Decode regime | Practical tok/s ceiling |
|---|---|---|---|---|---|
| Orin Nano 8 GB        | ~10              | ~50            | ~200              | bandwidth-bound | ~14 |
| Orin NX 16 GB         | ~25              | ~100           | ~250              | bandwidth-bound | ~28 |
| Orin AGX 64 GB        | ~50              | ~200           | ~250              | bandwidth-bound | ~57 |
| H100 SXM              | ~989             | ~3 350         | ~295              | bandwidth-bound | ~957 |

Two things jump out:

1. **Even H100 is bandwidth-bound for single-stream decode.** That is why datacenters batch aggressively — batching turns GEMV back into GEMM and crosses the roofline knee.
2. **On Jetson you cannot batch.** Edge use cases (assistants, on-device RAG) decode one sequence at a time. So bandwidth is destiny. Your only levers are: smaller model, lower-bit weights, smaller KV.

---

## 7. Diagnosing a Slow Trace on Jetson

A real example from the wild:

```
Prefill: 6 tokens in 145 547 ms
Decode:  4 tokens in 102 662 ms     →  ~0.04 tok/s
Power:   0W mode, GPU @ 0 MHz
```

That decode rate is two orders of magnitude below the roofline. The "GPU @ 0 MHz" tells you immediately: the GPU is **not running**. Either you're on CPU fallback, or DVFS has parked the clocks, or you're stuck in `nvpmodel` mode 0 (15 W cap) without `jetson_clocks` locking the frequencies up.

### 7.1 First-pass diagnostic checklist

```bash
# 1. Lock to max performance
sudo nvpmodel -m 0          # max-N power mode (or -m 2 for MAXN_SUPER on JP6)
sudo jetson_clocks          # pin GPU/CPU/EMC clocks high
sudo jetson_clocks --show

# 2. Watch real-time utilization during inference
tegrastats --interval 500

# Look for:
#   GR3D_FREQ NN%       <- GPU 3D engine load — should be > 80% during decode
#   EMC_FREQ NN%        <- memory controller — should be near 100% if you're bandwidth-bound
#   CPU [...]           <- if CPU is high and GR3D is low, you're on CPU fallback

# 3. Confirm CUDA is actually used
nvidia-smi               # works on AGX; not all Orin variants
# or
sudo /usr/local/cuda/extras/demo_suite/deviceQuery

# 4. Profile the runtime
nsys profile -t cuda,nvtx,osrt -o llm_decode ./your_runtime
nsys-ui llm_decode.nsys-rep
# Look at the timeline: are there big gaps between kernels?
#                        Are kernels < 50 µs each? That's launch overhead dominating.
```

### 7.2 Common edge-LLM pathologies

| Symptom | Likely cause | Fix |
|---|---|---|
| `GR3D_FREQ 0%`, CPU pegged | CPU fallback; CUDA backend not compiled in | rebuild with `LLAMA_CUDA=1` or `-DGGML_CUDA=ON` |
| `EMC_FREQ < 50%`, GPU mid | weights in pageable memory, hitting unified-memory thrash | pin / use cudaMallocHost or mmap with `MADV_HUGEPAGE` |
| Many tiny CUDA launches | unfused QKV, unfused gate/up | switch to runtime build with fusion (llama.cpp ≥ b3000, mlc-llm) |
| Sub-roofline at high `GR3D_FREQ` | kernel occupancy too low for given shapes | tune block size, or switch quant format (Q4_K kernels often outperform Q4_0) |
| Long pauses every N tokens | KV cache reallocation | preallocate full `n_ctx` at start, never grow |
| Throughput drops at high context | KV bandwidth > weight bandwidth | INT8 KV or paged attention |

---

## 8. Where Optimization Actually Lives

The bandwidth wall is broken at four layers. Each is a real engineering specialty:

### 8.1 Model layer — fewer bytes by design

* GQA / MQA over MHA.
* Sliding-window attention (Mistral) or chunked attention to bound KV.
* Tied embeddings (LM head shares weights with token embedding) — removes one huge final GEMV's worth of unique weights.
* Mixture-of-Experts only if you have the routing memory budget; usually a bad fit at the 8 GB scale.

### 8.2 Numerical layer — fewer bytes per weight

* Q4_K_M as a default starting point for 7B-class.
* AWQ / GPTQ for 4-bit with calibration-based weight reordering — better perplexity than uniform Q4 at the same bit-rate.
* W4A8 / W4A16 split: weights at 4 bits, activations at 8 or 16 bits.
* INT8 / INT4 KV cache.

### 8.3 Kernel layer — fewer reads per byte

* **Fused QKV**, **fused gate+up** (SwiGLU), **fused dequant+matmul** (the K-quant kernels already do this).
* Persistent kernels that hold weight tiles in shared memory across multiple decoded tokens — only practical with speculative or parallel decoding.
* Warp-specialized loaders on Ampere/Ada (Orin is Ampere — limited but real benefit).
* On CPU paths: ARM NEON / SVE for dequant, with carefully-aligned 64-byte reads.

### 8.4 System layer — fewer fetches from DRAM

* Pin weights in unified memory and disable the GPU's L2 cache eviction for hot tiles (where the runtime exposes it).
* On JP6, use `MADV_HUGEPAGE` for the mmapped weight file.
* `jetson_clocks` to keep EMC at max — DVFS will throttle EMC under "0 W idle" heuristics if you let it.
* Avoid host↔device copies; on Jetson the GPU and CPU share physical DRAM, so zero-copy is free if you use it correctly (`cudaHostAllocMapped` or unified memory with the right hints).

---

## 9. Hands-On Exercises

1. **Build a roofline plot for your Jetson.** Use the bandwidth-test (`bandwidthTest` from CUDA samples) and a single-shape GEMM benchmark to find the measured peak FP16 TFLOPS and measured DRAM BW. Compute the knee. Then run llama.cpp at Q4_K_M, Q5_K_M, Q6_K, F16 for the same model and place each on the plot. Confirm decode points cluster on the bandwidth slope; identify any that fall below (look for an `nvpmodel`/`jetson_clocks` issue).

2. **GEMV trace dissection.** Run llama.cpp with `LLAMA_LOG_LEVEL=DEBUG` (or `--verbose` on builds that expose it) on a model whose architecture you know. Save the trace for a single generated token. Annotate each GEMV with the transformer block name (Q, K, V, O, gate, up, down, LM head). Compute total bytes read; predict tokens/sec; compare to observed.

3. **Quant ladder bandwidth measurement.** Take the same base model and quantize to Q4_0, Q4_K_M, Q5_K_M, Q6_K, F16. Measure tokens/sec on the same Jetson at the same prompt. Plot tok/s vs effective bytes/weight. The relationship should be roughly linear with a constant — that constant is your achievable bandwidth, and the slope tells you whether you're losing efficiency at a specific quant format (often Q4_0 underperforms because its kernels are less optimized than Q4_K).

4. **Fused QKV check.** Export the same model with and without QKV fusion (`mlc-llm` lets you toggle this in the compile pass). Compare kernel counts in Nsight Systems and end-to-end tok/s. Quantify the win on Orin Nano vs Orin AGX — fusion matters more where launch overhead is a larger fraction of kernel time.

5. **KV cache compression study.** Run a 4 k-context decode at FP16 KV vs INT8 KV (where your runtime supports it — e.g. mlc-llm, vLLM with CUDA-capable backend). Measure tok/s near token 1, 1 k, 2 k, 4 k. Plot. You should see FP16 KV degrade gracefully then steeply; INT8 KV should hold flatter for longer. Identify the cross-over point.

6. **Diagnose a sandbagged run.** Boot Jetson in `nvpmodel -m 1` (10 W cap), do *not* run `jetson_clocks`, and run an inference. Capture `tegrastats`. Then progressively (a) run `jetson_clocks`, (b) switch to `nvpmodel -m 0`, (c) ensure the runtime is CUDA-built. At each step record tok/s. You will get a four-row table that is more persuasive than any blog post about why edge LLM perf is "slow".

---

## 10. Key Takeaways

| Takeaway | Why it matters for AI hardware |
|---|---|
| Decode = GEMV; prefill = GEMM | Two different rooflines, two different engineering problems |
| Edge decode is bandwidth-bound, full stop | Designs and benchmarks centered on TFLOPS lie about this workload |
| QKV is the first three GEMVs of every layer | Fusing them is the single highest-ROI runtime change |
| Quant format = bytes/weight = decode speed | Pick the lowest bit-rate that meets quality, not the highest you can fit |
| The KV cache is the *second* bandwidth story | Long-context decode is bottlenecked there, not on weights |
| `nvpmodel` + `jetson_clocks` are not optional | Default DVFS will silently halve your tok/s |
| Roofline reasoning beats vibes | A 5-minute calculation tells you whether a fix is worth weeks of work |

---

## Resources

* **[ggml / llama.cpp source](https://github.com/ggerganov/llama.cpp):** The reference for K-quant kernels, GEMV implementations, and the quant-format `enum` you saw as `type=12`.
* **[mlc-llm](https://llm.mlc.ai/):** Compiler-driven LLM runtime built on TVM/Relax; cleanest place to study fused QKV and fused SwiGLU at IR level.
* **["Roofline: An Insightful Visual Performance Model" (Williams, Waterman, Patterson, 2009)](https://dl.acm.org/doi/10.1145/1498765.1498785):** Foundational paper — still the right mental model for LLM decode.
* **["GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"](https://arxiv.org/abs/2210.17323):** The 4-bit calibration story.
* **["AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"](https://arxiv.org/abs/2306.00978):** Often outperforms GPTQ at the same bit-rate; mobile-friendly.
* **["GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"](https://arxiv.org/abs/2305.13245):** The K/V reduction trick used by Llama 3, Mistral, Gemma.
* **["Efficient Streaming Language Models with Attention Sinks"](https://arxiv.org/abs/2309.17453):** Sliding-window KV without quality collapse.
* **[NVIDIA Jetson Linux Developer Guide — `nvpmodel` & `jetson_clocks`](https://docs.nvidia.com/jetson/archives/r36.2/DeveloperGuide/):** The canonical reference for power modes and clock locking.
* **[`tegrastats` man page / Jetson Stats `jtop`](https://github.com/rbonghi/jetson_stats):** Live GPU/EMC/CPU utilization on Jetson.
* **[Nsight Systems on Jetson](https://docs.nvidia.com/nsight-systems/):** The profiler that actually shows you kernel-level gaps on Orin.
* **[Phase 4 — Jetson Real-Time Inference guide](../../../Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/1.%20Nvidia%20Jetson%20Platform/Orin-Nano-Real-Time-Inference/Guide.md):** Companion deep dive on Jetson Orin Nano inference.
* **[Phase 4 — Track C — Quantization guide](../../../Phase%204%20-%20Track%20C%20-%20ML%20Compiler%20and%20Graph%20Optimization/DL%20Inference%20Optimization/04%20-%20Quantization/Guide.md):** Compiler-side view of the quant formats discussed here.
