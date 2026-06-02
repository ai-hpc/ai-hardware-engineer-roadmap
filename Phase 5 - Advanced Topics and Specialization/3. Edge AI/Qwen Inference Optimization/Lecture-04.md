# Lecture 4: Qwen2.5-72B-Instruct FP16 — Multi-GPU Inference

## Overview

Qwen2.5-72B-Instruct at FP16 is ~145 GB of weights. No single accelerator on the planet holds that in HBM. The model only runs by **splitting across multiple GPUs** — tensor parallel within a layer, optionally pipeline parallel across layers, all coordinated by NCCL collectives that sit in the decode hot path.

This lecture is the inference engineer's view of getting Qwen2.5-72B to serve at production quality on realistic boxes:

* 4 × H100 80 GB (NVL or SXM) — the sweet spot.
* 8 × A100 80 GB — common in 2024-vintage deployments.
* 8 × L40S 48 GB — cheap, no NVLink.
* 4 × MI300X 192 GB — increasingly common.

No quantization. Pure FP16 (the model ships BF16; we cast to FP16 for inference). The goal is correct partitioning, minimal collective overhead in decode, and serving high enough throughput that the answer doesn't matter — both single-stream latency and batched throughput.

By the end you should be able to:

* Compute exactly which slices of which tensors live on which GPU under tensor parallel.
* Predict NCCL bandwidth requirements per decoded token.
* Choose TP vs PP vs hybrid for a given (model, hardware, latency goal) triple.
* Diagnose where vLLM / SGLang / TRT-LLM is sitting on the roofline.

---

## 1. Footprint — Where Does Qwen2.5-72B Fit?

Per-tensor footprint at FP16:

| Tensor group | Per-layer FP16 bytes | Total (× 80 layers) |
|---|---|---|
| `attn_q.weight` (8192 × 8192) | 134 MB | 10.7 GB |
| `attn_k.weight` (8192 × 1024) | 16.8 MB | 1.3 GB |
| `attn_v.weight` (8192 × 1024) | 16.8 MB | 1.3 GB |
| `attn_o.weight` (8192 × 8192) | 134 MB | 10.7 GB |
| `attn_qkv.bias` | ~26 KB | 2 MB |
| `attn_norm.weight` | 16 KB | 1.3 MB |
| `ffn_gate.weight` (8192 × 29568) | 485 MB | 38.8 GB |
| `ffn_up.weight` (8192 × 29568) | 485 MB | 38.8 GB |
| `ffn_down.weight` (29568 × 8192) | 485 MB | 38.8 GB |
| `ffn_norm.weight` | 16 KB | 1.3 MB |
| Per-layer total | ~1.78 GB | **142.3 GB** |
| `token_embd.weight` (152064 × 8192) | — | 2.49 GB |
| `output.weight` (152064 × 8192, untied) | — | 2.49 GB |
| `output_norm.weight` | — | 16 KB |
| **Grand total** | | **~147.3 GB** |

KV cache (per token, all layers):

```
2 (K+V) × 80 layers × 8 KV heads × 128 head_dim × 2 bytes = 327 680 bytes
At 32 k context, single sequence:                            10.0 GB
At 131 k context, single sequence:                           40.0 GB
```

You need weights + KV + activations + CUDA contexts + NCCL buffers in aggregate VRAM:

| Box | Aggregate VRAM | Headroom after 147 GB weights | KV for batch=1 @ 32k | Notes |
|---|---|---|---|---|
| 8 × L40S 48 GB | 384 GB | 237 GB | trivial | No NVLink → PCIe-only collectives |
| 8 × A100 40 GB | 320 GB | 173 GB | trivial | NVLink 3, 600 GB/s |
| 4 × A100 80 GB | 320 GB | 173 GB | trivial | NVLink 3 |
| 4 × H100 80 GB | 320 GB | 173 GB | trivial | NVLink 4, 900 GB/s |
| 8 × H100 80 GB | 640 GB | 493 GB | trivial | NVLink 4 + NVSwitch |
| 4 × MI300X 192 GB | 768 GB | 621 GB | trivial | Infinity Fabric |

Even the smallest box (8 × L40S) has 237 GB free after weights. The constraint is **not capacity** — it's **bandwidth across the inter-GPU fabric** during decode.

---

## 2. Tensor Parallelism: Split Each Layer Across GPUs

Tensor parallel (TP) splits **each matrix multiply** across the GPUs that share the layer. Each GPU holds a **slice of the weights** and computes a slice of the output.

### 2.1 TP on QKV

Standard pattern (Megatron-style):

* **Q, K, V are sharded by head.** For Qwen2.5-72B with `n_heads = 64`, `n_kv_heads = 8`:

```
TP=2:  GPU0 gets 32 Q heads + 4 KV heads
       GPU1 gets 32 Q heads + 4 KV heads
TP=4:  each GPU gets 16 Q heads + 2 KV heads
TP=8:  each GPU gets  8 Q heads + 1 KV head   ← matches n_kv_heads exactly
TP=16: cannot evenly split 8 KV heads — KV gets replicated on some pairs
```

The natural TP degree is **whatever divides `n_kv_heads`** evenly. Qwen2.5-72B's 8 KV heads makes TP=8 the largest "clean" split. Above TP=8 you must duplicate KV across GPU pairs, which wastes VRAM but is still done in practice for latency reasons.

* **Output projection (W_O) is sharded by input.** Each GPU produces a partial output that gets summed across the TP group with **AllReduce**.

### 2.2 TP on FFN

* **Gate and Up are sharded by output (column-parallel).** Each GPU holds `intermediate / TP` columns. After SwiGLU, each GPU has a slice of the intermediate vector.
* **Down is sharded by input (row-parallel).** Each GPU multiplies its slice of the intermediate by its slice of W_down. Partials get summed across the TP group with **AllReduce**.

So per layer, you have **two AllReduce operations** (one after attention's W_O, one after FFN's W_down), each on a vector of size `d_model = 8192`.

### 2.3 NCCL bandwidth in the hot path

```
Per token, per layer:
  AllReduce post-attention:  8192 floats × 2 bytes = 16 KB
  AllReduce post-FFN:        8192 floats × 2 bytes = 16 KB
                                                   = 32 KB / layer
× 80 layers                                        = 2.56 MB / token
```

At 50 tok/s target this is 128 MB/s — trivially within NVLink-4 (900 GB/s per pair) or even PCIe Gen4 x16 (~32 GB/s).

**But** AllReduce has a latency floor. On NVLink-4 a 16 KB AllReduce takes ~10 µs. On PCIe-only (8 × L40S without NVLink), it can take 50–100 µs. Times 2 ops per layer × 80 layers:

* NVLink-4: ~1.6 ms latency floor per token
* NVLink-3 (A100): ~2.4 ms
* PCIe-only (L40S): 8–16 ms

**On NVLink boxes the collective latency floor caps you at ~600 tok/s.** Far above any actual throughput you'd hit. On PCIe-only L40S, the floor caps you at ~60-120 tok/s — which **can** matter when you're optimizing the rest of the stack hard.

### 2.4 Sequence parallelism for the activations

A subtle TP optimization: between the AllReduce points, each GPU only needs a slice of the activation tensor (since the next op is column-parallel). **Sequence parallelism** turns the AllReduce into an AllGather + ReduceScatter pair, which keeps each GPU's per-step activation memory proportional to `1/TP` rather than the full `d_model`.

For Qwen2.5-72B at long context with high batch, this is a real memory win. vLLM, TRT-LLM, and DeepSpeed-Inference all support it; whether they enable it by default is version-dependent. Worth checking.

---

## 3. Pipeline Parallelism: Split Layers Across GPUs

Pipeline parallel (PP) puts entire transformer blocks on different GPUs:

```
PP=2:  GPU0 = layers 0..39     GPU1 = layers 40..79
PP=4:  GPU0 = 0..19   GPU1 = 20..39   GPU2 = 40..59   GPU3 = 60..79
```

Pros:
- No collectives during the layer computation; only point-to-point handoff between adjacent stages.
- Each stage holds fewer weights — lower per-GPU memory pressure.

Cons:
- **Latency adds in series.** Decoding one token requires N stage roundtrips.
- For batch=1 decode, a pipeline is **strictly slower** than tensor parallel because there's no batch dimension to fill bubbles.
- The classic "fill the pipeline" only works with batch >> stages, which is throughput-mode serving.

**Recommendation:** for Qwen2.5-72B inference, use TP within a node, optionally PP across nodes. Never use PP within a single 4-or-8-GPU NVLink island unless you have a very specific reason (which usually turns out to be wrong).

---

## 4. Recipe Table for Production Boxes

| Hardware | Recommended TP/PP | Why |
|---|---|---|
| **4 × H100 SXM 80 GB** | TP=4 (one node) | Cleanest setup; weights ~37 GB/GPU; NVLink-4 fast collectives |
| **8 × H100 SXM 80 GB** | TP=8 | Lets KV head replication = 1; highest batch capacity |
| **8 × A100 80 GB SXM** | TP=8 | Same logic as H100; slightly slower collectives |
| **4 × A100 40 GB** | TP=4 | Tight — weights are ~37 GB/GPU, activations push you near limit at long ctx |
| **8 × L40S 48 GB** | TP=8 over PCIe + NCCL P2P enabled | Cost-effective but watch collective latency floor |
| **2 × node × 4 × H100** | TP=4 in-node, PP=2 across | Useful only for batch >> 16 |
| **4 × MI300X 192 GB** | TP=4 | ROCm + Infinity Fabric; vLLM-ROCm works as of 2026 |

---

## 5. Decoding One Token — The Annotated Wall Clock

Take TP=4 on 4 × H100 SXM, batch=1, ctx=2 k. Per-token wall clock breakdown (approximate):

```
Step                                  Time     What dominates
─────────────────────────────────────────────────────────────
Token embedding lookup                 5 µs    L2 cache hit
Per layer × 80:
  RMSNorm + residual                  10 µs    GPU compute
  QKV GEMV (fused, sharded)           45 µs    HBM bandwidth (W_QKV slice)
  RoPE on Q,K                          5 µs    GPU compute
  KV append + FlashAttention-decode   80 µs    HBM bandwidth (KV cache slice)
  AllReduce post-W_O                  10 µs    NVLink (small payload)
  RMSNorm + residual                  10 µs    GPU compute
  Gate+Up GEMV (fused, sharded)       95 µs    HBM (intermediate matrices)
  SwiGLU                               5 µs    GPU compute
  Down GEMV (sharded)                 95 µs    HBM
  AllReduce post-W_down               10 µs    NVLink
                                  ─────────
                                     365 µs / layer

× 80 layers                       =  29.2 ms
LM head GEMV (sharded)               850 µs    HBM (huge W_lm_head slice)
Softmax + sample                      30 µs
                                  ─────────
Total per token                  ≈   30.1 ms = 33 tok/s
```

That's the steady-state single-stream number you should expect from a competent runtime on H100. vLLM in mid-2026 reports ~30–38 tok/s on this configuration; TRT-LLM with custom AllReduce kernels has shipped ~42 tok/s in published benchmarks.

For **batch=8**, the cost per-token-per-stream stays nearly the same (you're bandwidth-bound on weights, which are reused across the batch). So aggregate throughput goes to ~250+ tok/s.

---

## 6. Continuous Batching and Paged Attention

The single biggest production runtime difference between "8-tokens-per-second" and "300-tokens-per-second" Qwen2.5-72B serving is **continuous batching with paged attention** (the vLLM paper, now table stakes).

### 6.1 Why static batching fails

Naive batching: collect a batch of N requests, run them together until the **slowest finishes**, then start the next batch. Two killers:

- Requests have wildly different completion lengths.
- Each batch step is bounded by the longest sequence's remaining tokens.

The result: GPU utilization < 30% on production traffic.

### 6.2 Continuous batching

Treat each request as a sequence of decoding steps. At each step, **insert new requests** into the batch as old ones finish. The forward pass operates on a "ragged" batch where sequences have arbitrary lengths and ages.

### 6.3 Paged attention

The naive KV-cache implementation allocates `[max_seq_len × n_kv_heads × head_dim]` per sequence up front — **wastes memory on short sequences**. Paged attention manages the KV cache as fixed-size **blocks** (typical: 16 tokens) and has a per-sequence **block table** mapping logical positions to physical blocks.

```
Logical KV for sequence A:  [block_3][block_7][block_9][block_2]
Logical KV for sequence B:  [block_4][block_1][block_8]

Physical block pool: { 1: ..., 2: ..., 3: ..., 4: ..., ... }
```

When a sequence finishes, its blocks go back into the pool. New sequences allocate blocks as needed. This gives near-100% KV-cache utilization in practice.

The kernel impact: FlashAttention-decode has to follow the block table during the K and V reads. There's a custom kernel variant for this — vLLM ships their own, TRT-LLM has its own, SGLang has its own.

---

## 7. Long Context — YaRN and Chunked Prefill

Qwen2.5-72B's native context is 32 k. With `rope_scaling: yarn`, it extends to 131 k.

### 7.1 YaRN at inference time

At inference, YaRN is "applied" by:
1. Computing rotary frequencies `theta_i = 1e6 ^ (-2i/d_head)` for `i = 0..d_head/2`.
2. **Scaling** those frequencies by a position-dependent factor that smoothly interpolates between native-range (no scaling) and extended-range (logarithmic scaling).
3. Applying RoPE with the scaled frequencies.

No additional weights, no inference-time cost. The runtime just needs to compute the right `cos / sin` table at startup.

**Catch:** runtimes that hard-code "max context = 32 k" without checking `rope_scaling` will silently truncate. Verify your config flag is `--rope-scaling yarn` or equivalent.

### 7.2 Chunked prefill

Prefilling a 100 k-token prompt as one GEMM uses ~100 GB of activation memory — too much. Split prefill into chunks of e.g. 2048 tokens, process sequentially, KV grows incrementally.

Implementation detail: the same kernel that does decode-step `seq_len=1` attention can be reused for prefill chunks — the chunk's queries attend to all of the KV accumulated so far. Generalized FlashAttention does this in one kernel form.

vLLM, SGLang, and TRT-LLM all do chunked prefill automatically. Whether they overlap it with ongoing decode steps (which improves serving TTFT) is a per-runtime decision.

---

## 8. Choosing a Production Runtime in 2026

| Runtime | Best at | Notes |
|---|---|---|
| **vLLM** | OpenAI-API-compatible serving, broad model support, AWQ/GPTQ | The default. Marlin kernels for quantized models. |
| **SGLang** | Programmatic generation control, structured output | Beats vLLM on cached-prefix workloads |
| **TensorRT-LLM** | Best raw throughput, NVIDIA-only, harder ops | Used at production scale (NVIDIA Triton-LLM integration) |
| **DeepSpeed-Inference (MII)** | Microsoft-stack integration | Faded since 2024 but still used in enterprise |
| **LMDeploy (TurboMind)** | Optimal for Qwen specifically — InternLM-team origin | Best out-of-box numbers on Qwen2.5-72B historically |
| **vLLM-ROCm / Aiter** | MI300X support | Catching up to CUDA path in throughput |

For Qwen2.5-72B specifically, the InternLM/Alibaba ecosystem has shipped optimized recipes for **LMDeploy** that consistently outperform other runtimes by 15–25% on Qwen models. If you're locked to Qwen and Nvidia, evaluate LMDeploy before committing to vLLM.

---

## 9. Practical Deployment Recipe — 4 × H100 SXM

```bash
# Pull the model
huggingface-cli download Qwen/Qwen2.5-72B-Instruct --local-dir ./qwen72b

# vLLM
docker run --gpus all --ipc=host -p 8000:8000 \
  -v $(pwd)/qwen72b:/model \
  vllm/vllm-openai:latest \
  --model /model \
  --tensor-parallel-size 4 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.92 \
  --enable-chunked-prefill \
  --rope-scaling '{"type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \
  --served-model-name qwen2.5-72b
```

Expected numbers from a clean vLLM deployment in mid-2026:

| Metric | Value |
|---|---|
| Single-stream decode | ~30 tok/s |
| Batch=8 aggregate decode | ~220 tok/s |
| Batch=32 aggregate decode | ~560 tok/s |
| TTFT @ 2 k prompt | ~250 ms |
| TTFT @ 32 k prompt (chunked) | ~2.5 s |
| Peak VRAM/GPU | ~75 GB (out of 80) |

If you're significantly off these, the usual suspects:
- **NCCL not using NVLink** — check `NCCL_P2P_LEVEL=NVL` and that `nvidia-smi topo -m` shows NVLink between all four GPUs.
- **Chunked prefill disabled** — explicitly enable.
- **CUDA Graphs disabled** in vLLM (`--enforce-eager` should NOT be set).
- **GPU not at full power** — `nvidia-smi -q -d POWER` should show ~700 W/GPU at saturation.

---

## 10. AMD MI300X Path

Brief because the user didn't ask, but it changes the playbook:

* **vLLM-ROCm** has caught up to ~85% of CUDA path throughput as of 2026.
* MI300X holds 192 GB HBM per GPU — Qwen2.5-72B FP16 fits in **one** GPU. Pure TP=1 inference works.
* For latency-sensitive serving, TP=2 across two MI300X still helps because per-GPU bandwidth is the limiter even on 192 GB cards.
* Infinity Fabric provides ~896 GB/s peer-to-peer — competitive with NVLink-4 in aggregate.

---

## Hands-On Exercises

1. **TP-degree sweep.** On a 4×H100 or 8×A100 box, run Qwen2.5-72B at TP=2, 4, and (if 8 available) 8. Measure single-stream decode and batch=16 aggregate decode. Confirm single-stream tok/s saturates near TP=8 and batch throughput scales nearly linearly with TP up to NCCL bandwidth limits.

2. **NCCL collective trace.** Run `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COLL python serve.py` and identify, for each decoded token, the size and count of AllReduce calls. Confirm they match 2 × n_layers × `d_model` × 2 bytes.

3. **Chunked vs unchunked prefill.** On a 32 k prompt, run with `--enable-chunked-prefill` and without. Measure TTFT and peak VRAM. Quantify the win.

4. **YaRN coherence test.** Generate a 100-token completion at the **end** of a 64 k-token prompt (with `--rope-scaling yarn`). Compare to the same generation at the end of a 4 k-token prompt. The 64 k case should still produce coherent, grammatical output. If it doesn't, your YaRN config isn't being applied — check the runtime flag.

5. **vLLM vs LMDeploy.** Deploy Qwen2.5-72B on the same hardware under both runtimes. Run an identical 5-minute load test (e.g., 50 concurrent users, mixed prompt lengths). Compare aggregate tok/s, p95 latency, and TTFT. Identify which workload class favors which runtime.

6. **Paged-attention KV utilization.** Watch `vllm metrics` (Prometheus endpoint) during a real load test. Confirm KV-cache utilization climbs to >90% with continuous batching. If it's < 50%, you're under-batched or paged attention is misconfigured.

---

## Key Takeaways

| Takeaway | Why it matters |
|---|---|
| 145 GB FP16 weights — TP is mandatory | No single GPU; this is a system-engineering exercise |
| TP=8 is the natural degree (n_kv_heads=8) | Cleanly partitions KV without replication |
| Two AllReduces per layer, on tiny tensors | NVLink latency floor matters more than bandwidth |
| PP only helps when batch >> stages | Within an NVLink island, TP wins |
| Continuous batching + paged attention is non-negotiable | 3-10× throughput multiplier over static batching |
| YaRN gives you 131 k context with no weight changes | Verify your runtime applies it |
| LMDeploy often beats vLLM on Qwen models specifically | Test both before locking in |

---

## Resources

* **[vLLM paper — Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180):** The continuous-batching + paged-attention reference.
* **[Megatron-LM TP paper](https://arxiv.org/abs/1909.08053):** Original tensor parallelism design.
* **[FlashAttention-2 paper](https://arxiv.org/abs/2307.08691):** Inference kernel reference.
* **[FasterTransformer / TRT-LLM kernel guides](https://github.com/NVIDIA/TensorRT-LLM):** Custom AllReduce and decode kernels for production deployments.
* **[NCCL Tuning Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html):** P2P configuration, algorithm selection.
* **[LMDeploy / TurboMind](https://github.com/InternLM/lmdeploy):** Qwen-optimized inference engine.
* **[YaRN paper](https://arxiv.org/abs/2309.00071):** The context-extension math.
* **[Qwen2.5 technical report](https://arxiv.org/abs/2412.15115):** Original release with config and benchmarks.
* **[vLLM-ROCm](https://github.com/ROCm/vllm):** AMD path.
* **[Phase 5 — GPU Infrastructure — NCCL Deep Dive](../../1.%20GPU%20Infrastructure/Nvidia%20GPU/HPC%20Setup/NCCL-Deep-Dive/README.md):** The companion deep dive on the collective layer.
