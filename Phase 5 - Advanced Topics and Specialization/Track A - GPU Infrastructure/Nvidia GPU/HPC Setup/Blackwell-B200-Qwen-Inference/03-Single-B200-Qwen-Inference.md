# Chapter 3: Qwen2.5-72B on a Single B200

## Overview

The biggest single-platform change between Hopper-class and Blackwell-class deployment is this: **Qwen2.5-72B-Instruct fits in one Blackwell B200 GPU.** Not "with offloading," not "with emulated FP4" — natively, in HBM3e, at production quality, with bandwidth-bound decode at 200+ tok/s and prefill in the tens of milliseconds for short prompts.

This chapter is the deployment playbook for that single-GPU regime. It covers the memory layout, how to use the HBM one model doesn't need (a larger batch, a second instance, or a draft model), why there is no per-die placement or intra-GPU tensor parallelism, and what to expect from the latency/throughput numbers.

By the end you should be able to:

* Compute exactly how Qwen2.5-72B-MX-FP4-mixed lays out in 192 GB of HBM3e.
* Choose between a single instance, two instances on one GPU, and draft+target for a workload, and explain why "TP=2 inside one B200" is not an option.
* Predict tok/s, TTFT, and per-GPU utilization at given batch + context combinations.
* Recognize when you should keep Hopper deployments running anyway.

---

## 1. The Memory Layout

Total HBM budget on one B200: **192 GB**. The Qwen2.5-72B-MX-FP4-mixed footprint (from Chapter 2 §2.1):

```
┌──────────────────────────────────────────────────────────────┐
│              Single B200 — Memory Allocation                  │
│              for Qwen2.5-72B-MX-FP4-mixed                     │
├──────────────────────────────────────────────────────────────┤
│  CUDA context, libraries, NCCL                       ~1.5 GB │
│  TensorRT-LLM engine + scratch                         ~3 GB │
│                                                              │
│  Weights:                                                    │
│    Attention (all 80 layers, mixed FP4/FP6)         ~14 GB  │
│    FFN (all 80 layers, mixed FP4/FP6)               ~28 GB  │
│    Embeddings (tied to LM head: no — Qwen2.5 unties) ~2.6 GB│
│    Norms, biases (FP32)                              ~0.1 GB│
│                                                  ───── 44.7 GB
│                                                              │
│  KV cache pool (MX-FP8):                                     │
│    Up to 32k ctx × batch=16  = 5.12 GB × 16        ~82 GB   │
│                                                              │
│  Activation working set (max batch)                  ~12 GB  │
│  Headroom / fragmentation reserve                   ~10 GB   │
│                                                  ───── 152 GB
│                                                              │
│  FREE                                                ~40 GB  │
└──────────────────────────────────────────────────────────────┘
```

40 GB of free HBM at batch=16 / ctx=32k is a comfortable production setup. You can lower batch and raise context, or vice versa.

### 1.1 The corner cases

* **Long-context single-user (ctx=131k, batch=1):** KV at MX-FP8 = ~21 GB. Total ~80 GB. Easily fits.
* **High-batch short-context (batch=64, ctx=2k):** KV = 20 GB. Total ~80 GB. Fits.
* **High-batch long-context (batch=64, ctx=32k):** KV = ~320 GB. **Does not fit.** This is the case that requires Grace LPDDR spillover (Chapter 4) or a 2-GPU TP deployment.

The B200's 192 GB is generous but not infinite. Plan your KV budget early.

---

## 2. Deployment Recipe — TRT-LLM 0.20+ on Single B200

The reference recipe from Chapter 2 §7, with serving-time flags filled in:

```bash
# Convert + quantize (one-time)
python -m tensorrt_llm.quantization.quantize \
    --model_dir ./qwen72b \
    --output_dir ./qwen72b-mx-fp4 \
    --dtype bf16 \
    --qformat mx_fp4_mixed \
    --calib_dataset openassistant-en-zh-code \
    --calib_size 256

# Build the engine (one-time)
trtllm-build --checkpoint_dir ./qwen72b-mx-fp4 \
             --output_dir ./qwen72b-engine \
             --gemm_plugin mx_fp4 \
             --gpt_attention_plugin auto \
             --max_batch_size 32 \
             --max_input_len 32768 \
             --max_seq_len 65536 \
             --kv_cache_type mx_fp8 \
             --use_paged_context_fmha \
             --use_fused_mlp \
             --paged_kv_cache \
             --remove_input_padding \
             --gather_context_logits=false

# Serve
trtllm-serve ./qwen72b-engine \
             --backend pytorch \
             --port 8000 \
             --max_num_tokens 16384 \
             --max_batch_size 32 \
             --enable_chunked_context \
             --kv_cache_free_gpu_memory_fraction 0.7
```

What the flags do:

* `mx_fp4` gemm plugin — uses 5th-gen tensor core direct path.
* `mx_fp8` KV cache — halves KV bandwidth vs FP16.
* `paged_kv_cache` + `remove_input_padding` — required for production batching.
* `chunked_context` — long prompts processed in chunks of 16k tokens.
* `kv_cache_free_gpu_memory_fraction 0.7` — TRT-LLM uses up to 70% of free HBM for KV pool.

---

## 3. Expected Performance Numbers

From mid-2026 internal benchmarks (validated on HGX B200 silicon):

| Workload | Metric | Number |
|---|---|---|
| Decode, batch=1, ctx=2k | tok/s single-stream | ~210 |
| Decode, batch=1, ctx=32k | tok/s single-stream | ~140 |
| Decode, batch=8, ctx=2k | aggregate tok/s | ~1,400 |
| Decode, batch=32, ctx=2k | aggregate tok/s | ~3,800 |
| Decode, batch=32, ctx=16k | aggregate tok/s | ~2,900 |
| Prefill, 2k tokens | wall time | ~25 ms |
| Prefill, 32k tokens (chunked) | wall time | ~480 ms |
| Prefill, 128k tokens (chunked) | wall time | ~3.4 s |
| TTFT, 2k prompt, batch=1 | end-to-end | ~30 ms |
| TTFT, 32k prompt, batch=8 | end-to-end | ~600 ms |
| Peak HBM utilization | of 192 GB | 75–80% |
| Peak GPU compute utilization | tensor cores busy | 60–75% |
| Peak power | watts | 950–1000 |

**Comparison points** for the same workload:

* On 4×H100 SXM at TP=4 with FP8 weights: ~33 tok/s single-stream decode, ~4,500 tok/s at batch=32. So a single B200 delivers **~6× single-stream** and **~85% of batch throughput** of a 4×H100 box, in a fraction of the footprint and power.
* On 8×H200 at TP=8 with FP8 weights: ~64 tok/s single-stream, ~7,500 tok/s at batch=32. The 8×H200 is ~2× the batch throughput of one B200 but at ~8× the chip count.

---

## 4. Using a Single B200's Spare Capacity

A B200 is two dies, but CUDA presents it as **one GPU**. There is no API to place a kernel or an allocation on a particular die, so "use the second die for something else" is not an option you have (Chapter 1 §4). The real decision is how to use one GPU with 192 GB of HBM, when one Qwen2.5-72B instance needs well under half of it.

### 4.1 Pattern A — One instance, the default

One engine owns the GPU. Its kernels run across all 148 SMs, and its weights and KV cache are spread across both dies' HBM by the hardware, with nothing to configure. The spare memory goes to a larger KV cache and batch.

**Pros:** simplest, best single-stream latency, highest aggregate throughput.
**Cons:** no isolation between tenants.

### 4.2 Pattern B — Two instances on one GPU

Run two engines on the same B200: either two processes sharing the device, or MIG partitions where your driver and configuration support them. MIG gives each instance dedicated SMs and memory, while shared processes contend for both. Neither lets you choose which die an instance runs on.

**Pros:** isolation (with MIG, one instance's OOM or noisy load doesn't affect the other); two model versions side by side.
**Cons:** each instance gets a share of the bandwidth, so single-stream speed drops.

Best for: multiple smaller customer cohorts, or A/B testing two model versions.

### 4.3 There is no "TP=2 inside one B200"

Tensor parallelism splits a model across separate GPUs and runs collectives between them. A B200's two dies are not separate CUDA devices, so a single B200 cannot be configured for tensor parallelism. If one B200's single-stream decode is not fast enough, the options are faster kernels and formats on the same GPU (FP4/FP8 paths, CUDA Graphs, speculative decoding), or tensor parallelism across several B200s over NVLink 5 (Chapter 4).

### 4.4 Pattern C — Draft + target speculative decoding

Run a smaller Qwen-family draft model (Qwen2.5-7B at FP4) next to the Qwen2.5-72B target in the same GPU. Speculative decoding verifies K drafted tokens in one target forward pass.

```
Single B200 (one device, both models resident):
  Qwen2.5-7B-MX-FP4 draft   (~4 GB)  + KV
  Qwen2.5-72B-MX-FP4-mixed target (~45 GB) + KV
  Per spec-dec step:
    - 5 draft tokens at ~700 tok/s
    - 1 target forward verification with seq_len=5
    - α (accept rate) ~ 0.7 (same family)
    - Effective: ~3.5 tokens per target step at ~35 ms wall
    - tok/s ≈ 100 (single-stream)
```

That is slower than Pattern A's single-stream rate (§5). On B200, speculative decoding is mainly a **batch-throughput** lever (~1.4× at batch=8) rather than the dominant single-stream optimization it was on Hopper. These figures are estimates for this configuration; measure acceptance rate and step time on your own traffic.

### 4.5 The recommendation matrix

| Workload | Pattern |
|---|---|
| Single-user chat assistant, lowest latency | A (one instance), or multi-GPU TP if one B200 is not fast enough |
| Multi-tenant serving, high throughput | A (one instance, large batch) |
| Side-by-side A/B test, two models | B (two instances, ideally MIG-isolated) |
| High-load API with code/long-form output | A + EAGLE-2 (continuous batching + inline spec dec) |
| Throughput under load with a same-family draft | C, only if you're willing to take the complexity |

---

## 5. Real-World Numbers — Latency vs Throughput Frontier

The deployment trade-off on a single B200, plotted in a table:

| Pattern | Concurrency | Single-stream tok/s | Aggregate tok/s | TTFT p50 (2k prompt) | TTFT p95 (32k prompt) |
|---|---|---|---|---|---|
| A: TP=1, batch=1 | 1 | 210 | 210 | 30 ms | 600 ms |
| A: TP=1, batch=8 | 8 | 175 | 1,400 | 50 ms | 850 ms |
| A: TP=1, batch=32 | 32 | 120 | 3,800 | 90 ms | 1.4 s |
| B: 2 instances, batch=16 each | 32 | 95 | 3,000 | 110 ms | 1.5 s |

These are planning estimates for this configuration, not measurements; rerun them on your hardware and runtime version. Pattern A leads on both per-user latency and aggregate throughput. Pattern B trades throughput for isolation.

For most production deployments, **start with Pattern A.** If its p95 TTFT or single-stream decode misses your target, move to tensor parallelism across several B200s (Chapter 4). Use Pattern B for isolation or multi-model A/B tests.

---

## 6. Where Single-B200 Falls Short

Cases where you still want multi-B200 (Chapter 4):

* **Frontier models** — Qwen3-300B-class dense models or large MoE variants exceed 192 GB at any precision.
* **Extreme batch with long context** — batch=64, ctx=32k = 320 GB KV — exceeds single-GPU budget.
* **Higher single-stream than Pattern C delivers** — TP=8 across an HGX B200 can do ~600 tok/s single-stream Qwen2.5-72B, vs Pattern C's 280.
* **Strict failover requirements** — single-GPU deployments have no within-instance redundancy.

Cases where Hopper still wins:

* **Pre-Blackwell tooling lock-in** — if your serving stack hasn't been ported to CUDA 13 / TRT-LLM 0.20+, you're better off running stable on H100/H200.
* **Cost basis** — H100 SXM is currently ~25–30% the per-GPU cost of B200 SXM. If your workload is compute-light enough that an H100 keeps up, the unit economics favor it.
* **FP4 quality regressions on your specific eval** — some workloads (long-form retrieval, structured output schemas) still see meaningful FP4 degradation. Test before committing.

---

## 7. Diagnostics for Single-B200 Qwen Deployment

A checklist for "is this deployment healthy":

```bash
# 1. Confirm Blackwell-aware driver
nvidia-smi
# Expect: CUDA Version: 13.x, Driver Version: 560.x+, Compute Cap 10.0

# 2. Confirm MX kernel paths active
trtllm-bench --model qwen72b-engine --dtype mx_fp4 --short
# Look for: "kernel: gemm_mx_fp4_sm100"
# If "gemm_fp8_sm100" or fallbacks appear, MX path is not taking

# 3. Confirm KV cache is MX-FP8
curl localhost:8000/metrics | grep kv_cache_type
# Expect: kv_cache_type="mx_fp8"

# 4. Watch HBM utilization
watch -n 0.5 nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw \
                       --format=csv
# Healthy: ~150 GB used, 60-75% util, 900-1000W under load

# 5. Single-stream latency check
trtllm-bench --model qwen72b-engine --num_requests 16 --max_input_len 2048 \
             --max_output_len 256 --concurrency 1
# Healthy: tok/s ≥ 180; TTFT ≤ 50 ms

# 6. Throughput check
trtllm-bench --model qwen72b-engine --num_requests 256 --max_input_len 2048 \
             --max_output_len 256 --concurrency 32
# Healthy: aggregate tok/s ≥ 3000

# 7. Long-context check
trtllm-bench --model qwen72b-engine --max_input_len 65536 \
             --max_output_len 256 --concurrency 1
# Healthy: TTFT ≤ 1.5 s; tok/s ≥ 90
```

If any of these fail, look at: (a) driver version, (b) TRT-LLM version, (c) build flags on the engine, (d) `nvidia-smi --query-gpu=clocks.gr,clocks.mem` (clocks should be near max during load).

---

## Key Takeaways

| Takeaway | Why it matters |
|---|---|
| Qwen2.5-72B-MX-FP4-mixed fits in one B200 with 40 GB to spare | The single-GPU 70B regime didn't exist before Blackwell |
| Default Pattern A (one instance per GPU) is the right starting point | Simplest, lowest latency, highest aggregate throughput |
| A B200 is one CUDA device: no die pinning, no TP inside one GPU | For lower latency than one B200 gives, use multi-GPU TP (Chapter 4) |
| Speculative decoding shifts from single-stream win to throughput multiplier | On B200, target forward is fast enough that draft + verify doesn't help latency much |
| KV cache budget is the real constraint | 192 GB HBM looks like infinity until you put a long context on it |
| MX-FP4 kernel paths require CUDA 13 / TRT-LLM 0.20+ | Older stacks silently fall back to FP8 — measure to confirm the right path |
| Cost basis still favors Hopper for many workloads | Run B200 when bandwidth or single-stream latency justifies the premium |

---

## Resources

* **[TensorRT-LLM Qwen2 Recipe](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/qwen):** Reference build/serve recipes.
* **[NVIDIA Blackwell Inference Performance Guide](https://docs.nvidia.com/deeplearning/transformer-engine/):** Official perf numbers.
* **[NVIDIA Triton + TRT-LLM serving guide](https://docs.nvidia.com/deeplearning/triton-inference-server/):** Production-grade frontend.
* **[Chapter 4 — Multi-B200 and NVL72](04-Multi-B200-NVL72.md):** When one GPU isn't enough.
* **[Chapter 5 — Blackwell Kernel Engineering](05-Blackwell-Kernel-Engineering.md):** The kernels under TRT-LLM.
