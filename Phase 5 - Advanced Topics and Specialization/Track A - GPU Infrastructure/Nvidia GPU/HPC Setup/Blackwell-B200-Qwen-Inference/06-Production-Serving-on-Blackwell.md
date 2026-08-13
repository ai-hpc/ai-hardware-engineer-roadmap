# Chapter 6: Production Serving of Qwen on Blackwell

## Overview

You have an HGX B200 box. You have a Qwen2.5-72B-MX-FP4-mixed engine that benchmarks well. Now you need to serve a thousand concurrent users at p99 < 500 ms and not have a 3 a.m. page every Tuesday. This chapter is the production engineering layer — runtime selection, batching, observability, capacity planning, cost economics, and the failure modes that show up only under real load.

Most of what's in the production-serving lecture of the Qwen Inference Optimization series (Edge AI / Lecture 05) still applies on Blackwell. This chapter focuses on what **changes** when you move from H100/H200 to B200 — and what doesn't change but is worth re-validating.

By the end you should be able to:

* Choose between TRT-LLM, vLLM (Blackwell backend), SGLang, and LMDeploy for a given workload.
* Set up observability that catches Blackwell-specific failure modes (kernel fallback to FP8, NVLink-5 degradation, Grace memory paths).
* Plan capacity for a Qwen2.5-72B chat product with realistic peak factors.
* Quantify the cost economics vs H100/H200 — and identify the workloads where Hopper still wins.

---

## 1. Production Runtime Choices, Mid-2026

| Runtime | Blackwell support | Best at | Caveat |
|---|---|---|---|
| **TensorRT-LLM 0.20+** | First-class, NVIDIA-tested | Highest raw throughput, MX-FP4 mature, full Triton-Inference-Server integration | NVIDIA-only; build flow is heavier |
| **vLLM (Blackwell backend)** | Merged Q1 2026 | OpenAI-API serving, broad model support, AWQ/GPTQ fallbacks | MX-FP4 path is younger; less mature on edge cases |
| **SGLang on B200** | Active port, Q2 2026 | Programmatic generation, structured outputs, prefix caching | Smaller team, slower release cadence |
| **LMDeploy / TurboMind** | Late-2026 Blackwell port | Qwen-specific optimizations, InternLM-team focus | Catching up to TRT-LLM/vLLM on Blackwell |
| **Custom (CUTLASS 4 + persistent kernels)** | Roll-your-own | Research, novel quant formats, model archs | Don't unless stock runtimes fall short |

For Qwen2.5-72B on a single B200 or HGX B200 box in mid-2026 production: **TRT-LLM 0.20+** is the default choice. vLLM is competitive on most workloads and easier to deploy in a Kubernetes-native shop. SGLang wins for prefix-cache-heavy workloads (RAG with long shared system prompts).

---

## 2. Deployment Pattern — Triton Inference Server + TRT-LLM

The reference NVIDIA stack:

```
┌──────────────────────────────────────────────────────────┐
│   Triton Inference Server (port 8000)                    │
│   ┌────────────────────────────────────────────────┐    │
│   │  TRT-LLM backend                               │    │
│   │   - Continuous batching                        │    │
│   │   - Paged KV cache                             │    │
│   │   - Chunked prefill                            │    │
│   │   - Speculative decoding (EAGLE-2)             │    │
│   │   - Streaming responses                        │    │
│   └──────────────────┬─────────────────────────────┘    │
│                      │                                   │
│                      ▼                                   │
│        Qwen2.5-72B-MX-FP4 engine                         │
│        (sm_100, persistent kernels)                      │
│                      │                                   │
│                      ▼                                   │
│             B200 (or HGX B200 × N)                       │
└──────────────────────────────────────────────────────────┘
                      │
                      │ REST/gRPC
                      ▼
              Application layer
              (OpenAI-compatible API)
```

Deployment manifest (Docker Compose excerpt):

```yaml
services:
  triton:
    image: nvcr.io/nvidia/tritonserver:25.06-trtllm-python-py3
    runtime: nvidia
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 8                    # HGX B200 full board
              capabilities: [gpu]
    environment:
      - NCCL_P2P_LEVEL=NVL
      - NCCL_DEBUG=WARN
      - CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
      - LD_LIBRARY_PATH=/opt/tritonserver/backends/tensorrtllm
    volumes:
      - ./qwen72b-mx-fp4-engine-tp8:/models/qwen
    command: >
      tritonserver
        --model-repository=/models
        --http-port=8000
        --grpc-port=8001
        --log-verbose=1
        --strict-readiness=false
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/v2/health/ready"]
      interval: 30s
      timeout: 10s
      retries: 3
```

Expected baseline numbers from this deployment (mid-2026):

| Metric | HGX B200, TP=8 |
|---|---|
| Single-stream decode | ~620 tok/s |
| Batch=32 aggregate decode | ~14,000 tok/s |
| Batch=128 aggregate decode | ~32,000 tok/s |
| TTFT @ 2k prompt, B=8 | 25 ms |
| TTFT @ 32k prompt (chunked) | 180 ms |
| HBM utilization at saturation | ~90% |
| GPU compute utilization | 70-85% |
| Power per GPU at saturation | 950-1000 W |
| Total box power | 8-9 kW |

---

## 3. Observability — What to Watch

The dashboards from the Edge AI / Qwen / Lecture 05 still apply. What's added for Blackwell:

### 3.1 Blackwell-specific metrics

* **MX format kernel hit rate** — fraction of GEMM calls hitting the MX-FP4 path vs the FP8 fallback path. TRT-LLM exposes this via `kernel_dispatch_fp4_count` vs `kernel_dispatch_fp8_count`. If FP8 fallbacks exceed 1%, your recipe is too aggressive.
* **Transformer Engine 2 promotion events** — TE2 logs when a block's quant gets promoted at runtime (overflow detection). Counter: `te2_block_promotion_count`. Healthy: < 100/hour under steady load.
* **NVLink-5 link health** — `nvidia-smi nvlink --errors -l 0` for each GPU should show zero CRC errors. Errors here silently degrade throughput by 20–50% for the affected GPU.
* **Grace coherent memory access ratio** (GB200/NVL72) — fraction of memory reads that crossed the NVLink-C2C from B200 to Grace. Above ~10% suggests KV spill is dominant and worth re-tuning.
* **TMA-2 multicast utilization** — Nsight metric `tma_multicast_loads_per_sec`. Should be high in attention kernels; absence means FA-3 isn't actually active.

### 3.2 The Prometheus scrape

A minimal Prometheus query set for a Triton + TRT-LLM Blackwell deployment:

```promql
# Latency
histogram_quantile(0.50, rate(triton_inference_compute_input_duration_us_bucket[1m]))
histogram_quantile(0.95, rate(triton_inference_request_duration_us_bucket[1m]))

# Throughput
rate(trtllm_decoded_tokens_total[1m])

# KV cache health
trtllm_kv_cache_used_blocks / trtllm_kv_cache_total_blocks

# GPU
DCGM_FI_DEV_GPU_UTIL
DCGM_FI_DEV_MEM_COPY_UTIL
DCGM_FI_DEV_POWER_USAGE
DCGM_FI_DEV_NVLINK_BANDWIDTH_L0   # NVLink utilization

# Blackwell-specific
trtllm_mx_fp4_kernel_dispatches_total
trtllm_te2_promotion_events_total
trtllm_tma2_multicast_loads_total
```

### 3.3 Failure modes you'll see in production

| Symptom | Cause | Fix |
|---|---|---|
| Tok/s drops 30% overnight without code change | Driver update reset kernel cache; first req cold-paths | Warm-up step in deploy hook |
| Specific GPU shows 60% util while others 80% | NVLink-5 CRC errors on that GPU's links | Re-seat or RMA; rebalance shard |
| MX-FP4 hit rate < 80% | Engine built without `--gemm_plugin mx_fp4` | Rebuild with correct flags |
| TE2 promotion events spike on layer 47 | Activation distribution drift on a tensor | Re-calibrate with broader dataset |
| Grace memory access ratio climbs over time | KV cache fragmentation in Grace tier | Restart serving instance; investigate paging policy |
| TTFT p99 spikes correlate with long prompts | Chunked prefill not enabled or too small | Increase chunk size or enable |
| Cold-start TTFT >10s after deploy | TensorRT graph compile from scratch | Cache compiled engine; preload at boot |
| OOM at certain concurrency despite headroom | Activation working set spike, paged-KV race | Lower max_num_tokens; raise free fraction |

---

## 4. Capacity Planning — Worked Example

Same scenario as the Edge AI / Qwen / Lecture 05 example, re-derived for Blackwell:

**Scenario:**
- 5,000 daily active users (5× the original example).
- 50 turns/user/day.
- 300 input + 400 output tokens average.
- 40% to Qwen2.5-72B-class cloud (60% on edge for free).

**Daily cloud load:**
```
Queries:     5000 × 50 × 0.40 = 100,000 / day
Output tok:  100,000 × 400    = 40 M / day
Avg tok/s:   40 M / 86,400    ≈ 460 tok/s
Peak tok/s (5× peak factor):  ~2,300 tok/s
```

**Sizing options:**

| Platform | Capacity at p95 SLA | Boxes needed | Capex | Yearly OPEX (rented) |
|---|---|---|---|---|
| HGX H100 (FP8, TP=8) | ~5,000 tok/s sustained | 1 | ~$250k | ~$170k |
| HGX H200 (FP8, TP=8) | ~7,000 tok/s sustained | 1 | ~$280k | ~$190k |
| **HGX B200 (MX-FP4, TP=8)** | **~15,000 tok/s sustained** | **1** | **~$280k** | **~$220k** |
| Single B200 (intra-TP=2) | ~3,000 tok/s sustained | 1 | ~$45k | ~$45k |

For a 2,300 tok/s peak you actually have several viable options:

* **Single B200** with intra-package TP=2 fits the peak with ~25% headroom. Cheapest by a wide margin if you don't need redundancy.
* **HGX H100/H200** also works but with less headroom and a near-future capacity ceiling.
* **HGX B200** is overkill for this load but gives you 6× growth headroom — pick this if you expect to 5× users in 12 months.

For redundancy/failover, always pair: e.g., **2 × single B200** in active/passive, or **1 HGX B200 + 1 spare B200** for graceful degradation. Don't run production single-instance.

---

## 5. Cost Economics — Where Blackwell Wins, Where Hopper Persists

Per-million-tokens-served cost is the production-relevant metric. Mid-2026 numbers (rented cloud capacity, fully-loaded):

| Platform | Cost per 1M output tokens, batch-32 steady state |
|---|---|
| Single H100 SXM | $0.85 |
| HGX H100 8-GPU, TP=8 | $0.34 |
| HGX H200 8-GPU, TP=8 | $0.28 |
| **Single B200** | **$0.30** |
| **HGX B200 8-GPU, TP=8** | **$0.18** |
| OpenAI gpt-4.1-mini API list price | $0.60 |

The HGX B200 8-GPU result is ~2× better than HGX H200, ~3× better than H100, **and competitive with API list prices for hosted models**. This is the cost-economics inflection point that drove the rapid market shift to Blackwell through 2026.

### 5.1 When Hopper still wins on cost

* **Workloads that don't use FP4** — if your eval shows MX-FP4 regression you can't tolerate, you're running FP8 on B200. The cost advantage shrinks to ~1.4× over H200 — still meaningful, but less commanding.
* **Latency-tier-0 with low concurrency** — at single-user serving, the per-GPU cost dominates and Hopper's lower chip price wins. Single H100 chat costs about half what single B200 chat costs.
* **Existing fleet write-off** — if you already paid for a Hopper fleet, the marginal cost of running it through depreciation is small. Refresh on capacity-add, not on full-replace.
* **Compliance/contracts** — some procurement processes are locked to specific SKUs. Migration timelines are quarters, not weeks.

---

## 6. The Production Checklist

Before shipping Qwen on Blackwell to real traffic:

- [ ] CUDA 13.x driver installed, verified `cuda-compute-capability=10.0`.
- [ ] TRT-LLM 0.20+ installed; `pip list | grep tensorrt_llm` shows expected version.
- [ ] Engine built with `--gemm_plugin mx_fp4 --kv_cache_type mx_fp8` (verify via cuobjdump).
- [ ] FlashAttention-3 kernels active (verify Nsight or kernel disasm).
- [ ] NCCL `NCCL_P2P_LEVEL=NVL` set; `nvidia-smi topo -m` shows NV18 between all pairs.
- [ ] `nvidia-smi nvlink --errors` clean on all GPUs.
- [ ] Calibration set matches deployment domain (chat / code / multilingual).
- [ ] Eval baseline established: MMLU within 0.5 pt of BF16, IFEval within 1.5 pt.
- [ ] Long-context eval (needle-in-haystack at >50k) passes.
- [ ] Health endpoints respond: `/v2/health/ready`.
- [ ] Prometheus scraping `triton_*` and `DCGM_FI_DEV_*` metrics.
- [ ] Grafana dashboards include: TTFT p50/p95/p99, ITL p50/p95, KV occupancy, MX-FP4 hit rate, NVLink errors, GPU power.
- [ ] Alerts wired: TTFT p95 > SLA, GPU power < 800W (suggests low utilization), MX-FP4 hit rate < 80%, NVLink errors > 0.
- [ ] Capacity tested at 1.5× expected peak.
- [ ] Failover plan documented: what happens if one GPU fails on the HGX board.
- [ ] Warm-up step in deploy hook: send N requests after start to populate kernel/engine caches.
- [ ] Cost monitoring: tokens-served / $ tracked daily.

---

## 7. The Next 12 Months — Speculative

What changes between mid-2026 and mid-2027 in this space:

* **B300 / Blackwell Ultra** likely lands in volume. Same MX format support, marginally faster memory, more HBM capacity (288 GB+). Drop-in for B200 in most deployments.
* **FP6 maturity** — currently FP4-mixed is the sweet spot; FP6-pure may become the production default as kernels mature. Cuts memory ~25% vs FP6-mixed at near-zero quality loss.
* **Better speculative decoding** — EAGLE-3 ports to Blackwell; throughput multipliers in the 2-3× range likely.
* **Cross-arch portability of quantized models** — running an MX-FP4 quantized model on Hopper via emulation is slow today; this gap closes.
* **MoE serving optimization** — for Qwen3-MoE and similar, the Grace LPDDR expert pool pattern matures, dropping the cost of expert serving by 2-3×.
* **NVL576 (multi-rack)** — frontier customers stack NVL72 racks via additional NVLink generations. Trillion-parameter inference at scale becomes routine.

---

## Key Takeaways

| Takeaway | Why it matters |
|---|---|
| TRT-LLM 0.20+ is the production default for Qwen on B200 in mid-2026 | Mature MX-FP4 path, Triton-server integration, NVIDIA-tested |
| HGX B200 8-GPU replaces 4–6 HGX H100 boxes for the same throughput | The fleet refresh inflection point |
| MX-FP4 hit rate is a top-tier production metric | Below 80% means you're not getting the Blackwell advantage |
| NVLink-5 CRC errors silently degrade throughput | Add to your alerts; check `nvidia-smi nvlink --errors` |
| Blackwell wins per-token cost ~2× over H200, ~3× over H100 | Drives the market shift through 2026 |
| Single B200 beats Hopper on cost for many workloads with no multi-GPU overhead | Don't default to HGX-class boxes if a single chip suffices |
| Hopper still wins on capex-amortized fleet refresh and FP4-incompatible workloads | Migration is a quarter-scale project, not a week-scale one |
| Capacity planning hasn't changed — measure, then size | Same TTFT/ITL/KV/util metrics as Hopper |

---

## Resources

* **[TensorRT-LLM 0.20 Release Notes](https://github.com/NVIDIA/TensorRT-LLM/releases):** Blackwell support and MX-FP4.
* **[Triton Inference Server User Guide](https://docs.nvidia.com/deeplearning/triton-inference-server/):** Front-end and orchestration.
* **[NVIDIA DCGM exporter](https://github.com/NVIDIA/dcgm-exporter):** Prometheus exporter for GPU metrics.
* **[vLLM Blackwell backend tracking issue](https://github.com/vllm-project/vllm):** Status of vLLM Blackwell features.
* **[SGLang documentation](https://sgl-project.github.io/):** Structured generation on B200.
* **[LMDeploy / TurboMind](https://github.com/InternLM/lmdeploy):** Qwen-optimized inference engine.
* **[Phase 5 — Edge AI / Qwen Inference Optimization / Lecture 05](../../../../Track%20C%20-%20Edge%20AI/Qwen%20Inference%20Optimization/Lecture-05.md):** The original production-serving playbook.
* **[Chapter 1 — Blackwell Architecture](01-Blackwell-Architecture.md):** Where this all starts.
