# Part 2 · Lecture 04 — Single-Node Multi-GPU Serving: Tensor Parallelism on 8× H100/H200

## Overview

A 70B-class dense model in FP16 is **~140 GB**. No single GPU in the Hopper generation holds it without quantization — H100 has 80 GB, H200 has 141 GB. To serve it at higher precision than INT4, or to serve at larger batch / longer context, the model is **split across multiple GPUs** in a single node, joined by NVLink.

The dominant 2026 split is **tensor parallelism (TP)** — the canonical approach pioneered by Megatron-LM. This lecture covers TP end-to-end for the anchor pair:

1. The TP partitioning of attention and FFN layers — which matrices split how.
2. The NCCL all-reduce that dominates TP step time.
3. NVLink + NVSwitch fabric reality on HGX H100/H200 8-GPU boxes.
4. TP=2 vs TP=4 vs TP=8 — what each tradeoff actually costs in throughput and latency.
5. Runtime-specific TP configuration — vLLM, SGLang, TRT-LLM.
6. Diagnosis — when the all-reduce is the bottleneck and what to do.
7. Sequence parallelism and tensor-parallel-FFN overlays.

By the end you should be able to set up TP on 8× H100/H200 for either Llama 3.3 70B or Qwen 2.5 72B, measure the all-reduce overhead, and tell whether a given workload should use TP=4 or TP=8.

---

## 1. Tensor parallelism — the partitioning

TP splits *individual matrix multiplications* across GPUs. For each Linear layer, the weight matrix is **partitioned** and each GPU computes a slice of the output. The slices are joined by **all-reduce**.

### 1.1 Attention partition

Q, K, V projections are partitioned by head:

```text
On 4-GPU TP, each GPU holds 16 of 64 Q heads + 2 of 8 KV heads:
  GPU 0: Q_heads_0..15, K_heads_0..1, V_heads_0..1
  GPU 1: Q_heads_16..31, K_heads_2..3, V_heads_2..3
  GPU 2: Q_heads_32..47, K_heads_4..5, V_heads_4..5
  GPU 3: Q_heads_48..63, K_heads_6..7, V_heads_6..7

Attention computed locally per GPU on its head shard:
  attn_local = softmax(Q_local · K_local^T) · V_local

Output projection partitioned by *rows*:
  W_o split: rows 0..2047 on GPU 0, rows 2048..4095 on GPU 1, etc.
  
  Per-GPU partial: attn_local · W_o_local → partial_d
  All-reduce across GPUs: full_d = sum(partial_d) over 4 GPUs
```

The **all-reduce** is the cross-GPU step. Without it, each GPU has only its head shard's contribution.

### 1.2 FFN partition

The two FFN matmuls split differently:

```text
W_gate, W_up: split by columns (output dim).
  Per GPU computes (B, d) × (d, d_ff/N) → (B, d_ff/N)
  No communication yet — local result.

SwiGLU: silu(gate) * up — element-wise, local.

W_down: split by rows (input dim).
  Per GPU computes (B, d_ff/N) × (d_ff/N, d) → (B, d) partial
  All-reduce across GPUs to get full (B, d)
```

Again, the FFN ends with an all-reduce.

### 1.3 Total all-reduces per layer

For each transformer layer, **two all-reduces** — one after attention output projection, one after FFN down projection. For an 80-layer model that's **160 all-reduces per token** at decode.

This is the **single largest performance consideration** in TP serving.

---

## 2. The NCCL all-reduce — what it actually does

NCCL (NVIDIA Collective Communications Library) implements all-reduce as either **ring** or **tree** algorithm depending on size and topology.

### 2.1 Ring all-reduce

For 4 GPUs with NVLink:

```text
Round 1: GPU 0 sends slice to GPU 1, GPU 1 to GPU 2, etc.
Round 2: each GPU receives + sums slice
... 2(N-1) rounds total to fully reduce + share
```

Bandwidth: each GPU sends `2 * (N-1) * (size/N)` bytes. For large messages, near-optimal bandwidth use.

### 2.2 Tree all-reduce

For 8+ GPUs, hierarchical tree reduces are sometimes faster:

```text
log2(N) levels of pairwise reduce → final value at root → log2(N) levels of broadcast back
```

Lower latency for small messages, higher latency for large messages (less bandwidth-efficient).

### 2.3 What NCCL picks

NCCL auto-tunes per (message size, topology). On HGX H100/H200 with NVLink Switch (fully-connected 8-GPU domain), ring is typical for the all-reduces in TP because the messages are large (~16-32 MB per layer at FP16).

### 2.4 The bandwidth cost

For an FFN down-projection all-reduce in 70B FP16:

```text
message_size_per_gpu = batch × hidden_size × bytes
                     = 32 × 8192 × 2
                     = 0.5 MB at batch=32

Across 80 layers × 2 all-reduces per layer:
total bytes = 80 × 2 × 0.5 MB = 80 MB per token decode

On NVLink (900 GB/s aggregate) the all-reduce time is small for this message size,
but the *latency* (round-trips) adds up.
```

For prefill at batch=1, prompt=2048:

```text
message_size = 2048 × 8192 × 2 = 32 MB per layer per all-reduce
total bytes per prompt = 80 × 2 × 32 MB = 5.1 GB cross-GPU traffic per prefill
```

This is significant. Prefill TP=8 spends **~25% of step time in NCCL all-reduces**.

---

## 3. NVLink + NVSwitch on HGX boxes

The physical fabric for 8× H100 / H200 in standard HGX configurations:

* **NVLink 4** — 50 GB/s per link, 18 links per GPU → 900 GB/s aggregate per GPU.
* **NVSwitch v3** — 4 switches in the HGX baseboard interconnect the 8 GPUs in an all-to-all topology.
* **Each pair of GPUs has 900 GB/s effective bandwidth.**
* **Latency** — sub-microsecond for small messages.

This fabric is the reason **TP=8 is practical at all**. Without NVLink, TP across 8 GPUs through PCIe (128 GB/s aggregate to host, 64 GB/s peer-to-peer if not blocked) would be **10–14× slower**.

### 3.1 NVLink Switch chiplets and the H200 NVL configuration

Newer HGX configurations and the upcoming **NVL** chassis use NVLink switches as separate chips that interconnect across servers. For inference-on-single-node-server workloads (which is the entire scope of this lecture), the baseboard NVSwitch is what matters.

---

## 4. TP=2 vs TP=4 vs TP=8

For Llama 3.3 70B / Qwen 2.5 72B:

| TP | Per-GPU weight share (FP16) | Per-GPU KV share | Communication overhead | When to use |
|----|-----------------------------|-------------------|------------------------|-------------|
| 1 | doesn't fit at FP16 | n/a | none | INT4 only on H200 |
| 2 | ~70 GB | 50% | ~10% of step time | H100 NVL or H200 single-node |
| 4 | ~35 GB | 25% | ~15% of step time | best chat-shape throughput |
| 8 | ~18 GB | 12.5% | ~25% of step time | max batch / long context |

### 4.1 The TP scaling efficiency

Ideal: 4× more GPUs → 4× more throughput.
Real: 4× more GPUs → 2.5–3.5× more throughput at this hardware/model.

The gap is **communication**. The plot looks like:

```text
   per-replica throughput
        ▲
        │       •  TP=2   (95% efficiency vs single-GPU)
        │   •        TP=4   (85% efficiency)
        │            TP=8 •  (65–75% efficiency)
        │
        └────────────────────► TP degree
```

Scaling efficiency depends on workload:

* **Chat (decode-dominant, batch=32):** TP=4 is the sweet spot. TP=8 sacrifices ~25% to gain headroom for KV cache / longer context.
* **Batch / offline (prefill-dominant):** TP=8 can win because prefill is compute-bound; FFN matmuls amortize the all-reduce cost.
* **Long context (128K):** TP=8 because the KV cache pressure forces it; the throughput hit is acceptable because long-context throughput is bottlenecked elsewhere.

### 4.2 The per-GPU throughput trap

A common metric mistake: comparing TP=4 vs TP=8 in **per-replica throughput**.

* TP=4 at 1200 tok/s → 300 tok/s/GPU.
* TP=8 at 1600 tok/s → 200 tok/s/GPU.

TP=8 has **higher absolute throughput** (better TPOT, more users served per replica) but **worse $/MTok** if each H200 has a fixed cost.

For chat products, **TP=4 is usually the cost-efficient pick**. TP=8 is for cases where TP=4 cannot fit the workload (e.g., 128K context at large batch).

### 4.3 When the TP size is not yours to choose — FP8 block alignment

There is a hard constraint that overrides the cost/throughput reasoning above: **block-scaled FP8 weights require every tensor-parallel shard's dimension to be a whole number of quantization blocks.** If `dim / TP` is not a multiple of the FP8 block size, the engine fails to load.

For Qwen 2.5 72B the FFN intermediate `29568 = 128 × 231` is *not* divisible by 128 after splitting across 2, 4, or 8 GPUs — so block-FP8 + TP=8 will not load until you drop to a TP that re-aligns (or use a framework build that pads). The mechanism and the fix-order are in [Lecture 03 §5.4](Lecture-03.md). The takeaway for *this* lecture: when you serve FP8, **validate that the model loads at your chosen TP before you reason about throughput** — quantization can force a smaller TP than the workload alone would pick.

---

## 5. Runtime-specific TP configuration

### 5.1 vLLM

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.3-70B-Instruct",
    tensor_parallel_size=4,        # the TP degree
    dtype="float16",
    gpu_memory_utilization=0.92,
)
```

vLLM auto-discovers GPUs and assigns ranks. Works on standard HGX systems out of the box.

### 5.2 SGLang

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-72B-Instruct \
    --tp-size 4 \
    --dtype float16 \
    --port 30000
```

SGLang's TP integrates with its RadixAttention and EP support. Same fabric assumptions as vLLM.

### 5.3 TensorRT-LLM

```bash
trtllm-build \
    --checkpoint_dir ./fp8_checkpoint \
    --output_dir ./engine \
    --tp_size 4 \
    ...
```

TRT-LLM compiles the model graph for the chosen TP size. Changing TP requires a rebuild — this is a deployment-time decision, not a runtime config.

### 5.4 Cross-runtime parity

For Llama 3.3 70B at FP16, TP=4, batch=32 on 4× H100:

| Runtime | TPOT (mean) | Throughput tok/s/GPU |
|---------|-------------|----------------------|
| vLLM 0.22 | ~30 ms | ~310 |
| SGLang 0.5 | ~29 ms | ~315 |
| TRT-LLM 1.3 BF16 | ~26 ms | ~360 |
| TRT-LLM 1.3 FP8 | ~16 ms | ~580 |

Numbers approximate; replicate in your lab. TRT-LLM's lead is mostly **kernel quality on Hopper** (FA4, WGMMA, TMA) and FP8 maturity. vLLM is closing the gap quarter by quarter.

---

## 6. Diagnosis — is the all-reduce the bottleneck?

A profile-driven diagnosis:

1. **Nsight Systems timeline** — look for NCCL kernel time as a fraction of step time.
2. **Per-layer breakdown** — `nsys` shows attention matmul + all-reduce + FFN matmul + all-reduce per layer.
3. **NCCL communication time** — is it overlapping with compute? On Hopper, vLLM 0.22+ and TRT-LLM use `cudaStream`-based overlap; older runtimes serialize.

Quick diagnostic:

| If you see | Suspect |
|------------|---------|
| NCCL > 30% of step | All-reduce dominant — try fewer GPUs or smaller messages |
| NCCL serial with compute (no overlap) | Outdated runtime — upgrade |
| Per-GPU FLOPs achievement low | Kernel quality / batch size — try larger batch |
| Step time scales poorly with TP | All-reduce — TP=4 → TP=8 is rarely 2× |

### 6.1 The reduce-scatter + all-gather pattern

For very large all-reduces, NCCL implements it as reduce-scatter (gather partial sums to one GPU each) + all-gather (broadcast back). This is what produces the 2× efficiency gap vs single-GPU on TP=8.

---

## 7. Sequence parallelism — the FFN overlay

A subtle but important optimization: **sequence parallelism** (SP) reduces the activation memory cost of TP by partitioning the sequence dimension across GPUs during non-matmul operations (RMSNorm, residual add, etc.).

Without SP, TP-partitioned models **duplicate full activations** across GPUs during RMSNorm and residual. SP partitions those operations too — each GPU holds only `seq/N` worth of activations.

* **Wins for long-context workloads** — at 128K context, SP saves substantial HBM.
* **vLLM 0.22+** integrates SP transparently when `--enable-sequence-parallelism` is set.
* **TRT-LLM** has SP behind a flag.

For TP=8 + SP, communication doubles (reduce-scatter + all-gather instead of all-reduce), but each communication is smaller. Net win for long-context workloads where the activation memory was the constraint.

---

## Lab — measure TP scaling on Llama 3.3 70B

Goal: produce TP=2 / 4 / 8 throughput numbers on the same hardware, with the all-reduce cost annotated.

1. **Hardware** — 8× H100 SXM (HGX) or 8× H200 if you have access.
2. **Model** — Llama 3.3 70B Instruct, FP16 (or FP8 if you have TRT-LLM).
3. **Runtime** — vLLM 0.22+ V1.
4. **Bench at three TP degrees** — TP=2, 4, 8. Same batch size (32), same prompt (1024), same output (256), same iterations (100, with 20 warmup).
5. **Profile one TP=4 and one TP=8 run** with Nsight Systems. Identify NCCL fraction of step time.
6. **Compute** per-GPU throughput, scaling efficiency, and per-replica throughput.
7. **Plot** the scaling curve. Annotate the bottleneck at each TP.

Pass criterion: you can defend the choice of TP for a chat product at 32K context with measured numbers — and explain why a different TP would be picked for a long-context batch product.

---

## Self-check

1. For Llama 3.3 70B FP16 on 4× H100 SXM, predict the all-reduce time per token decode at batch=32. Use 900 GB/s NVLink bandwidth and 160 all-reduces per token.
2. A teammate proposes TP=8 for a chat product to "double throughput." The TP=4 → TP=8 measurement shows 1.5× throughput at 65% per-GPU efficiency. Defend or reject in two sentences using $/MTok reasoning.
3. Your Nsight trace shows NCCL kernels running serially with compute kernels (no overlap). What runtime upgrade would you try first?
4. Sequence parallelism is "free" memory savings — why is it always-on then? What is the cost?
5. For Qwen 2.5 72B at TP=4 vs Llama 3.3 70B at TP=4 on the same hardware, which has worse scaling efficiency and why?

---

## References

* Megatron-LM tensor parallelism paper — [arXiv:1909.08053](https://arxiv.org/abs/1909.08053)
* "Reducing Activation Recomputation in Large Transformer Models" — [arXiv:2205.05198](https://arxiv.org/abs/2205.05198) — sequence parallelism
* NCCL documentation — [docs.nvidia.com/deeplearning/nccl/](https://docs.nvidia.com/deeplearning/nccl/)
* NVIDIA HGX H100 platform brief — [nvidia.com/en-us/data-center/hgx/](https://www.nvidia.com/en-us/data-center/hgx/)
* vLLM TP documentation — [docs.vllm.ai/en/latest/serving/distributed_serving.html](https://docs.vllm.ai/en/latest/serving/distributed_serving.html)
* TensorRT-LLM multi-GPU — [nvidia.github.io/TensorRT-LLM/](https://nvidia.github.io/TensorRT-LLM/)
* DistServe (referenced for disaggregation contrast) — [arXiv:2401.09670](https://arxiv.org/abs/2401.09670)

Cross-references:

* [Phase 5 → ML Systems Engineering Guide → Stage 5 Distributed Training Systems](../../Guide.md#stage-5-distributed-training-systems) — the training-side foundation
* [Phase 5 → GPU Infrastructure → 8x-H200-Training-Inference → 02 Training Setup](../../../1.%20GPU%20Infrastructure/Nvidia%20GPU/HPC%20Setup/8x-H200-Training-Inference/02-Training-Setup.md)

---

## Current as of 2026-06

NCCL 2.30+, vLLM 0.22+ V1, SGLang 0.5+, TRT-LLM 1.3+, NVLink 4, NVSwitch v3. Refresh when NCCL 3.0 / vLLM V1 stabilizes further / a new NVLink generation lands.

---

## Next

* Next: [Lecture 05 — Modern serving stack: continuous batching, paged KV, prefix cache, speculation](Lecture-05.md)
* Previous: [Lecture 03 — Quantizing Llama 3.3 70B and Qwen 2.5 72B](Lecture-03.md)
* Up: [Part 2 — Dense at Hopper](README.md)
