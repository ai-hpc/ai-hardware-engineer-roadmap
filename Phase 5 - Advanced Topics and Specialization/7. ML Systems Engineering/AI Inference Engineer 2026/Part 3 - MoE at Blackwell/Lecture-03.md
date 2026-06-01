# Part 3 · Lecture 03 — Expert Parallelism (EP) and the Gating Hot Path

## Overview

Tensor parallelism (Part 2 Lecture 04) splits a single matrix multiply across multiple GPUs and joins the result with an all-reduce. **Expert parallelism splits the experts of an MoE layer across multiple GPUs and joins the result with an all-to-all.** This one structural difference is the most consequential change in the inference graph going from dense to MoE.

This lecture covers:

1. **What EP partitions** and how token routing works across GPUs.
2. **The all-to-all communication pattern** — and why it's harder than TP's all-reduce.
3. **TP × EP × PP combinations** for MoE — when each parallelism axis pays its rent.
4. **DeepEP** and DeepSeek-specific optimizations.
5. **Expert load balancing** — the runtime decision that decides whether EP scales.
6. **Gating computation cost** — small per-token, large per-cluster.
7. **Token-level routing on NVL72** — what scales and what doesn't.
8. The Llama-style dense TP fallback (when EP loses to TP for small MoEs).

By the end you should be able to pick the right TP × EP combination for DeepSeek V3.1 or Qwen3-MoE 235B-A22B on a given Blackwell cluster, predict the all-to-all overhead, and diagnose load-imbalance bottlenecks.

---

## 1. What EP partitions

In expert parallelism, **each GPU holds a subset of the experts**. For DeepSeek V3.1 with 256 routed experts at EP=8:

```text
GPU 0: experts 0..31
GPU 1: experts 32..63
GPU 2: experts 64..95
GPU 3: experts 96..127
GPU 4: experts 128..159
GPU 5: experts 160..191
GPU 6: experts 192..223
GPU 7: experts 224..255

Shared experts: replicated on every GPU
Gating network: replicated on every GPU
```

At a token's MoE layer, the gating network produces top-8 expert IDs. These IDs may be on any GPU. The token's hidden state must be **dispatched to the right GPUs**, the FFN run there, and the results **gathered back**.

This is the all-to-all communication. Every GPU sends tokens to every other GPU according to which experts those tokens need.

### 1.1 The dispatch table

At each MoE layer, per step:

```text
Input: hidden states for all tokens in the batch — shape (B, hidden)
Gating: top-8 expert IDs per token — shape (B, 8)

For each (token, expert_id) pair:
  expert_owner_gpu = expert_id // (n_experts / EP_size)

Build per-source-rank send buffers:
  send[rank_r] = [token i, hidden_state_i for all tokens routed to expert on rank r]

NCCL all-to-all dispatch
Each rank receives: tokens needing local experts
Compute FFN per expert locally on received tokens
NCCL all-to-all combine: send back to original ranks

Original ranks combine outputs weighted by gating scores
```

### 1.2 Why this is harder than TP all-reduce

TP all-reduce: every rank contributes the same-sized buffer, NCCL reduces in a ring or tree. Predictable cost.

EP all-to-all: every rank sends *different-sized* buffers to *different* destinations (because tokens are routed unevenly to experts). NCCL `alltoallv` handles variable sizes but communication efficiency depends on:

* **Token distribution** — if all tokens go to one expert, only one rank sees traffic.
* **NVLink topology** — ideally all-to-all uses the fabric symmetrically.
* **Message size** — small messages hit the latency floor; large messages saturate bandwidth.

At low batch sizes, EP all-to-all is **latency-dominated**. At high batch sizes, it becomes bandwidth-dominated. Both regimes have different optimization knobs.

---

## 2. The all-to-all cost

For DeepSeek V3.1 at EP=8 on 8× B200, batch=64, decode (1 token per request):

```text
Per layer per step:
  - 64 tokens × 8 experts each = 512 token-expert pairs
  - Average tokens per expert: 512 / (256/8) = 16
  - Per token data: hidden_size × bytes = 7168 × 1 (FP8) = 7 KB

Send per source rank: 64 tokens × ~50% local = 32 tokens elsewhere
                    ≈ 32 × 7 KB / 8 destinations = ~28 KB per pair

NCCL alltoallv: ~28 KB to 7 destinations, asymmetric
  Latency-bound regime — bandwidth not the constraint
  Approx 100-200 μs per all-to-all

Per layer: 2 all-to-alls (dispatch + combine) ≈ 200-400 μs
Per token (61 MoE layers): ~12-24 ms in all-to-all alone
```

Compare to the bandwidth-bound weight read on B200:

```text
Active weights: 37B × 0.5 bytes (FP4) = 18.5 GB
Per GPU at EP=8: 18.5 / 8 ≈ 2.3 GB (only the experts hit per token)
HBM read time: 2.3 / 8 ≈ 0.3 ms per layer
Per token (61 layers): ~18 ms in weight reads
```

So in a TP=8 EP=8 deployment, weight read and all-to-all are **roughly equal contributors** to decode time. This is why MoE inference is fundamentally a **communication-bound** problem — even more than dense TP — and why NVLink 5's bandwidth doubling pays off.

### 2.1 At larger batch sizes

At batch=512:

```text
Per layer: 512 tokens × 8 experts = 4096 token-expert pairs
Average tokens per expert: 4096 / 32 = 128
Per all-to-all message: ~30 tokens × 7 KB = ~200 KB per destination

Now bandwidth-dominated: 200 KB × 7 destinations ÷ 1.8 TB/s ≈ 1 μs (much less than latency floor)
But NCCL setup + small-message handling ≈ 50-100 μs

Per layer: ~100-200 μs all-to-all
Per token (61 layers): ~6-12 ms
```

Larger batch amortizes the per-step all-to-all cost. **MoE strongly prefers higher batch sizes than dense.** Continuous batching is essential.

---

## 3. TP × EP × PP combinations

For an MoE model, three parallelism axes:

* **Tensor parallelism (TP)** — splits each layer's matrices across GPUs (same as dense).
* **Expert parallelism (EP)** — splits experts across GPUs.
* **Pipeline parallelism (PP)** — splits layers across GPUs (rarely used in inference, common in training).

The total degree: `TP × EP × PP` GPUs per replica.

### 3.1 Pure EP (no TP)

Each GPU holds a subset of experts plus a full copy of the attention block.

* For DeepSeek V3.1 (attention is dense at ~110M params per layer): each GPU has the full attention weights — cheap.
* All-to-all only for MoE layers.
* Attention runs locally — no TP all-reduce needed.

**Pure EP is the standard 2026 MoE recipe.** EP=8 for 8× B200, EP=16 for 16× B200, EP=64 for the NVL72.

### 3.2 TP + EP (combined)

For very large active params or memory-tight deployments:

* TP partitions the attention block and the per-expert FFN matrices.
* EP partitions the experts.
* Combined: each GPU has 1/TP of each expert's weights and 1/EP of the experts.

This is used for:

* DeepSeek V3.1 at high concurrency where the attention itself is a bottleneck.
* TP=2, EP=8 (16 GPUs total) for a balanced split.

The cost: more all-reduces (TP) + more all-to-alls (EP). Diminishing returns; rarely worth it unless attention compute is the bottleneck.

### 3.3 Recommendation table

| Model | Cluster | Recipe |
|-------|---------|--------|
| Qwen3-MoE 235B-A22B | 2× B200 | EP=2, FP4 weights, FP8 KV |
| Qwen3-MoE 235B-A22B | 8× B200 | EP=8, FP4 weights, FP8 KV |
| DeepSeek V3.1 | 8× B200 | EP=8, FP4 weights, BF16 MLA-KV |
| DeepSeek V3.1 | 16× B200 | EP=16, FP4 weights, FP8 KV |
| DeepSeek V3.1 | NVL72 (single replica) | TP=2 × EP=32 = 64 GPUs, FP4 |
| DeepSeek V3.1 | NVL72 (multi-replica) | 4 replicas × 16 GPUs each |

---

## 4. DeepEP and DeepSeek-specific optimizations

DeepSeek released **DeepEP** ([github.com/deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP)) — an EP communication library specifically tuned for the DeepSeek V3 family.

Key optimizations:

* **Asymmetric dispatch and combine** — uses NVLink RDMA and CUDA IPC primitives directly instead of going through NCCL collectives.
* **Two-phase routing** — first all-to-all for intra-node, second for inter-node (when NVL72 is partitioned).
* **Token-tier rebalancing** — re-shuffles tokens between layers to balance expert load.

Benchmark gains over default NCCL `alltoallv` on NVL72:

* 1.5–2× faster all-to-all latency on small batches.
* 1.2–1.5× faster on large batches (the bandwidth regime is already close to NVLink peak).

SGLang and TRT-LLM integrate DeepEP for DeepSeek-family deployments. vLLM does so for the DeepSeek model path. For Qwen3-MoE 235B-A22B (which is not DeepSeek), the same techniques apply but the library integration varies.

---

## 5. Expert load balancing

The single most subtle MoE inference issue.

### 5.1 The problem

Not all experts are activated equally. Some experts get many tokens, some get few. The slowest expert sets the step time (a "straggler" pattern).

In training, MoE loss includes a **load-balancing auxiliary term** that penalizes uneven routing. At inference time, this is fixed — but training-time imbalance can persist as natural distribution bias.

Real-world: a deployment might see expert utilization vary 3-10×. The most-loaded expert does 3-10× more work per step than the least-loaded.

### 5.2 Approaches

**Token-level rebalancing** (DeepEP, SGLang):
* Detect imbalance per layer.
* Re-shuffle a small fraction of tokens to less-loaded experts even if the gating score is suboptimal.
* Trade slight quality loss for substantial throughput.

**Expert replication** (when budget allows):
* Replicate hot experts on multiple GPUs.
* Route tokens to either copy.
* Cost: more HBM (typically not affordable at FP4 already).

**Drop-token policies** (older, less common):
* If an expert is over capacity, drop the lowest-score tokens.
* Quality degradation; rarely used in inference (acceptable in training).

### 5.3 Measurement

Production runtimes (SGLang, vLLM) expose expert utilization metrics:

```text
expert 0..7   on GPU 0: tokens_per_step = [180, 95, 110, 75, 140, 88, 200, 95]
expert 0..7   on GPU 1: ...
```

Use Nsight Systems' MoE-aware view (newer Nsight versions) or runtime metrics endpoints. **Watch for ratio max/min > 3** — this is the threshold where rebalancing becomes worthwhile.

---

## 6. Gating computation cost

The gating network is `Linear(hidden, num_experts)`. For DeepSeek V3.1:

```text
Per token: 7168 × 256 = 1.83M FLOPs
Per layer × per batch: 1.83M × batch × seq
```

At batch=64, seq=128 (typical chat-shape):

```text
Per gating call: 1.83M × 64 × 128 ≈ 15 GFLOPs
On B200 FP16 (2,250 TFLOPs peak): ~7 μs

Per layer: gating + topk + scatter overhead ≈ 20-50 μs
Per token (61 layers): ~1.2-3 ms in gating-related ops
```

Small. Usually not the bottleneck unless the runtime's gating implementation is unoptimized.

The subtlety: gating is *frequent* and *latency-sensitive*. Even if individual cost is small, poor implementation can add 10-20% to total step time. vLLM 0.22+, SGLang 0.5+, and TRT-LLM all have optimized gating paths.

---

## 7. Token-level routing on NVL72

At full NVL72 scale (72 GPUs), some additional considerations:

* **EP=64** is achievable with 8 GPUs holding shared infrastructure (gating, attention, embed).
* **Cross-domain communication** — within NVL72 fabric, NVLink is uniform. Outside (multi-rack), latencies increase.
* **Topology-aware routing** — DeepEP and SGLang use NVLink topology hints to optimize.

For NVL72 single-replica DeepSeek V3.1:

* Per-GPU expert holding: 256 / 64 = 4 experts.
* Per-token routing: 8 experts → at most 8 GPUs touched per token.
* All-to-all communication scales like O(EP × batch × hidden), which is still tractable on 130 TB/s aggregate NVLink BW.

### 7.1 When EP doesn't pay

For **small batches** (concurrency < 16), EP overhead can exceed the bandwidth savings. For these workloads, **fewer-GPU TP** can be faster than many-GPU EP:

* Qwen3-MoE 235B-A22B at concurrency 8 on 2× B200 (TP=2, EP=2): TPOT ~12 ms.
* Same on 8× B200 (EP=8): TPOT ~10 ms — only 17% faster despite 4× the GPUs.

For low-batch chat products, smaller-cluster deployments are often more cost-efficient.

---

## Lab — bench EP scaling on Qwen3-MoE

Goal: measure EP scaling and identify the all-to-all overhead crossover.

1. **Hardware** — 2× B200, 4× B200, 8× B200 (or NVL72 partition).
2. **Model** — Qwen3-MoE 235B-A22B FP4 (or FP8 if FP4 path not ready).
3. **Runtime** — SGLang 0.5+ V1 (best DeepEP-style integration for non-DeepSeek MoE) or vLLM 0.22+.
4. **Bench at three EP degrees** — EP=2, EP=4, EP=8. Same batch=64, prompt=1024, output=256, iterations=100 with 20 warmup.
5. **Profile one EP=4 run** with Nsight Systems. Identify all-to-all fraction of step time.
6. **Plot** per-replica throughput, per-GPU throughput, and all-to-all overhead percentage.
7. **Identify** the scaling-efficiency curve.

Pass criterion: you can defend EP=4 vs EP=8 for a chat product at concurrency 64 with measured numbers.

---

## Self-check

1. For DeepSeek V3.1 at EP=8, batch=64, predict the per-layer all-to-all time on NVL72 NVLink 5 (assume 28 KB messages × 7 destinations).
2. Why does NVLink 5's 2× bandwidth improvement specifically help MoE EP all-to-all more than dense TP all-reduce?
3. A teammate measures expert utilization on GPU 0 as [180, 95, 110, 75, 140, 88, 200, 95]. What's the imbalance ratio? Should you enable token-level rebalancing?
4. EP=8 vs TP=8 for DeepSeek V3.1: which has more cross-GPU communication overhead? Why?
5. At batch=2 on NVL72 EP=64, the per-step gating + all-to-all overhead per layer is ~80 μs. Per layer compute is ~30 μs. What's wrong, and what's the fix?

---

## References

* DeepSeek V3 technical report (EP section) — [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)
* DeepEP library — [github.com/deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP)
* DeepSeek-MoE paper — [arXiv:2401.06066](https://arxiv.org/abs/2401.06066)
* Switch Transformers (original MoE EP discussion) — [arXiv:2101.03961](https://arxiv.org/abs/2101.03961)
* "GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding" — [arXiv:2006.16668](https://arxiv.org/abs/2006.16668) — the foundational EP paper
* Tutel (MoE communication library) — [github.com/microsoft/tutel](https://github.com/microsoft/tutel)
* SGLang DeepSeek serving guide — [sgl-project.github.io](https://sgl-project.github.io/)
* NCCL all-to-all documentation — [docs.nvidia.com/deeplearning/nccl/](https://docs.nvidia.com/deeplearning/nccl/)

Cross-references:

* [Part 2 → Lecture 04 — Tensor parallelism](../Part%202%20-%20Dense%20at%20Hopper/Lecture-04.md) — for TP-vs-EP comparison
* [Phase 5 → GPU Infrastructure → Long-Context-MoE-Foundation-Training → 05 MoE Systems & Infrastructure](../../../1.%20GPU%20Infrastructure/Nvidia%20GPU/HPC%20Setup/Long-Context-MoE-Foundation-Training/05-MoE-Systems-Infrastructure.md)

---

## Current as of 2026-06

NCCL 2.30+, DeepEP latest, SGLang 0.5+ MoE path, vLLM 0.22+ V1 MoE support, NVL72. Refresh when DeepEP 2.x or successor lands.

---

## Next

* Next: [Lecture 04 — Disaggregated prefill / decode](Lecture-04.md)
* Previous: [Lecture 02 — Blackwell hardware story](Lecture-02.md)
* Up: [Part 3 — MoE at Blackwell](README.md)
