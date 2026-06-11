# Part 2 · Lecture 07 — Inside the Communication Layer: NCCL, Custom All-Reduce, and the vLLM Communicator Stack

## Overview

Lecture 04 established that tensor parallelism runs **two all-reduces per layer** and that on `TP=8` those collectives can eat ~25% of step time. This lecture opens the box underneath that fact: **how a production runtime actually moves bytes between GPUs**, why it does not just call NCCL and stop, and which path wins in which regime.

We use **vLLM's `distributed/device_communicators/` layer** as the worked example because it is the most-read open-source implementation of this idea. The exact filenames and APIs drift between releases — treat them as a map of the *structure*, not a pinned spec — but the architecture is stable across runtimes (SGLang and TRT-LLM layer their collectives the same way).

This lecture covers:

1. The three-layer communicator architecture.
2. The backend primitives — NCCL, the CUDA glue, and shared memory.
3. Why all-reduce has *many* implementations, and the small-message problem.
4. The optimized paths — custom one-shot/two-shot all-reduce, fused all-reduce+norm, and the fallback ladder.
5. The routing brain — how a runtime *chooses* a path at runtime.
6. all2all — the MoE collective (a forward pointer to Part 3).
7. The mental model: a runtime is a **dynamic communication optimizer**.
8. Inference-engineering takeaways — diagnosing and tuning the collective path.

By the end you should be able to look at a multi-GPU decode trace, identify which collective path is running, and explain whether it is the right one for that message size and topology.

---

## 1. The three-layer architecture

A serving runtime does not expose "NCCL" to the model. It exposes an **abstraction** — `tensor_model_parallel_all_reduce(x)` — and routes that call through three layers:

<div class="lecture-map" markdown>

| Layer | Role | Example modules |
|-------|------|-----------------|
| **Orchestration / router** | Pick the best path for *this* tensor, topology, and budget; fall back if unavailable | `cuda_communicator.py` (the dispatcher) |
| **Communication engines** | *How* the collective happens | `custom_all_reduce.py`, `flashinfer_all_reduce.py`, `quick_all_reduce.py`, `all2all.py` |
| **Backend primitives** | Talk to hardware / libraries directly | `pynccl.py` (NCCL), `cuda_wrapper.py` (CUDA runtime/streams), `shm_*` (host shared memory) |

</div>

The key idea: **the engine is chosen per call, not once at startup.** A prefill all-reduce and a decode all-reduce on the same GPUs may take different paths because their message sizes differ by 100×.

---

## 2. The backend primitives

These are the bottom layer — thin bindings, no policy.

* **`pynccl` — NCCL bindings.** Direct access to `ncclAllReduce`, `ncclAllGather`, `ncclReduceScatter`, plus grouped `ncclSend`/`ncclRecv` (NCCL has no native all-to-all collective; it is composed from send/recv pairs). NCCL is the stable, works-everywhere baseline (Lecture 04 §2). vLLM wraps it directly rather than going through `torch.distributed` for the hot path, so it controls streams and avoids framework overhead.
* **`cuda_wrapper` — the CUDA glue.** Stream creation, `cudaMemcpyAsync`, event/handle management, device context. This is "Python safely driving CUDA streams"; the collectives launch onto these streams so communication can overlap compute.
* **`shm_*` — host shared memory.** CPU↔CPU transfer between *worker processes* (the engine/driver and its TP workers, or Ray actors). This is **not** the GPU hot path — it carries control messages, small metadata, and CPU-offload tensors between processes on one node. Important for startup and scheduling, irrelevant to per-token latency.

> The split matters: when you see a collective in a profile, it is almost always a GPU primitive (NCCL or a custom kernel). `shm` traffic is host-side and shows up on the CPU timeline, not the GPU one.

---

## 3. Why all-reduce has many implementations — the small-message problem

NCCL's **ring all-reduce** is *bandwidth-optimal*: for a message of `N` bytes across `P` GPUs it moves `~2N(P−1)/P` bytes per GPU and saturates NVLink. It is the right choice for **large** messages — i.e., **prefill**, where the all-reduce carries `[many tokens × hidden]`.

But **decode is different.** At batch=1, one token, the per-layer all-reduce message is tiny — `[1 × 8192] × 2 B ≈ 16 KB`. For a message that small:

* Ring all-reduce is **latency-bound, not bandwidth-bound.** You pay `2(P−1)` sequential hops of kernel-launch + handshake latency to move 16 KB. The links are nearly idle; you are paying for *round-trips*, not *bytes*.
* At `TP=8` that is 14 hops per all-reduce × 2 per layer × 80 layers = **2,240 latency-bound hops per token** — exactly the "comm-bound at TP=8 decode" result from Lecture 04, now explained mechanistically.

This is the entire reason a runtime ships more than NCCL: **small-message latency.**

---

## 4. The optimized paths and the fallback ladder

### 4.1 Custom all-reduce (the small-message win)

The headline optimization. For small messages on GPUs with **full peer-to-peer / NVLink** connectivity, a custom CUDA kernel does a **one-shot** (or two-shot) all-reduce: each GPU reads its peers' buffers directly over NVLink and reduces in a **single kernel launch**, instead of NCCL's multi-hop ring.

* **One-shot:** every GPU reads all peers → reduces → writes its result. Best for the smallest messages (decode).
* **Two-shot (reduce-scatter + all-gather):** better as the message grows but still below the ring's crossover.
* **Requires P2P/NVLink.** On PCIe-only boxes (no full peer access), it cannot run and the runtime falls back to NCCL. **Topology decides the path.**

This is the single most important collective optimization for **multi-GPU decode latency**. In vLLM it is on by default when supported (`VLLM_USE_CUSTOM_ALL_REDUCE`), and disabling it is the first A/B test when decode latency looks comm-bound.

### 4.2 Fused all-reduce + norm (FlashInfer)

The next step removes *kernel launches and HBM round-trips* by **fusing** the all-reduce with the surrounding pointwise work (residual add + RMSNorm). Instead of:

```text
all_reduce(x) → write HBM → read HBM → residual+RMSNorm → write HBM
```

a fused kernel does **reduce → residual → norm → write-back** in one pass. [FlashInfer](https://arxiv.org/abs/2501.01005) (the kernel engine from Lecture 05 §2.4) provides this `allreduce_fusion` path. It is gated on the same conditions as custom all-reduce plus a **workspace budget** (see §5) and a fixed tensor layout (contiguous `[tokens, hidden]`). When it does not apply, the runtime falls back.

### 4.3 The ladder

<div class="lecture-map" markdown>

| Priority | Path | Wins when |
|----------|------|-----------|
| 1 | **Fused all-reduce+norm** (FlashInfer) | small/medium msg, contiguous, fits workspace, P2P available |
| 2 | **Custom one-shot/two-shot all-reduce** | small msg (decode), full NVLink/P2P |
| 3 | **NCCL ring** (`pynccl`) | large msg (prefill), or no P2P, or unsupported shape/dtype |
| 4 | **Shared-memory / host fallback** | cross-process control + CPU tensors (not the GPU hot path) |

</div>

A "quick" / lightweight reduce path also exists for small reductions where heavy workspace setup is not worth it; think of it as a thin shortcut between the custom kernel and NCCL.

---

## 5. The routing brain — choosing a path at runtime

Every optimized engine exposes a predicate — conceptually `should_use_this_path(tensor)` — checked **per call**. The checks are always some subset of:

* **Is it a CUDA tensor, contiguous, and the expected rank/shape** (`[tokens, hidden]`)?
* **Is the message within the workspace budget?** A fused kernel pre-allocates a scratch buffer sized for a maximum token count: `max_tokens ≈ workspace_bytes / (hidden × dtype_bytes)`. A batch larger than that cannot be fused in one launch → fall back.
* **Is the backend installed and is P2P/NVLink available?** If FlashInfer is absent or peer access is off, the path disables itself at init.
* **world_size > 1** (no collective needed on a single GPU).

If **any** check fails, the router drops to the next rung of the ladder. This is why the same model can show different collective kernels in prefill vs decode, or on an NVLink box vs a PCIe box — *the routing is dynamic and topology-aware.*

---

## 6. all2all — the MoE collective (forward pointer to Part 3)

Everything above is the **tensor-parallel** collective set (all-reduce / all-gather / reduce-scatter). Mixture-of-Experts adds a different one: **all2all**, used for **expert dispatch and combine**.

```text
tokens (after routing)
  GPU0 → experts {1,3}     GPU1 → experts {2,5}     GPU2 → experts {0,4}
        └──────────────── all2all exchange ────────────────┘
  each GPU now holds the tokens routed to ITS experts
```

Each token is shipped to whichever GPU holds its chosen expert (dispatch), computed, then shipped back (combine). all2all is the **dominant communication cost of expert parallelism** and behaves very differently from all-reduce (irregular, payload-dependent volume). Part 3 — MoE at Blackwell — treats it in depth; here, just register that it lives in the **same communicator layer** (`all2all.py`) and goes through the same fallback discipline.

---

## 7. The mental model

> A serving runtime is **not "using NCCL."** It is a **dynamic communication optimizer** that picks, per collective, the cheapest correct path for the message size, dtype, layout, and GPU topology — fused kernel → custom all-reduce → NCCL ring → host fallback — and degrades gracefully when the fast path's preconditions are not met.

Hold this next to the roofline mental model from Part 1: just as compute has a memory-bound vs compute-bound regime, **communication has a latency-bound (small message) vs bandwidth-bound (large message) regime**, and the runtime switches collective algorithms across that boundary the same way it switches GEMV vs GEMM.

---

## 8. Inference-engineering takeaways

* **The collective path is a real latency lever at `TP ≥ 4` decode.** Small-message all-reduce is latency-bound; the custom one-shot kernel can materially cut per-layer comm vs NCCL ring. This is the concrete mechanism behind Lecture 04's "TP=8 sacrifices ~25% to communication."
* **Topology gates the fast path.** Custom/fused all-reduce needs full P2P/NVLink. On a PCIe-only or partially-connected box it silently falls back to NCCL — so the *same* `TP=8` config can have very different decode latency on different chassis. Verify NVLink/NVSwitch (Lecture 04 §3) before blaming the model.
* **Diagnose by trace.** In Nsight Systems, a healthy small-message decode shows a **custom all-reduce kernel** (or a fused allreduce-norm), *not* a chain of NCCL ring kernels. NCCL ring kernels dominating the decode timeline = the fast path is disabled (no P2P, unsupported dtype/shape, or it was turned off).
* **The flags to know:** `VLLM_USE_CUSTOM_ALL_REDUCE` (on by default; disable to A/B against NCCL), `NCCL_DEBUG=INFO` (confirms NCCL topology/algorithm), and the FlashInfer attention/fusion backend selector from Lecture 05. Pin and record these in the bench harness, like every other version.
* **Don't over-index on prefill.** For large prefill messages, plain NCCL ring is already near-optimal — the custom/fused paths buy little there. The win is concentrated in **decode**, which is where your $/MTok lives.

---

## Lab — see the collective path switch

Extend the Part 2 bench harness:

1. **Serve Llama 3.3 70B FP8 at `TP=8`** on an NVLink box and capture a **decode** trace (Nsight Systems, ~50 steps). Identify the per-layer all-reduce kernel — custom AR or NCCL ring?
2. **Disable the custom path** (`VLLM_USE_CUSTOM_ALL_REDUCE=0`), re-trace, and measure the **TPOT delta**. Attribute it to the NCCL ring's small-message latency.
3. **Repeat for prefill** (long prompt) and show the delta is much smaller — the message is now bandwidth-bound and NCCL ring is competitive.
4. **(If available)** run the same config on a PCIe-only box and show the fast path never engages.

Pass criterion: a one-page report that states, with trace evidence, which collective path ran in prefill vs decode, what the custom-AR-off penalty was, and why it differed between phases.

---

## Self-check

1. Why does a runtime ship a custom all-reduce when NCCL already implements all-reduce? Answer in terms of message size and what's actually being paid for at decode.
2. A teammate reports that `TP=8` decode is slow on a new server but fast on the old one, same model and vLLM version. Name the first thing you'd check and why.
3. What does fusing all-reduce with RMSNorm save, concretely (count the HBM round-trips before and after)?
4. Why is all2all fundamentally harder to optimize than all-reduce? (Think about whether the per-rank payload size is known ahead of time.)
5. In an Nsight decode trace you see a chain of NCCL ring kernels per layer instead of a single custom-AR kernel. List three causes.

---

## References

* vLLM distributed communicators — [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm) (`vllm/distributed/device_communicators/`)
* NCCL — [docs.nvidia.com/deeplearning/nccl](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html) (ring/tree algorithms, all-reduce/all2all)
* FlashInfer (fused attention + collective kernels) — [arXiv:2501.01005](https://arxiv.org/abs/2501.01005)
* Cross-reference: [Lecture 04 — Tensor parallelism](Lecture-04.md) (the all-reduce cost model) · [Lecture 05 §2.4 — FlashInfer](Lecture-05.md) · [Part 3 — MoE at Blackwell](../Part%203%20-%20MoE%20at%20Blackwell/README.md) (all2all / expert parallelism)

---

## Current as of 2026-06

Reflects the vLLM `device_communicators` architecture (custom all-reduce, FlashInfer fused all-reduce, NCCL baseline, all2all for EP) as of the 0.22-era lineage. Exact module names and the fused-path availability shift between releases — re-pin against the installed vLLM version. Refresh when a new intra-node collective primitive (e.g., NVLink-native switch collectives, NVLS) becomes the default path.

---

## End of Part 2

You have now seen the dense-at-Hopper stack from model anatomy down to the per-collective kernel. The harness, precision recipes, TP-scaling discipline, serving knobs, and the communication-path lens all carry forward — Part 3 reuses them and extends to sparse MoE on Blackwell, where all2all becomes the dominant collective.

* Next: [Part 3 — MoE at Blackwell](../Part%203%20-%20MoE%20at%20Blackwell/README.md) — DeepSeek V3.1 + Qwen3-MoE 235B-A22B anchors, EP, FP4, disaggregated P/D
* Previous: [Lecture 06 — Long context at 128K on Hopper](Lecture-06.md)
* Up: [Part 2 — Dense at Hopper](README.md)
