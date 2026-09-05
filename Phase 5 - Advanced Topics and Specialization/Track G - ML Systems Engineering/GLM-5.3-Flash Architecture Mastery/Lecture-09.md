# Module 09 — The 8-GPU Memory Model

**Collection:** [GLM-5.3-Flash Architecture Mastery](README.md) | **Previous:** [← Module 08](Lecture-08.md) | **Next:** [Module 10 →](Lecture-10.md)

---

Every prior module derived one mechanism's cost in isolation. This module adds them up — for a concrete deployment: **8× RTX 5090, PCIe Gen4, no P2P**, serving the NVFP4 checkpoint (the `sparkinfer-frontier` case study this course's capstone builds toward). The result is not "total memory ÷ 8." Tensor-parallel serving of a latent-attention architecture has a specific, well-documented replication trap, and this module exists to make sure you don't fall into it while building your own deployment's budget.

---

## Learning objectives

By the end of this module you should be able to:

1. Compute lower-bound weight storage at several precisions and reconcile them against a real reported deployment footprint.
2. Compute the KDA recurrent-state and MLA latent-cache budgets for a stated context length, per GPU.
3. State the tensor-parallel MLA replication trap precisely, and explain why naive `÷8` division is wrong.
4. Build a complete per-GPU memory equation, term by term, correctly classifying each term as sharded, replicated, request-dependent, or transient.

---

## 1. Weight storage: lower bounds, then reality

For a nominal 320B parameters, uniform-precision lower bounds:

```text
   2 bytes/parameter  (BF16/FP16)   :   596 GiB
   1 byte/parameter   (FP8)          :   298 GiB
   0.5 bytes/parameter (naive INT4)  :   149 GiB
```

```text
   ┌────────────────────────────────────────────────────────────────┐
   │  These are ARITHMETIC LOWER BOUNDS from parameter count alone —  │
   │  not complete checkpoint or runtime sizes. Real storage adds     │
   │  block scales, and most deployments keep selected tensors        │
   │  (embeddings, LM head, some attention projections) at higher     │
   │  precision than the bulk of the model — exactly the allocation   │
   │  decision Hardware-Aware LLM Quantization — Module 11 formalizes │
   │  as a knapsack problem rather than a single global bit-width.     │
   └────────────────────────────────────────────────────────────────┘
```

**Reconcile against the actual format.** [Hardware-Aware LLM Quantization — Module 02](../Hardware-Aware%20LLM%20Quantization/Lecture-02.md) derives NVFP4's *real* cost as 4.5 bits — not the naive 4-bit / 0.5-byte assumption above — because of its per-16-element E4M3 block scale:

```text
   NVFP4, real (0.5625 bytes/param)  :  320e9 × 0.5625  =  180 GB  =  167.64 GiB
```

**Cross-check against the reported deployment.** The `sparkinfer-frontier` repository reports approximately **185 GiB of model data**, with a reported residency of **24.8 GB per GPU** across 8 GPUs. Two consistency checks, both worth running on any reported figure before trusting it:

```text
   uniformity check:   185 GiB / 8  =  23.125 GiB/GPU  =  24.83 GB/GPU
                       ✓ matches the reported 24.8 GB/GPU almost exactly —
                         consistent with roughly uniform 8-way weight sharding

   format check:       185 GiB reported  ÷  167.64 GiB pure-NVFP4 floor
                       =  1.104×   (a 10.4% overhead)
                       ✓ plausible: block-scale metadata plus embeddings/LM-head/
                         selected attention tensors kept at a higher precision
                         than the bulk NVFP4 body
```

```text
   ┌────────────────────────────────────────────────────────────────┐
   │  These are figures the sparkinfer-frontier repository reports —  │
   │  not measurements taken for this course. Treat the arithmetic    │
   │  above as a CONSISTENCY CHECK on a third-party report, which is  │
   │  a different epistemic status than an independent measurement.   │
   │  The check passing is reassuring; it is not the same as having   │
   │  measured it yourself.                                            │
   └────────────────────────────────────────────────────────────────┘
```

---

## 2. KDA state: fixed in context length, not free

From [Module 03 §1](Lecture-03.md): 64 heads × 128×128 state matrices, across 34 KDA layers, at FP32 (recurrent state is numerically sensitive — keep it at higher precision even in an otherwise-quantized deployment):

```text
   M_KDA  =  34 layers × 64 heads × 128 × 128 × 4 bytes
          =  142,606,336 bytes
          =  136.0 MiB                    ← for ONE request, ALL KDA layers, ALL heads
```

```text
   ┌────────────────────────────────────────────────────────────────┐
   │  This figure excludes convolution state (Module 04 §3),          │
   │  allocator overhead, and any extra snapshots a speculative-       │
   │  decoding rollback scheme keeps (Module 08 §3). Treat 136 MiB     │
   │  as a floor for one request's KDA state, not the complete cost.   │
   └────────────────────────────────────────────────────────────────┘
```

With ideal head-sharding across 8 GPUs (each GPU owns 8 of the 64 heads, for every layer):

```text
   136 MiB / 8  =  17 MiB / GPU / request
```

Small in isolation — but **concurrency multiplies it directly** (100 concurrent requests → 1.7 GiB/GPU just for KDA state), and any rollback scheme that checkpoints multiple speculative prefixes ([Module 08 §3](Lecture-08.md)) multiplies it again. Size this term against your actual target concurrency, not against one request.

---

## 3. MLA cache: linear in context length

From [Module 05 §2](Lecture-05.md): 512-wide BF16 latent, per token, per MLA layer, across all 11 MLA layers:

```text
   M_MLA(T)  =  11 layers × T tokens × 512 × 2 bytes
```

| Context `T` | Logical latent cache, one request |
|---:|---:|
| 128K = 131,072 | **1.375 GiB** |
| 512K = 524,288 | **5.5 GiB** |
| 1M = 1,048,576 (the configured max) | **11 GiB** |

```text
   ┌────────────────────────────────────────────────────────────────┐
   │  These are calculated LOGICAL cache sizes from the architecture, │
   │  before replication (§4), metadata, or allocator effects — the   │
   │  same "logical vs. actual resident" gap Module 09 of the         │
   │  Quantization course draws for the KV cache generally. Treat     │
   │  these numbers as the floor a real deployment must clear, not     │
   │  the number you'll actually observe in nvidia-smi.                │
   └────────────────────────────────────────────────────────────────┘
```

---

## 4. The trap: you cannot divide every term by 8

The single most damaging mistake in building this budget is assuming uniform tensor-parallel sharding applies to every term the way it applies to weights:

```text
   WRONG:   M_cache,per-GPU  =  M_cache,logical / 8
```

**Why this fails specifically for MLA.** Conventional head-parallel tensor parallelism shards *heads* across GPUs — each GPU owns a subset of attention heads and only needs the K/V data for *its* heads. But MLA's entire memory advantage ([Module 05 §2](Lecture-05.md)) comes from caching one **shared latent** `c_t` that every head's key and value get reconstructed *from* — the latent is not head-specific. A head-parallel scheme that shards heads across 8 GPUs can require **every participating GPU to hold the full shared latent representation**, because any GPU might need to reconstruct any head's key/value from it, depending on how the attention computation is partitioned. This replication requirement for tensor-parallel latent attention is exactly the problem documented in the literature on this specific scaling pattern — it is not a hypothetical edge case, it is the default behavior of a naive head-parallel MLA implementation.

```text
   weights            →  genuinely shardable — each GPU holds a DISJOINT slice
                          (this is what makes the §1 ÷8 check work)

   MLA latent cache   →  can require FULL REPLICATION across GPUs under a
                          naive head-parallel scheme — NOT automatically ÷8

   KDA state          →  shardable by HEAD, similar to weights, IF the
                          implementation actually partitions heads across
                          GPUs rather than replicating the full state
                          (verify this against your actual runtime — don't
                          assume it from the weight-sharding pattern alone)
```

**The question to ask for every single term, before writing a memory budget:** is it sharded, replicated, request-dependent, or transient? Guessing from how weights shard is exactly the mistake this trap is named for.

---

## 5. The complete per-GPU equation

```text
   M_GPU  =  M_weights,local
           +  M_MLA,local
           +  M_KDA,local
           +  M_indexer
           +  M_workspace
           +  M_graphs
           +  M_allocator
```

| Term | Typical classification | Where it's derived |
|---|---|---|
| `M_weights,local` | sharded (disjoint slice per GPU) | §1 |
| `M_MLA,local` | **verify — may be REPLICATED, not sharded** (§4) | §3 |
| `M_KDA,local` | sharded by head **if your runtime partitions heads**; verify | §2 |
| `M_indexer` | request-dependent — the DSA selection metadata from [Module 06](Lecture-06.md), scaled by concurrent requests | Module 06 |
| `M_workspace` | transient — attention/FFN intermediate buffers, freed after use | — |
| `M_graphs` | fixed overhead if using CUDA graph capture for decode | — |
| `M_allocator` | fixed overhead — fragmentation, reserved blocks | — |

```text
   ┌────────────────────────────────────────────────────────────────┐
   │  Write this equation in LOCAL (per-GPU) allocations, term by     │
   │  term — never start from a global total and divide. A budget     │
   │  built by dividing an aggregate is only correct for the terms     │
   │  that happen to shard uniformly, and §4 just showed you MLA's     │
   │  cache is not guaranteed to be one of them.                        │
   └────────────────────────────────────────────────────────────────┘
```

### Worked example: feasibility at 1M context, this deployment

```text
   per-GPU weights (from §1's cross-check)         ≈  23.1 GiB
   MLA cache at 1M context, IF REPLICATED (§4)      =  11.0 GiB   ← on EVERY GPU, not ÷8
   KDA state, 100 concurrent requests (§2)          ≈   1.7 GiB
   indexer + workspace + graphs + allocator         ≈   (measure — don't guess)
   ─────────────────────────────────────────────────────────────
   running total, before workspace/indexer          ≈  35.8 GiB   >  32 GiB card capacity
```

That single arithmetic line is the entire point of this module: **a naive "divide the MLA cache by 8" version of this budget would have shown comfortable headroom** (11 GiB ÷ 8 ≈ 1.4 GiB, leaving the running total near 26.2 GiB). The correct, unreplicated accounting shows the deployment is **already over budget on a 32 GB card before workspace, indexer overhead, or a second concurrent request's worth of KDA state are even counted.** Whether your actual runtime replicates the MLA cache this way is an empirical question about its specific tensor-parallel implementation — but the direction of the error if you guess wrong is always the same: naive division makes an infeasible deployment look feasible, never the reverse.

---

## Checkpoint

You should now be able to:

1. Compute weight-storage lower bounds at three precisions and reconcile a reported real-world figure against the NVFP4-specific 0.5625 bytes/param rate.
2. Compute the KDA state budget for a stated concurrency and the MLA cache for a stated context length.
3. State exactly why MLA's shared latent breaks the naive tensor-parallel `÷8` assumption that weights satisfy.
4. Build a complete per-GPU memory equation and correctly classify each term.
5. Explain why guessing "replicated" versus "sharded" wrong always fails in the same direction (understating usage), never the other.

---

## Ship it

This is **Stage 7 of the [capstone ladder](Lecture-12.md)**: build the full per-GPU memory equation for your own target deployment (context length, concurrency, and GPU count), predict per-GPU usage term by term, then measure actual per-GPU allocation under load. For every term where predicted and measured disagree by more than a small margin, **identify which classification (sharded/replicated/request-dependent/transient) you got wrong** — that discrepancy, explained, is worth more than the prediction matching on the first try.

---

## Current as of

* **Timeless:** the sharded-vs-replicated classification discipline, and the specific tensor-parallel latent-attention replication trap in §4.
* **Checkpoint-and-deployment-specific:** the 320B/18B parameter split ([Module 01](Lecture-01.md)), the 34-KDA/11-MLA layer split, and the `sparkinfer-frontier` 185 GiB / 24.8 GB-per-GPU figures are cited from the checkpoint configuration and the named repository respectively — re-verify both against current sources before relying on them for a capacity decision, and always re-derive whether your specific runtime replicates or shards the MLA cache rather than assuming either.

---

**Next:** [Module 10 — Kernel Roofline & Serving Decisions →](Lecture-10.md)
