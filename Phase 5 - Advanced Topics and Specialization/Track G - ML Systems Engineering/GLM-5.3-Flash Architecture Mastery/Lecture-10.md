# Module 10 — Kernel Roofline & Serving Decisions

**Collection:** [GLM-5.3-Flash Architecture Mastery](README.md) | **Previous:** [← Module 09](Lecture-09.md) | **Next:** [Module 11 →](Lecture-11.md)

---

Modules 01–09 gave you the arithmetic to *predict* cost. This module is about turning those predictions into a profiling plan — organized as falsifiable hypotheses about seven distinct regions of the model, because "optimize GLM-5.3-Flash" is not an experiment and "is the MoE dispatch path launch-bound at batch size 4" is.

---

## Learning objectives

By the end of this module you should be able to:

1. Write the per-operator lower-bound roofline and apply it before profiling anything.
2. State a specific, falsifiable hypothesis for each of the seven regions this architecture introduces.
3. Explain why a network's latency is governed by its critical path, not by summing per-GPU capability numbers.
4. Explain what prefill/decode disaggregation requires for this specific hybrid state, beyond what a conventional transformer needs.
5. Explain why reducing padded work can fail to move latency at all, and when it should.

---

## 1. The lower-bound model, before you profile anything

For any single operator:

```text
   t_operator  ≳  max( operations / effective_compute_rate ,  bytes_moved / effective_bandwidth )
```

This is the same roofline argument [Hardware-Aware LLM Quantization — Module 01](../Hardware-Aware%20LLM%20Quantization/Lecture-01.md) develops in full — compute a prediction *before* touching a profiler, then use the profiler to explain the gap between prediction and measurement, not to discover the prediction from scratch. Apply it region by region:

```text
   MoE expert GEMM             :  compute-bound at large batch (many tokens per expert),
                                   bandwidth-bound at batch 1 (Module 02's traffic argument)

   KDA recurrent update         :  the state is tiny (128×128/head) — almost certainly
                                   bandwidth/launch-bound, not compute-bound, at any
                                   realistic batch size

   MLA absorbed attention       :  depends on regime — Module 05 §5 already flagged
                                   this as compute-bound in prefill, bandwidth-bound
                                   in decode; profile BOTH, don't assume one

   DSA indexer scoring          :  scales with T/4 candidates (Module 06 §5) —
                                   likely bandwidth-bound on the candidate-latent read
```

Then inspect launches, dependencies, synchronization, and exposed communication. **A prediction that matches measurement tells you the mechanism you modeled is the real bottleneck. A prediction that doesn't match tells you something you didn't model — launch overhead, synchronization stalls, a fallback kernel path — is what's actually binding**, which is a more valuable finding than confirmation would have been.

---

## 2. Seven regions, seven hypotheses

Organize profiling around specific, falsifiable claims — not "make the model faster":

| Region | Hypothesis to test |
|---|---|
| **MoE projections** | Expert-row occupancy and padding determine GEMM efficiency; weight reuse across a batch's routing collisions determines actual traffic ([Module 02 §5](Lecture-02.md)) |
| **KDA decode** | State read/write traffic, register pressure, and spills dominate — not the arithmetic of the recurrence itself ([Module 03](Lecture-03.md)) |
| **KDA prefill** | Chunk utilization and intermediate-storage overhead from the composed-transition computation ([Module 04 §1](Lecture-04.md)) determine whether chunking actually beats a naive loop at your chunk size |
| **DSA indexer** | Candidate-pool scan traffic and top-k selection cost scale with `T/4`, not with the fixed selection budget `K` ([Module 06 §5](Lecture-06.md)) |
| **Sparse MLA** | Gather locality (are selected positions' latents contiguous or scattered in cache?) and latent reuse determine achieved bandwidth on the absorbed-form computation ([Module 05 §3](Lecture-05.md)) |
| **mHC + normalization** | Small-kernel launch overhead and redundant activation traffic from many cheap collapse/mix operations accumulate even though each individual operation is inexpensive ([Module 07 §4](Lecture-07.md)) |
| **Multi-GPU execution** | Exposed (non-overlapped) collectives, CPU staging, and NUMA placement — not raw per-GPU compute or bandwidth — determine end-to-end latency on this specific 8-GPU, PCIe Gen4, no-P2P topology |

Every one of these traces to a specific derivation earlier in this course — that traceability is what turns "profile the model" into a plan you can actually execute and interpret.

---

## 3. Critical path, not summed capability

```text
   WRONG:   "8 GPUs × 1792 GB/s each = 14.3 TB/s of aggregate bandwidth,
             so the model should serve at roughly 14.3TB/s ÷ bytes/token."

   RIGHT:   latency is governed by the DEPENDENCY CHAIN through the
            computation — including every point where one GPU must
            wait for data from another before it can proceed.
```

This matters specifically on the stated deployment topology: **PCIe Gen4, no P2P**. Without peer-to-peer GPU-to-GPU transfers, inter-GPU communication for any operation requiring cross-device data (an all-reduce after tensor-parallel attention, an all-to-all for MoE expert dispatch, the MLA latent-replication pattern from [Module 09 §4](Lecture-09.md) if your implementation needs it) routes through a slower path than P2P or NVLink would provide — and if that communication is not **overlapped** with compute on other GPUs, it sits directly on the critical path as pure added latency, invisible to any calculation that only sums per-GPU capability.

```text
   ┌────────────────────────────────────────────────────────────────┐
   │  On a no-P2P, PCIe Gen4 topology specifically, ask for every     │
   │  cross-GPU operation: is this communication OVERLAPPED with      │
   │  useful compute on other devices, or is it EXPOSED — sitting     │
   │  on the critical path with nothing else happening while it       │
   │  waits? Exposed communication is where "aggregate bandwidth"     │
   │  arithmetic silently stops applying.                              │
   └────────────────────────────────────────────────────────────────┘
```

---

## 4. Prefill/decode disaggregation, correctly scoped

Disaggregation places prefill and decode workloads on separate serving resources so each can be provisioned and scheduled independently — SGLang implements this as a serving-system capability, and the general pattern is standard practice for large-scale serving. Two things specific to this architecture change what a correct implementation needs:

**The handoff carries more than a KV cache.** [Module 08 §3](Lecture-08.md) already established the full state inventory a rollback must reconcile; the exact same inventory is what a prefill→decode handoff must transfer intact:

```text
   handoff payload  =  MLA latent cache (token-indexed, truncatable — Module 05)
                     +  KDA recurrent state (per-head, NOT token-indexed — Module 03)
                     +  convolution state (Module 04 §3)
                     +  DSA indexer selection metadata (Module 06)

   A disaggregation design that only moves "the KV cache" between prefill
   and decode workers is repeating the same incomplete-state mistake
   Module 08 named for speculative rollback — just at a different point
   in the serving pipeline.
```

**Placement is not free just because roles are split.** For your current full eight-GPU placement, splitting prefill and decode roles onto (say) 4 GPUs each **does not create two independently-resident full replicas of the model** — you have divided one placement's resources between two roles, not doubled your capacity. Two genuinely independent replicas of the full 8-GPU placement would require 16 GPUs. Confusing "we now have separate prefill and decode services" with "we now have more serving capacity" is a capacity-planning error with the same shape as [Module 09](Lecture-09.md)'s `÷8` trap: a design decision that looks like it multiplies a resource, when it actually just reallocates a fixed one.

---

## 5. Treat every hypothesis as an experiment

Two concrete examples of predictions that can go either way, stated as a reminder that the point of §2's table is to *run* the experiments, not to assume their outcomes:

```text
   "Reducing padded MoE arithmetic will speed up decode."

     Can be TRUE:  if MoE GEMM efficiency is the binding constraint at
                   your batch size.
     Can be FALSE: if the request is actually launch-bound or
                   communication-bound — removing padded FLOPs doesn't
                   touch either of those, and latency barely moves.


   "Increasing batch size will make [format/kernel choice] the better one."

     Batch size changes expert-routing collision rate (Module 02 §5),
     which changes achieved weight reuse, which can flip which kernel
     design is actually faster at the new operating point — a change
     in REGIME, not just in scale.
```

Neither outcome is more "correct" in the abstract — both are real possibilities, and which one holds for your specific deployment is exactly what profiling is for. Report the hypothesis, the measurement, and the actual outcome, including when the outcome was "no measurable effect" — a documented negative result about where the bottleneck *isn't* is exactly as valuable to a team debugging this architecture as a positive result about where it is.

---

## Checkpoint

You should now be able to:

1. Apply the per-operator roofline lower bound before running a profiler, for at least three of the seven regions.
2. State a specific, falsifiable hypothesis for each of the seven regions in §2.
3. Explain why summing eight GPUs' bandwidth or compute overstates achievable throughput on a no-P2P topology.
4. List the full state-handoff payload a correct prefill/decode disaggregation must transfer for this architecture.
5. Explain why splitting one 8-GPU placement's roles is not the same as adding capacity.

---

## Ship it

Produce a **profiling hypothesis log** for your own deployment: one row per region from §2's table, with your predicted bottleneck (from the roofline model in §1), the actual measurement, agreement or disagreement, and — critically — at least one row where the hypothesis was **wrong**, with an explanation of what you learned about the real bottleneck instead. A log with zero wrong hypotheses is a sign you profiled things you already understood, not a sign of a well-optimized system.

---

## Current as of

* **Timeless:** the per-operator roofline lower bound, the critical-path-not-summed-capability argument, the general disaggregation pattern and its capacity-planning trap.
* **Deployment-specific:** the "PCIe Gen4, no P2P, 8× RTX 5090" topology is this course's named case study — re-derive the exposed-versus-overlapped communication analysis for any different interconnect (NVLink, InfiniBand) before reusing this module's conclusions about where communication becomes a bottleneck.

---

**Next:** [Module 11 — Correctness as Architecture Mastery →](Lecture-11.md)
