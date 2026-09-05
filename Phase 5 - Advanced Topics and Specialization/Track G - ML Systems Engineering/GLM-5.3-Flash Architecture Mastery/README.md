# GLM-5.3-Flash Architecture Mastery

<div class="course-identity mlsys" markdown="1">
<div class="course-identity__icon">Sₜ</div>
<div markdown="1">
<p class="course-identity__eyebrow">Phase 5 · ML Systems Engineering · Principal-Engineer Deep Dive</p>
<p class="course-identity__title">Derive the computations, predict the memory traffic, find the real serving bottleneck, and modify the implementation without silently changing the model.</p>
<p class="course-identity__meta">Artifact: a validated per-GPU memory model + one correctness-gated optimization on an 8× RTX 5090 deployment · Measure: bytes/token by mechanism, per-operator roofline, output/state agreement across execution paths</p>
</div>
</div>

> *Understanding an architecture diagram is not the goal. Being able to rebuild the recurrence from five lines, predict the bytes it moves, and prove your kernel didn't quietly change the model — that is the goal.*

GLM-5.3-Flash is not GLM-5.3 with fewer layers or lower precision. Z.ai describes a **redesigned hybrid architecture**: sparse parameter activation (MoE), recurrent sequence memory (KDA), compressed per-token history (MLA), selective historical retrieval (DSA/KPool), and manifold-constrained multi-stream residuals (mHC) — five mechanisms, each attacking a different cost, composed into one decoder. Treating any two of them as interchangeable, or assuming one subsumes another, is the single most common way to misread this model.

**Target deployment:** an 8× RTX 5090 (PCIe Gen4, no P2P) system serving the NVFP4 checkpoint — the `sparkinfer-frontier` case study threaded through Modules 09–12.
**Level:** principal AI engineer. This course assumes you can already read a CUDA profiler and derive a roofline; it does not re-teach either.

**Layer mapping:** L4–L7 — model mathematics through multi-GPU serving. It sits directly on top of [Hardware-Aware LLM Quantization](../Hardware-Aware%20LLM%20Quantization/README.md) (this checkpoint *is* an NVFP4 deployment) and below nothing — this is the architecture the rest of Track G's tooling serves.

**Role targets:** Principal/Staff Inference Engineer · ML Systems Engineer (hybrid architectures) · GPU Runtime Engineer · Model-Serving Correctness Engineer

---

## Prerequisites

| Prerequisite | Why you need it |
|---|---|
| [Hardware-Aware LLM Quantization](../Hardware-Aware%20LLM%20Quantization/README.md) | This checkpoint is served in NVFP4. Module 09 here reuses that course's bits-per-weight arithmetic directly — this course does not re-derive block scaling. |
| [AI Inference Engineer 2026 — Part 3](../AI%20Inference%20Engineer%202026/Part%203%20-%20MoE%20at%20Blackwell/README.md) | General MoE/MLA serving patterns (expert parallelism, disaggregated prefill/decode). This course goes deeper on one specific hybrid model; Part 3 gives you the general serving vocabulary. |
| [Logprobs, Perplexity & KL Divergence](../Logprobs,%20Perplexity%20and%20KL%20Divergence/README.md) | Module 11's correctness matrix leans on KL/agreement grading to distinguish "algebraically equivalent" from "silently different model." |
| Linear algebra fluency | Module 03's recurrence derivation is matrix algebra, not hand-waving. You should be comfortable multiplying out `(I − βkkᵀ)D` without a computer. |
| CUDA + roofline fluency | Module 10 assumes you already know what a memory-bound kernel looks like in Nsight Compute. |

**Pairs with:** [MLSys Deep Dives — Lecture 04](../MLSys%20Deep%20Dives/Lecture-04.md) (Mamba/SSM state-vs-KV-cache economics — the same "recurrent state replaces a growing cache" argument KDA makes here) and [AI Inference Engineer 2026 — Part 4](../AI%20Inference%20Engineer%202026/Part%204%20-%20Optimizing%20a%20Real%20Engine/README.md) (the same measure-before-you-optimize discipline, on a different real engine).

---

## The five mechanisms, kept separate

The single rule that prevents most misreadings of this architecture:

```text
   MoE   decides WHICH PARAMETERS execute           (capacity vs. arithmetic vs. traffic)
   KDA   compresses SEQUENCE HISTORY into fixed state (a memory, not a growing list)
   MLA   compresses the REPRESENTATION per token      (still token-indexed, just smaller)
   DSA   decides WHICH POSITIONS get read             (selection, separate from representation)
   mHC   changes HOW INFORMATION FLOWS between sublayers (connectivity, not computation)
```

They are complementary, not interchangeable. mHC's four residual streams do not divide the 45-layer serial depth by four. Prefill/decode disaggregation is a *deployment* decision, not a sixth mechanism. Every module in this course exists to make one of these five precise enough that you cannot confuse it with its neighbors.

---

## Course Map (12 modules)

<div class="lecture-map" markdown>

| # | Module | The thread |
|---|--------|-----------|
| [01](Lecture-01.md) | **The Exact Model** — checkpoint configuration, the execution trace, why this is a new hybrid model rather than a distilled GLM-5.3 | establish ground truth before deriving anything |
| [02](Lecture-02.md) | **MoE: Capacity, Work, and Traffic** — the sigmoid router with correction bias, clamped SwiGLU experts, and the 304.4B-parameter derivation | three quantities people conflate |
| [03](Lecture-03.md) | **KDA I: The Delta-Rule Recurrence** — the five-step update, the boxed closed form, a worked numeric example, state vs. checkpoint weights | master this most deeply first |
| [04](Lecture-04.md) | **KDA II: Chunked Parallelism & the Full Sublayer** — the affine-composition argument for parallel training, why prefill needs a different kernel than decode, everything the recurrence doesn't tell you | the recurrence is not the whole layer |
| [05](Lecture-05.md) | **MLA: Compressing Per-Token History** — the joint latent, the absorption algebra that avoids re-expanding history, the 64× cache ratio, why NoPE isn't order-blindness | still token-indexed, just smaller |
| [06](Lecture-06.md) | **DSA & KPool: Selective Retrieval** — pooled indexing, the 2,051-slot budget, causality at pool boundaries, why fixed top-k isn't O(1) | pooling is not deletion |
| [07](Lecture-07.md) | **mHC: Manifold-Constrained Residual Streams** — the doubly-stochastic mixing matrix, Sinkhorn normalization, why four streams isn't four attention modules | connectivity, not computation |
| [08](Lecture-08.md) | **Vision, MTP, and Hybrid Serving State** — the multimodal path, why speculative rollback needs to restore recurrent *and* convolution *and* latent state | separate subsystems, separate timers |
| [09](Lecture-09.md) | **The 8-GPU Memory Model** — weight-storage lower bounds, KDA/MLA per-request budgets, the tensor-parallel replication trap, the full per-GPU equation | don't divide everything by 8 |
| [10](Lecture-10.md) | **Kernel Roofline & Serving Decisions** — the per-operator lower-bound model, a profiling hypothesis table by region, disaggregation's full state-handoff requirement | experiments, not predetermined conclusions |
| [11](Lecture-11.md) | **Correctness as Architecture Mastery** — the full test matrix, the prefix/chunk/continuation invariant, specifying the numerical contract before benchmarking | a speedup with no stated contract isn't a result |
| [12](Lecture-12.md) | **Capstone: The KDA-First Mastery Ladder** — eight staged deliverables from one KDA head to a validated end-to-end optimization | build order, not reading order |

</div>

---

## Course Outcomes

By the end you should be able to:

* Read the checkpoint config and place every dimension — `d_model`, KDA head count/dim, MLA latent width, indexer pool width, MoE expert count/top-k, mHC stream count — without consulting a diagram.
* Derive the MoE router's actual score (sigmoid + correction bias for **selection**, raw sigmoid scores renormalized for **weighting**) and explain why that split matters for correctness.
* Write the KDA delta-rule update from the five named operations, expand it into the closed-form `(I − βkkᵀ)D·S + βkvᵀ`, and reproduce a worked example by hand.
* Explain why a chunked/parallel KDA implementation must match both **output and final state** against the recurrent reference — not just outputs on one prompt.
* Derive MLA's absorption algebra (`q̃ = (Uᴷ)ᵀq`) and state exactly what it does and does not change about the softmax.
* Compute DSA's indexer slot budget including the incomplete-tail case, and name the context lengths (3, 4, 5, 7, 8, 9) where causality bugs cluster.
* State why mHC's residual mixing does not reduce serial depth, and why "four streams" is a connectivity change, not four transformers.
* Build a full per-GPU memory equation for a tensor-parallel hybrid model that does **not** naively divide every term by the TP degree.
* Specify a numerical correctness contract *before* measuring a speedup, and name the invariant that catches a rollback bug a KV-length check would miss.

---

## What this course is not

* Not a general MoE/MLA serving course — [AI Inference Engineer 2026 — Part 3](../AI%20Inference%20Engineer%202026/Part%203%20-%20MoE%20at%20Blackwell/README.md) covers that ground generically.
* Not a re-derivation of NVFP4 — [Hardware-Aware LLM Quantization](../Hardware-Aware%20LLM%20Quantization/README.md) already did that; this course cites it.
* Not a claim that every number here was independently re-measured. Figures attributed to the published checkpoint configuration or to the `sparkinfer-frontier` repository are cited as such, not presented as measurements taken for this course — the distinction is load-bearing throughout, especially in Module 09.

---

## Currency / Refresh Discipline

* **Timeless:** the delta-rule algebra, the MLA absorption identity, the roofline lower-bound model, the correctness-contract discipline.
* **Moves with the checkpoint:** every dimension in Module 01's config table, the 304.4B/18B parameter split, the indexer budget. If a future GLM-5.x-Flash revision ships, re-verify Module 01 against its config before trusting anything downstream of it — every later module's arithmetic depends on those numbers.
* **Moves with the runtime:** which serving stack (SGLang, vLLM, TensorRT-LLM) has a fused KDA kernel, chunked-prefill support, and disaggregation for this specific hybrid state shape. Module 10 is the refresh surface.
* Every module closes with a **`## Current as of`** note separating settled math from checkpoint- and tooling-specific facts.

---

## Exit Criteria

You are done when you can take the checkpoint config alone and, without looking anything up:

* Reconstruct the attention schedule (`[KDA,KDA,KDA,DSA/MLA]×11 + KDA`) and the dense/MoE FFN split from the layer count.
* Derive the per-expert parameter count and the routed-expert total to within rounding.
* Write out the KDA update in five lines and expand it to the closed form without error.
* State, for any one of the five mechanisms, which cost it reduces and which of the other four it does *not* affect.
* Build the per-GPU memory equation for a stated context length and tensor-parallel degree, term by term, correctly marking each as sharded, replicated, request-dependent, or transient.
* Name a numerical invariant that a benchmark passing on "the output looks reasonable" would miss.

If you can recite the five mechanism names but cannot derive any of their update equations, you have vocabulary. The point of this course is the derivation.

---

*Related: [Hardware-Aware LLM Quantization](../Hardware-Aware%20LLM%20Quantization/README.md) · [AI Inference Engineer 2026](../AI%20Inference%20Engineer%202026/README.md) · [MLSys Deep Dives](../MLSys%20Deep%20Dives/README.md) · [Phase 5 — ML Systems Engineering Guide](../Guide.md)*
