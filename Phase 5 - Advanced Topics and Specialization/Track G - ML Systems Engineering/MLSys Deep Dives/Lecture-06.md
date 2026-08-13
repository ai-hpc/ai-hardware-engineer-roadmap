# Lecture 06 - Making Decode Fast: Speculative Decoding, DFlash, and Flash Kernels

**Collection:** [MLSys Deep Dives](README.md) | **Previous:** [← Lecture 05](Lecture-05.md) | **Next:** [Lecture 07](Lecture-07.md)

---

We have made the kernels fast (Lec 2–3) and the model lean (Lec 4–5). This lecture attacks the last and most stubborn cost: **autoregressive decode is sequential and memory-bound**, and no kernel makes a fundamentally serial loop parallel. The breakthrough of 2023–2026 was algorithmic — **speculative decoding** — a family of tricks that emit *multiple* tokens per pass over the weights, **with provably identical output** to normal decoding.

This is the densest-payoff layer in the stack: speculative decoding routinely delivers 2–6× with *zero* quality loss, and the 2026 frontier (DFlash, the MiMo + TileRT 1000-tok/s milestone) pushes further. We trace the algorithm lineage, the attention kernels that make the verify step cheap (FlashAttention-3, FlashInfer), and the research group at the center of most of it — **Together AI**.

---

## Learning objectives

By the end of this lecture, you should be able to:

1. Explain why decode is **memory-bound**, and why verifying K candidate tokens costs ~the same memory traffic as generating one.
2. Trace the speculative-decoding lineage: **draft model → Medusa → Hydra → Sequoia → Lookahead → EAGLE-1/2/3**, and what each added.
3. Describe **EAGLE-3** (direct-token prediction, "training-time test") and **DFlash** (block-diffusion drafting) and why they beat predecessors.
4. Use **acceptance length** as the governing metric and relate it to speedup.
5. Place **FlashAttention-3, FlashInfer, Flash-Decoding** as the kernels that make the verify/attention step fast.
6. Read the **MiMo + TileRT** 1000-tok/s result as a *stack* (quantization + DFlash + megakernel), and locate **Together AI**'s research across the whole layer.

---

## 1. Why decode is stuck — and the way out

Recall from Lecture 1: **prefill is compute-bound, decode is memory-bound.** Generating one token requires streaming the *entire* weight set (and traversing the whole KV cache) from HBM, to do a tiny amount of matmul. The Tensor Cores sit mostly idle; the bottleneck is bandwidth.

```text
   one decode step (generate 1 token):
   ┌────────────────────────────────────────────────────────┐
   │  stream ALL weights + KV from HBM  ───────────────────► │  ← dominates (memory bandwidth)
   │  matmuls (Tensor Cores ~idle)                           │  ← tiny (compute)
   │  emit 1 token                                           │
   └────────────────────────────────────────────────────────┘
```

Here is the key observation that unlocks everything: **verifying K candidate tokens costs essentially the same memory traffic as generating one** — it's still one pass over the weights. So if you can *guess* the next K tokens cheaply and then *verify* them all in a single pass, you amortize the dominant memory cost across multiple tokens:

```text
   speculative decoding:
   ① a cheap DRAFT proposes K tokens ahead
   ② the big TARGET verifies all K in ONE forward pass
   ③ accept the longest prefix the target agrees with;  re-draft from there
   ⇒ same HBM traffic per pass, up to ~K tokens out  →  more tokens per memory pass
   ⇒ OUTPUT IS PROVABLY IDENTICAL to normal target decoding (lossless)
```

That last line is what makes speculative decoding special: it is a **free lunch in quality** — same distribution, more speed. "Provably identical" is a strong claim, so here is the proof's engine — the **acceptance rule** (modified rejection sampling), which every method in this lecture inherits:

```text
   for each drafted token x, with draft probability q(x) and target probability p(x):
       accept x with probability  min(1, p(x) / q(x))
       on the first rejection:    resample from  norm( max(0, p − q) )   and stop there
   ⇒ the token that comes out — accepted or resampled — is distributed EXACTLY as p,
     for ANY draft q. (greedy is the easy case: accept while draft matches target argmax.)
```

Walk the intuition: where the draft *over*-proposes a token (`q > p`), acceptance is thinned by `p/q`; where it *under*-proposes (`q < p`), the token is always accepted *and* the leftover probability mass (`p − q`) is exactly what the rejection-resample distribution restores. The two effects cancel to `p` precisely. The consequence an engineer cares about: **a bad draft costs you speed (low acceptance), never correctness** — which is why the entire research race is about making the draft *cheaper and more accurate* (higher acceptance), so more of the K tokens survive verification.

---

## 2. The governing metric: acceptance length

Before the lineage, fix the metric, because every method is measured by it. The verify step proposes a sequence (or tree) of candidate tokens; the target accepts the **longest valid prefix**. The average number accepted per step is the **acceptance length** τ.

```text
   one verify cycle:  draft K tokens, verify once, keep τ on average

                         τ                  c  =  draft cost per token
   speedup   ≈      ─────────                     ─────────────────────
                     1 + K·c                      target cost per step

   τ = 4, K = 5, c = 0.05 (EAGLE-style head)  →  4 / 1.25  =  3.2×
   τ = 4, K = 5, c = 0.5  (a chunky draft LM) →  4 / 3.5   =  1.14×   ← same τ, draft ate the win
```

The two rows are the whole design space: **the same acceptance length is worth 3× or worth nothing depending on what the draft costs.** So a method wins by **raising τ** (better drafts → more accepted) or **lowering c** (cheaper to produce the guesses) — and the lineage in §3 is exactly the history of pushing both at once. EAGLE-3 reports τ improvements that translate to ~3–6.5×; DFlash pushes τ higher still with block drafting. When you benchmark spec decode, **τ is the number you report** — it's the mechanism-level truth behind the wall-clock speedup.

---

## 3. The lineage

Every method is a different answer to "how do I produce good candidate tokens cheaply?"

```text
   DRAFT MODEL     a small LM drafts K tokens; the big model verifies K in one pass
        │          (simple, but you must train/host a matched small model)
   MEDUSA          add lightweight DECODING HEADS to the frozen base; tree-attention
        │          over candidates. no separate model.            [Tri Dao co-author → Together]
   HYDRA           make the heads SEQUENTIALLY DEPENDENT (each conditions on prior
        │          drafted tokens) → ~1.3× over Medusa.
   SEQUOIA         find the OPTIMAL token-tree topology by dynamic programming;
        │          temperature-robust; HARDWARE-AWARE tree sizing.  [Together co-author]
   LOOKAHEAD       draft-FREE: break the sequential dependency with Jacobi iteration,
        │          generate & verify n-grams in parallel. no aux model.   [Hao AI Lab]
   EAGLE-1         autoregress at the FEATURE level (predict the target's 2nd-to-top
        │          hidden feature, reuse its LM head). ~2.7–3.5×.   [Li et al.]
   EAGLE-2         add a DYNAMIC, context-aware draft TREE (keep high-confidence branches).
        │
   EAGLE-3         drop feature-prediction → DIRECT token prediction; fuse multi-layer
        │          features via "TRAINING-TIME TEST". ~3–6.5×, +20–40% over EAGLE-2,
        │          and the speedup now SCALES WITH TRAINING DATA. the de-facto baseline.
   DFLASH          a small BLOCK-DIFFUSION draft predicts a WHOLE BLOCK in one pass.
                   2–3× over EAGLE-3 on synchronous requests; >6× overall; lossless.
```

A few points a senior engineer holds onto:

* **Self-drafting beat separate drafts.** Medusa/EAGLE attach the draft to the *base model* (heads or a light feature head), removing the need to train and host a separate, well-matched small model — a big operational simplification.
* **Trees beat chains.** Proposing a *tree* of candidates and verifying it with tree-attention (Medusa → EAGLE-2 → Sequoia) raises τ, because you hedge across multiple plausible continuations in one verify pass. Sequoia made the tree *optimal and hardware-aware*.
* **EAGLE-3's "training-time test"** is the subtle, important idea: train the draft head under the *same* multi-step autoregressive condition it faces at inference, so train and test distributions match. This is what let EAGLE-3 keep improving with more data where EAGLE-1/2 plateaued.

---

## 4. DFlash — drafting a whole block at once

**DFlash** (the "dfalsh"/"dflash" you may have seen; arXiv 2602.06036, Z-Lab, 2026) is the current frontier and a genuinely different draft mechanism. Instead of an autoregressive head that drafts token-by-token, DFlash uses a small **block-diffusion** draft model that predicts an **entire block of tokens in a single forward pass** — non-causal attention over the verifier's hidden states plus mask embeddings, conditioned on the target's features.

```text
   EAGLE-3 draft:  token → token → token → ...   (sequential head, K small passes)
   DFlash draft:   [ ▢ ▢ ▢ ▢ ▢ ▢ ] → fill the WHOLE masked block in ONE pass (diffusion)
                   → bigger blocks drafted cheaper → higher τ, fewer draft passes
```

Reported: **2–3× larger speedups than EAGLE-3 on synchronous requests**, **>6× overall**, still **lossless**, and it's integrated into **vLLM's "speculators"** framework (with a published checkpoint). It is also the speculative-decoding component inside the MiMo throughput result in §6. The takeaway: the draft mechanism is still actively improving — block-diffusion drafting is the 2026 state of the art, and "what's the best draft?" is a live research front, not a settled question.

---

## 5. The kernel layer: making the verify step cheap

Speculative decoding is the *algorithm*; the **attention kernel** still has to run the verify pass fast. Three Flash-family kernels (all closely tied to Tri Dao / Together) make that step efficient — a different axis from spec decode, stacked on top of it:

* **FlashAttention-3** (Jul 2024) — the Hopper-specific attention rewrite: **warp specialization** to overlap compute with async Tensor-Core + TMA data movement, interleaved matmul/softmax pipelining, and **FP8** support. Results: **1.5–2.0× over FA-2**, FP16 up to **~740 TFLOP/s (~75% H100 utilization)**, FP8 up to **~1.2 PFLOP/s** with 2.6× lower numerical error than baseline FP8. This is the kernel that makes attention itself near-peak on Hopper.
* **FlashInfer** (MLSys 2025 **Best Paper**) — not a single kernel but a **serving-oriented attention engine/library**: handles **KV-cache storage heterogeneity** (block-sparse + composable formats), **JIT-compiled customizable attention templates**, and **load-balanced scheduling** compatible with CUDA Graphs. It's adopted by **vLLM, SGLang, and MLC-Engine**, is now NVIDIA-backed (the upstreaming from Lecture 3), and reports **29–69% inter-token-latency reductions** vs compiler backends. When you serve attention in 2026, FlashInfer is very likely underneath.
* **Flash-Decoding** (Oct 2023) — the decode-phase trick for **long context, small batch**, where query length is 1 and plain FlashAttention uses <1% of the GPU. It **parallelizes the attention reduction across the KV (sequence) dimension** — splitting keys/values across SMs, then a small final combine — giving **near-constant latency to 64K+ tokens** and up to **~8× end-to-end** (and up to ~50× vs FA in the long-seq decode regime).

The mental model: **spec decode reduces *how many* memory passes you need; Flash kernels reduce *how expensive each pass* is.** They multiply. A serving stack runs FlashInfer/FA-3 kernels *and* EAGLE-3/DFlash drafting *together*.

---

## 6. Case study: MiMo + TileRT past 1000 tokens/s

The 2026 result that stitches this whole course together: **Xiaomi's MiMo team + the TileRT runtime pushed a 1-trillion-parameter MoE model past ~1000 tokens/s** (peaks ~1200) on **a single standard 8-GPU commodity node** — no exotic silicon. It is worth dissecting because it is *every layer of this course stacked*:

```text
   ① ARCHITECTURE   MiMo-V2.5-Pro: a 1T-param MoE (Lec 5), MTP-friendly, sparse-active
   ② QUANTIZATION   MXFP4 on the MoE expert layers (rest higher precision) → bytes ↓ (Lec 5/7)
   ③ SPEC DECODE    DFlash block-diffusion drafting (§4) → tokens/memory-pass ↑
                    (reported acceptance lengths ~6.30 coding / 5.56 math / 4.29 agent)
   ④ RUNTIME        TileRT persistent MEGAKERNEL (Lec 3): warp-specialized, GPU-resident,
                    overlaps compute/IO/comm, µs-scale overhead → kills the gaps
   ─────────────────────────────────────────────────────────────────────────────────
   ⇒ ~1000–1200 tok/s on a 1T model, one 8-GPU node.  priced ~3× the standard rate for ~10× speed.
```

No single trick did it. Architecture (sparse MoE) + precision (MXFP4) + algorithm (DFlash) + runtime (TileRT megakernel) **compounded**. That is the thesis of this course made concrete: the layers are one co-designed system, and the wins multiply.

> **Currency / sourcing flag.** This result was announced ~**2026-06-08** by Xiaomi/TileRT and covered by tech press; the figures (~1200 tok/s, acceptance lengths, MXFP4-on-experts, "8-GPU commodity node" — exact GPU model unspecified) are **vendor-reported and not yet independently reproduced**. Treat as a leading-edge data point, not a settled benchmark. Re-verify before quoting in a design review.

---

## 7. Together AI — the research thread through this whole layer

If one organization sits at the center of this lecture, it is **Together AI**, largely through **Tri Dao** (its Chief Scientist) and collaborators. Their portfolio *is* the modern decode-acceleration stack:

| Layer | Together-linked work |
|---|---|
| Attention kernels | **FlashAttention** (1/2/3), **Flash-Decoding** — IO-aware exact attention |
| Architecture | **Mamba** (via Dao; Lec 4) — the SSM line |
| Speculative decode | **Medusa** (Dao co-author), **Sequoia** (Together co-author) |
| Multi-agent inference | **Mixture-of-Agents (MoA)** — weak proposers + an aggregator; reported **65.1% AlpacaEval LC, beating GPT-4o's 57.5%** with open models |
| Serving | **Together Inference Engine** — production stack on Blackwell |

The positioning to understand: Together is known for **the full vertical — from the CUDA attention kernel, through efficient architectures, through speculative decoding, to the serving engine and its economics.** Their Inference-Engine marketing claims (e.g. **+31% TPS vs TensorRT-LLM**, **~2× better TTFT at saturation**, "**up to 10× lower cost per token**" on Blackwell, **76% cheaper than Claude Opus** on a coding-agent benchmark) are **vendor benchmarks on specific workloads** — directionally informative, not neutral third-party numbers, and you should treat them as such. The research, though (FlashAttention, Mamba, Medusa, Sequoia), is foundational and independently verifiable.

---

## 8. Hands-on / Measure it

Enable speculative decoding in a real serving stack and measure the three numbers that matter.

```python
from vllm import LLM, SamplingParams

# EAGLE-3 speculative decoding in vLLM (API surface evolves across versions — check your vLLM)
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    speculative_config={
        "method": "eagle3",
        "model": "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
        "num_speculative_tokens": 5,
    },
)
# simplest baseline to compare against: draft-free n-gram speculation
#   speculative_config={"method": "ngram", "num_speculative_tokens": 4}
```

Measure, against a no-spec baseline:

```text
   1. CORRECTNESS:   outputs identical to no-spec (greedy) — spec decode is LOSSLESS; verify it
   2. ACCEPTANCE τ:  mean tokens accepted per verify step (vLLM reports this)
   3. SPEEDUP:       tokens/s with spec ÷ tokens/s without  →  recompute $/Mtok (Lecture 1)
```

A result like "τ = 3.2, 2.4× tokens/s, identical outputs, `$/Mtok` $0.56 → $0.23" is the whole lecture in one line: **more tokens per memory pass, same quality, lower cost.** If τ is low, your draft is poorly matched to the workload — try a different draft method (EAGLE-3 vs ngram vs DFlash) or a domain-matched draft.

---

## 9. Mini-lab

1. **Baseline:** serve a model with no speculation. Record tokens/s, TTFT, TPOT, `$/Mtok`.
2. **Spec decode:** enable **EAGLE-3** (and, separately, **ngram**) speculation. Record τ, tokens/s, and confirm outputs are identical to greedy baseline. Compute the new `$/Mtok`.
3. **Workload sensitivity:** measure τ on two workloads (e.g. code vs open-ended chat). Explain why τ differs (predictable text → higher acceptance).
4. **(Stretch) stack it:** if your stack supports it, confirm FlashInfer/FA-3 is the attention backend, and reason about how spec decode (fewer passes) and the Flash kernel (cheaper passes) compound.

Deliverable: a `{baseline, ngram, EAGLE-3}` × `{τ, tokens/s, TPOT, $/Mtok, outputs-identical?}` table across two workloads, plus a paragraph on why τ moved and how much `$/Mtok` the *lossless* speedup bought. "Lossless cost reduction" is the most defensible win an MLSys engineer can put in a review — this lab produces one.

---

## Key takeaways

- Decode is **memory-bound**: each token streams all weights/KV from HBM. **Verifying K candidates costs ~one pass**, so speculative decoding emits more tokens per memory pass — **losslessly** (provably identical output).
- The governing metric is **acceptance length τ**: methods win by raising τ (better drafts) or lowering draft cost. Report τ, not just wall-clock.
- Lineage: **draft model → Medusa → Hydra → Sequoia → Lookahead → EAGLE-1/2/3**. Self-drafting beat separate drafts; trees beat chains; **EAGLE-3** (direct-token + "training-time test", ~3–6.5×) is the de-facto baseline.
- **DFlash** drafts a **whole block in one pass** (block-diffusion), 2–3× over EAGLE-3, in vLLM speculators — the 2026 frontier draft.
- **Flash kernels** make the verify/attention step cheap: **FlashAttention-3** (~75% H100 util, FP8), **FlashInfer** (serving attention engine, MLSys'25 best paper, in vLLM/SGLang), **Flash-Decoding** (long-context decode). Spec decode cuts *how many* passes; Flash kernels cut *cost per pass* — they multiply.
- The **MiMo + TileRT** ~1000-tok/s result = architecture (sparse MoE) + MXFP4 + **DFlash** + **TileRT megakernel** stacked — the course's "layers compound" thesis, made concrete (vendor-reported, June 2026).
- **Together AI** (via Tri Dao) anchors the whole layer — FlashAttention, Mamba, Medusa, Sequoia, MoA, the Together Inference Engine — research foundational, serving-engine numbers vendor-grade.

---

## References

- Li et al., "EAGLE-3," arXiv 2503.01840 · repo [https://github.com/SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE)
- Cai et al., "Medusa," arXiv 2401.10774: [https://arxiv.org/abs/2401.10774](https://arxiv.org/abs/2401.10774)
- Chen et al., "Sequoia," arXiv 2402.12374: [https://arxiv.org/abs/2402.12374](https://arxiv.org/abs/2402.12374)
- "DFlash: Block Diffusion for Flash Speculative Decoding," arXiv 2602.06036 · vLLM speculators [https://docs.vllm.ai/projects/speculators/](https://docs.vllm.ai/projects/speculators/)
- Shah et al., "FlashAttention-3," arXiv 2407.08608: [https://arxiv.org/abs/2407.08608](https://arxiv.org/abs/2407.08608)
- FlashInfer (MLSys 2025 Best Paper): [https://github.com/flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer)
- Flash-Decoding: [https://crfm.stanford.edu/2023/10/12/flashdecoding.html](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)
- Together AI, Mixture-of-Agents, arXiv 2406.04692: [https://arxiv.org/abs/2406.04692](https://arxiv.org/abs/2406.04692)
- Xiaomi MiMo + TileRT 1000-tok/s announcement (vendor, Jun 2026): [https://github.com/tile-ai/TileRT](https://github.com/tile-ai/TileRT)

---

## Current as of

2026-06. Pins: EAGLE-3 (2025, de-facto baseline), DFlash (arXiv 2602.06036, in vLLM speculators), FlashAttention-3 (Jul 2024, Hopper), FlashInfer (MLSys 2025 best paper). **MiMo + TileRT ~1000–1200 tok/s (announced ~2026-06-08) and Together Inference Engine numbers are vendor-reported, not independently reproduced** — flagged in-text. vLLM `speculative_config` API evolves across releases; verify against your version.

---

*Next: [Lecture 07 — The edge & physical-AI frontier](Lecture-07.md)*
