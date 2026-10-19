# Module 02 — MoE: Capacity, Work, and Traffic

**Collection:** [GLM-5.3-Flash Architecture Mastery](README.md) | **Previous:** [← Module 01](Lecture-01.md) | **Next:** [Module 03 →](Lecture-03.md)

---

"18B active parameters" is the number everyone repeats and the number that misleads the most engineers into underestimating what this model costs to serve. This module separates three quantities that MoE collapses into one popular sentence, derives the router's actual arithmetic (which is not "softmax, then top-8"), and computes exactly how the checkpoint's size survives sparse activation.

---

## Learning objectives

By the end of this module you should be able to:

1. Distinguish MoE **capacity**, **arithmetic**, and **memory traffic**, and give an example where two of the three move independently.
2. Write the router's actual computation — sigmoid scoring, additive correction bias for *selection*, renormalized raw scores for *weighting* — and explain why those are two different uses of two different quantities.
3. State the clamped-SwiGLU expert function and identify its asymmetry.
4. Derive the per-expert and total routed-expert parameter counts from the checkpoint's dimensions.
5. Explain why a correctly-selecting, incorrectly-weighting kernel produces plausible but wrong output.

---

## 1. Three quantities MoE separates

For an input `x` to a feed-forward sublayer, the output is:

```text
   y  =  f_shared(x)  +  Σ_{e in E(x)}  w_e(x) · f_e(x)
```

where `E(x)` is the set of 8 routed experts selected for this token, and the shared expert `f_shared` runs unconditionally, every token. This single equation hides three quantities that behave completely differently:

```text
   CAPACITY    :  all 288 routed + 1 shared expert weights that EXIST in the checkpoint
   ARITHMETIC  :  the 8 routed + 1 shared experts SELECTED for this one token
   TRAFFIC     :  the weights actually FETCHED, for the whole batch, given the serving strategy
```

```text
   two tokens select the SAME expert  ──▶  ONE weight-fetch can serve both   (reuse)
   two tokens select DIFFERENT experts ──▶  TWO weight-fetches, no reuse available

   Same arithmetic (8 experts each). Different traffic (1× vs 2× the fetch).
```

This is why "18B active" answers an arithmetic question (FLOPs per token) and answers almost nothing about memory traffic, which is what governs decode throughput on a bandwidth-bound GPU (see [Hardware-Aware LLM Quantization — Module 01](../Hardware-Aware%20LLM%20Quantization/Lecture-01.md) for the general argument). At batch size 1, the current token reads its selected experts, subject to cache residency and kernel layout. At larger batch sizes, the number of distinct selected experts determines potential weight reuse. Parameter counts alone do not describe those cache and batching effects.

---

## 2. The router is not "softmax, then top-8"

The reference computation, precisely:

```text
   r    =  W_r · x                            router logits, computed in FP32
   s_e  =  sigmoid(r_e)                       per-expert score

   E(x) =  TopK_8( s_e + b_e )                SELECTION uses score + correction bias

   w_e  =  2.5 · s_e / Σ_{j in E(x)} s_j       WEIGHTING uses the raw sigmoid score,
                                                renormalized over the selected set only
                                                (2.5 is the configured routing scale)
```

Read that carefully, because it contains a distinction that is easy to implement wrong:

```text
   b_e (correction bias)  ──▶  used ONLY to decide WHICH experts get selected
                                NEVER appears in the final mixture weight

   s_e (sigmoid score)    ──▶  used to decide selection (added to b_e)
                                AND reused, on its own, to decide the WEIGHT
```

```text
   ┌──────────────────────────────────────────────────────────────────┐
   │  A kernel that selects the correct 8 experts but then reuses      │
   │  (s_e + b_e) instead of s_e alone when computing w_e will select   │
   │  the RIGHT experts and MIX them with the WRONG weights.            │
   │                                                                    │
   │  The output will look fluent. It will not be this model.          │
   └──────────────────────────────────────────────────────────────────┘
```

This is exactly the kind of bug [Module 11](Lecture-11.md)'s correctness matrix exists to catch, and it is worth internalizing now: **plausible text is not evidence of a correct kernel.** A router implementation should be tested by comparing selected expert IDs *and* mixture weights independently against a reference — never by reading the generated text and deciding it "looks right."

---

## 3. The expert function: clamped SwiGLU

Each expert — routed or shared — computes:

```text
   a  =  min( W_g · x , 10 )                  gate projection, clamped ABOVE only
   u  =  clip( W_u · x , −10, 10 )             up projection, clamped BOTH sides

   f_e(x)  =  W_d · [ SiLU(a) ⊙ u ]
```

The asymmetry is the detail a generic SwiGLU implementation misses:

```text
   gate (a)  :  upper-clamped only     min(a, 10)
   up   (u)  :  clamped both sides     clip(u, −10, 10)
```

A drop-in "standard SwiGLU" replacement — which typically clamps neither, or clamps both projections symmetrically — will match this expert's behavior almost everywhere activations are small, and silently diverge exactly on the large-activation tail: the inputs [Hardware-Aware LLM Quantization — Module 06](../Hardware-Aware%20LLM%20Quantization/Lecture-06.md) calls **massive activations**. Those are rare, load-bearing, and precisely the tokens where a clamping mismatch would first show up — which means a short evaluation run can easily miss this bug entirely and a long one will not.

---

## 4. Deriving the checkpoint's actual size

The router picks 8 of 288 experts per token, but **all 288 must be available** for the model to serve arbitrary tokens. A fully GPU-resident deployment stores them in device memory; CPU/NVMe offload changes residency and adds transfer constraints. Compute the cost of that capacity directly from the checkpoint dimensions.

**Per-expert parameter count.** Expert intermediate width is 2,048; hidden width is 4,096. Three matrices — gate, up, down — each `4096 × 2048`:

```text
   P_expert  =  3 × 4096 × 2048  =  25,165,824  parameters
```

**Routed-expert total.** Across 42 MoE layers × 288 routed experts per layer:

```text
   P_routed  =  42 × 288 × 25,165,824
             =  12,096 experts  ×  25,165,824
             ≈  304.4 × 10⁹  parameters
```

The routed matrices account for about 304.4B parameters. Each token executes
8 routed plus 1 shared expert in each of 42 MoE layers: **378 expert applications**,
representing about 9.51B expert parameters. The advertised 18B active count covers
more than these expert matrices; it also includes other parts of the forward pass.
Sparse selection reduces arithmetic without deleting unselected checkpoint weights.


Add the shared experts (1 per MoE layer, always active, same shape as a routed expert):

```text
   P_shared  =  42 × 25,165,824  ≈  1.06 × 10⁹  parameters

   P_routed + P_shared  ≈  305.5 × 10⁹  parameters
```

That leaves roughly `320B − 305.5B ≈ 14.5B` for everything else in the model: all 45 layers' attention mechanisms (KDA and MLA/DSA), the 3 dense FFN layers, embeddings, and the LM head. **Sanity-check this split for yourself against the actual checkpoint** rather than trusting the arithmetic above as exhaustive — it is a lower bound built from the two largest, most cleanly-specified components, not a full parameter audit.

### Why this matters for serving, concretely

```text
   Fully GPU-resident weight budget    ≈  ALL 288 routed + 1 shared expert/layer
                                             (roughly the full 320B-parameter footprint)

   Per-token active-parameter estimate ≈  ~18B across the whole model

   These are different budgets. The first is the weight term in a fully resident design;
   activations, caches, workspace and communication buffers also need memory. [Module 09](Lecture-09.md) builds the
   full per-GPU budget from this starting point.
```

---

## 5. Capacity, arithmetic, and traffic — worked contrast

Put the three quantities from §1 next to each other for one concrete scenario: a batch of 32 decode requests, one MoE layer, no routing collisions assumed.

| Quantity | Value | What it governs |
|---|---:|---|
| **Capacity** | 289 × 25,165,824 ≈ 7.27 B parameters, including the shared expert | weight storage in a fully resident layer |
| **Arithmetic** | 32 × 9 × 25,165,824 ≈ 7.25 B weight uses; roughly 14.50 GFLOP for the projections counting a multiply-add as two operations | compute time, before activation/router costs |
| **Traffic (disjoint routed selections)** | 32 × 8 routed experts + 1 shared = 257 distinct expert weight sets | potential batched weight traffic, assuming one fetch per distinct expert |
| **Traffic (identical routed selections)** | 8 routed + 1 shared = 9 distinct expert weight sets | 257/9 ≈ 28.56× fewer distinct sets than the disjoint case |

**Arithmetic is identical in the last two rows. The potential distinct-weight traffic differs by about 28.56×.** The shared expert is always shared; counting it as a different expert for every request was an error. Actual fetch traffic can exceed this distinct-set bound when the kernel rereads weights or the cache cannot retain them. This is why MoE serving systems care intensely about routing locality, expert-parallel placement, and batch composition — none of which "18B active" or a naive FLOP count will surface. It is also the reason expert-parallel serving systems track *expert popularity skew* as a first-class metric: a batch that happens to concentrate on a few popular experts behaves like the best-case row; a batch with uniformly scattered routing behaves like the worst case, on the exact same hardware, same model, same token count.

---

## Checkpoint

You should now be able to:

1. Give an example where capacity, arithmetic, and traffic each take a different value for the same request.
2. Write the router's selection rule and weighting rule as two separate expressions, and say which uses the correction bias.
3. State the clamped-SwiGLU asymmetry and predict which activations expose a mismatched implementation.
4. Derive `P_expert = 25,165,824` and `P_routed ≈ 304.4B` from the checkpoint's stated dimensions, from memory.
5. Explain why routing collision rate is a serving-relevant metric that parameter count alone cannot predict.

---

## Ship it

This is half of **Stage 6 of the [capstone ladder](Lecture-12.md)** (paired with [Module 07](Lecture-07.md)'s residual-flow tests). Produce a **router/expert audit** against the reference implementation: (1) a test that compares selected expert IDs and mixture weights independently, using inputs constructed so that `s_e + b_e` and `s_e` alone would select or weight differently if confused; (2) a test that exercises the clamp boundaries on both the gate and up projections; (3) the parameter derivation from §4, reproduced against the actual checkpoint's tensor shapes rather than the config's stated dimensions alone.

---

## Current as of

**Reviewed October 18, 2026.** [Pinned config, upstream code and CPU evidence](CPU-Hybrid-State-Lab.md#sources-and-execution-boundary) identify what was checked. Hardware capacity and throughput remain deployment-specific.

* **Timeless:** the capacity/arithmetic/traffic distinction, the general DeepSeek-style sparse-MoE selection/weighting pattern this router follows.
* **Checkpoint-specific:** the routing scale (2.5), clamp bounds (±10, and the gate's asymmetric upper-only clamp), expert intermediate width (2,048), expert count (288 routed + 1 shared), and top-k (8) are properties of this checkpoint's configuration — verify against the actual config before reusing these constants for a different revision.

---

**Next:** [Module 03 — KDA I: The Delta-Rule Recurrence →](Lecture-03.md)
