# Module 01 — The Exact Model

**Collection:** [GLM-5.3-Flash Architecture Mastery](README.md) | **Previous:** [← Course index](README.md) | **Next:** [Module 02 →](Lecture-02.md)

---

Before studying any mechanism, fix the checkpoint you are actually studying. GLM-5.3-Flash is not GLM-5.3 run with fewer layers, nor a quantized shrink of a larger sibling — Z.ai describes it as a **redesigned architecture** combining linear and sparse attention with manifold-constrained hyper-connections, trained as its own model. Everything downstream in this course depends on the numbers in this module being right, so treat this as ground truth to memorize, not background to skim.

---

## Learning objectives

By the end of this module you should be able to:

1. Recite the checkpoint's configuration from memory: hidden width, layer count, attention schedule, MoE shape, residual stream count.
2. Reconstruct the attention schedule and the dense/MoE FFN split from the layer count alone.
3. Trace one token through the full execution path and name where each of the five mechanisms sits in it.
4. Explain why this model is a genuine architectural redesign rather than a variant of an existing one.
5. State which published numbers are checkpoint facts versus deployment-dependent claims.

---

## 1. The published configuration

```text
   Model class            Glm5NextForConditionalGeneration
   Parameters             ≈320B total, ≈18B active per token
   Language hidden width  4,096
   Decoder depth          45 layers
   Attention mixture      34 KDA layers + 11 sparse MLA/DSA layers
   Feed-forward mixture   first 3 layers dense; remaining 42 layers MoE
   Routed experts         288 per MoE layer, top-8 selected
   Shared experts         1 per MoE layer (always active)
   Residual streams       4, via mHC
   Configured max context 1,048,576 tokens  (2^20)
```

Two things to flag immediately, because they recur throughout the course:

* **"18B active" is a per-token compute estimate, not a memory figure.** The MoE router selects which experts *run* for a token; it does not shrink how many expert weights must be *resident* to serve any token in a batch. [Module 02](Lecture-02.md) makes this precise.
* **"1,048,576 tokens" is a configured ceiling, not a deployment guarantee.** Whether a specific 8-GPU deployment can actually allocate and correctly serve that length is a memory-and-correctness question this course answers in [Modules 09](Lecture-09.md) and [11](Lecture-11.md) — not something the config file settles by itself.

---

## 2. Reconstruct the attention schedule yourself

The schedule is stated as a repeating block:

```text
   [ KDA, KDA, KDA, DSA/MLA ]  ×11   =  44 layers
                                 +1 KDA   =  45 layers  (matches decoder depth)

   KDA layers total        =  3×11 + 1  =  34   ✓ matches "34 KDA layers"
   DSA/MLA layers total    =  1×11      =  11   ✓ matches "11 sparse MLA/DSA layers"
```

This is worth doing by hand rather than trusting the prose: it is the first of many places in this architecture where a published summary number ("34 KDA layers") is *derivable* from a simpler repeating structure, and being able to derive it is what lets you catch an error in someone else's re-implementation later.

The feed-forward split works the same way:

```text
   45 layers total  −  3 dense  =  42 MoE layers    ✓ matches "remaining 42 MoE"
```

Every decoder layer has both an attention sublayer (KDA or DSA/MLA) **and** a feed-forward sublayer (dense or MoE) — these are two independent axes, not one combined schedule. A layer's position determines its feed-forward type by simple threshold (`layer_idx < 3` → dense) and its attention type by the repeating pattern above. mHC wraps both sublayers identically, regardless of which variant of either is in use.

---

## 3. One token through the model

```text
   Text tokens ──▶ embeddings ─────────────────────────────────┐
                                                                │
   Images ──▶ vision encoder ──▶ language-width features ──────┤
                                                                ▼
                                                    Four residual streams
                                                                │
                            ┌───────────────────────────────────┐
                            │  mHC collapse + normalization      │
                            │  KDA  or  sparse MLA (+ DSA select) │
                            │  mHC residual mixing                │
                            │                                     │
                            │  mHC collapse + normalization      │
                            │  Dense FFN  or  MoE                 │
                            │  mHC residual mixing                │
                            └───────────────────────────────────┘
                                          × 45 layers
                                                                │
                                                    Final stream contraction
                                                                │
                                                    Normalization + LM head
                                                                │
                                                      Next-token logits
```

This is a **conceptual map of the native model flow**, not a list of separately launched GPU kernels — a real implementation fuses, reorders, and batches heavily across this diagram, and [Module 10](Lecture-10.md) is where you learn to reason about that gap. But every box in this diagram corresponds to something you must be able to name and locate in checkpoint code:

| Box | Mechanism | Module |
|---|---|---|
| vision encoder → language-width features | multimodal projection | [08](Lecture-08.md) |
| mHC collapse / mixing | manifold-constrained hyper-connections | [07](Lecture-07.md) |
| KDA | recurrent linear attention | [03](Lecture-03.md), [04](Lecture-04.md) |
| sparse MLA (+ DSA select) | compressed latent attention + selective retrieval | [05](Lecture-05.md), [06](Lecture-06.md) |
| Dense FFN / MoE | sparse parameter activation | [02](Lecture-02.md) |

---

## 4. Why "redesigned," not "distilled"

It is tempting to read a hybrid model as "take an existing dense or MoE transformer and swap in a cheaper attention." Two facts argue against that reading, and both matter for how you approach the rest of this course:

**First, the mechanisms are co-designed, not bolted on.** KDA's channel-wise decay gate and MLA's latent width were chosen together with the DSA pooling width (4 tokens) and the mHC stream count (4) — these are not independent hyperparameters you could vary freely without touching the others' effective capacity. Studying one mechanism in isolation (which this course does, module by module, for pedagogical reasons) is a simplification you should consciously undo once you reach [Module 09](Lecture-09.md)'s full memory model, where all five interact in one budget.

**Second, the training objective produced genuinely different weights, not a pruned subset of a larger model's weights.** A KDA layer's decay gates and a DSA indexer's pooling projections are parameters that only make sense for *this* architecture — there is no "GLM-5.3 minus these layers" checkpoint hiding underneath. Practically, this means techniques you might reach for on a pruned or distilled model (e.g., "restore the removed layer's weights to recover behavior") do not apply here. If GLM-5.3-Flash's behavior diverges from GLM-5.3's, the fix is not "put back what was removed" — there is nothing to put back.

---

## 5. What this module does *not* establish

Be precise about what a config file and an architecture description give you, versus what requires derivation or measurement:

```text
   FROM THE CONFIG (this module)          REQUIRES DERIVATION (later modules)
   ───────────────────────────────        ────────────────────────────────────
   layer count, hidden width              routed-expert parameter total   (02)
   attention/FFN schedule                 per-token bytes moved by KDA    (04)
   expert count, top-k                    MLA cache size at a given ctx   (05)
   residual stream count                  indexer slot budget            (06)
   configured max context                 whether 1M context FITS on 8 GPUs (09)
```

The right-hand column is the rest of this course. Nothing there is guessable from the left-hand column alone — it all has to be derived, and derived correctly, which is why [Modules 02–07](Lecture-02.md) exist before the memory model in [Module 09](Lecture-09.md) is allowed to use any of their results.

---

## Checkpoint

You should now be able to:

1. Recite all eight rows of the configuration table from memory.
2. Derive `34 KDA + 11 DSA/MLA = 45` and `42 MoE + 3 dense = 45` from the repeating-block structure alone.
3. Point to each box in the execution-trace diagram and name which module of this course explains it.
4. Explain, in one sentence, why "distilled from GLM-5.3" is the wrong mental model.
5. Sort five example claims about this model into "checkpoint fact" versus "requires derivation" versus "deployment-dependent."

---

## Ship it

Build a **checkpoint-to-code tensor map**: for each row of the configuration table, the exact config field name, its value, and the module/class in the reference implementation where it is consumed. This is Stage 1 of the [capstone ladder](Lecture-12.md), and every later module in this course assumes you have it — you will be reaching for it constantly.

---

## Current as of

* **Checkpoint-specific:** every number in §1 is a property of the released GLM-5.3-Flash configuration, current as of this writing. A future revision can change any of them — re-run §2's derivation against the new config before trusting anything downstream.
* **Timeless:** the distinction between "co-designed hybrid architecture" and "pruned/distilled variant," and the discipline of separating config facts from derived and deployment-dependent claims.

---

**Next:** [Module 02 — MoE: Capacity, Work, and Traffic →](Lecture-02.md)
