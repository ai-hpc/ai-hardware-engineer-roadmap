# Capstone — The KDA-First Mastery Ladder

**Collection:** [GLM-5.3-Flash Architecture Mastery](README.md) | **Previous:** [← Module 11](Lecture-11.md) | **Next:** [Course index](README.md)

---

Eleven modules produced derivations, worked examples, and per-mechanism test designs. This capstone turns them into a **build order** — eight staged deliverables, each one a prerequisite for the next, ending in one validated optimization on a real deployment. The ladder deliberately does not start with "the entire 320B model." It starts with **one KDA head.**

---

## 1. Why start with one KDA head, not the whole model

```text
   ┌────────────────────────────────────────────────────────────────────┐
   │   Start with ONE KDA head, not the entire 320B-parameter model.     │
   └────────────────────────────────────────────────────────────────────┘
```

Every other mechanism in this course (MoE routing, MLA's absorption algebra, DSA's pooling) is complex in its *bookkeeping* — many experts, many heads, many pooled positions — but each individual operation is comparatively simple once you see it clearly. KDA is the opposite: **one head's update is a five-line recurrence you can compute by hand** ([Module 03 §4](Lecture-03.md)), but getting its operator ordering wrong produces a state that silently diverges over many steps rather than crashing outright. It is simultaneously the easiest mechanism to build a first reference implementation for and the easiest one to get subtly, invisibly wrong. That combination is exactly why it is the right place to build your first habits of rigor before applying them to five mechanisms at once.

Once you can explain exactly what must be saved after a prefix, why the multiplication order matters, and how the recurrence becomes an efficient GPU computation, you have the foundation this entire course was building toward — and the remaining seven stages are that same rigor, applied outward to the rest of the architecture.

---

## 2. The eight stages

<div class="lecture-map" markdown>

| Stage | Deliverable | What demonstrates mastery | Built in |
|---|---|---|---|
| 1 | Architecture audit | A checkpoint-to-code tensor map — you can locate every major dimension, state, and projection | [Module 01](Lecture-01.md) |
| 2 | KDA reference | A small FP32 recurrent implementation — you can derive every update and explain its order | [Module 03](Lecture-03.md) |
| 3 | KDA execution | Recurrent/chunked comparison — outputs and final states agree across boundaries | [Module 04](Lecture-04.md) |
| 4 | MLA laboratory | Expanded and latent-space attention — you can prove equivalence and measure the traffic difference | [Module 05](Lecture-05.md) |
| 5 | KPool laboratory | Pooling, selection, tail, and mask tests — boundary and padding cases are correct | [Module 06](Lecture-06.md) |
| 6 | MoE + mHC audit | Router and residual-flow tests — you preserve selection, weighting, clamping, and stream semantics | [Modules 02](Lecture-02.md) + [07](Lecture-07.md) |
| 7 | Eight-GPU cost model | Predicted versus measured memory and latency — you explain discrepancies instead of hiding them | [Module 09](Lecture-09.md) |
| 8 | Optimization study | One validated end-to-end improvement — the speedup survives repeated runs and a correctness gate | Modules [10](Lecture-10.md) + [11](Lecture-11.md) |

</div>

Notice the shape of the ladder: **stages 1–6 are entirely about correctness** — building reference implementations and proving equivalence, mechanism by mechanism. Only stages 7–8 touch performance at all, and stage 8 is explicitly gated on stage 6's correctness suite passing first. This ordering is deliberate and mirrors the course's central argument: you cannot responsibly optimize a mechanism you have not first proven you can implement correctly, and every module in this course exists to make that proof possible for one specific mechanism at a time.

---

## 3. Source navigation, in order

For each stage, work outward from primary sources in this order:

```text
   1. the checkpoint configuration itself     — ground truth for every dimension (Module 01)
   2. the Transformers reference implementation — the canonical, if not fastest, semantics
   3. the serving implementation in your fork  — what actually runs in production,
                                                  including every fused/reordered
                                                  optimization Module 10 warned you
                                                  the conceptual diagram doesn't show
   4. the KDA and mHC papers                   — for WHY the mathematics has this
                                                  particular structure, not as a
                                                  substitute for reading the actual
                                                  checkpoint implementation
```

That ordering matters: reading a paper first and assuming the checkpoint matches it exactly is how "generic SwiGLU" replaces "clamped SwiGLU" ([Module 02 §3](Lecture-02.md)) in someone's mental model. Papers explain intent; the checkpoint and its reference implementation are the actual contract.

---

## 4. Stage 8, in detail: the optimization study

Stages 1–7 produce understanding and a cost model. Stage 8 is where that understanding earns its keep — and it has its own internal discipline, borrowed directly from [Module 11](Lecture-11.md) and from [Hardware-Aware LLM Quantization — Module 12](../Hardware-Aware%20LLM%20Quantization/Lecture-12.md):

```text
   1.  PICK ONE HYPOTHESIS from Module 10 §2's table — not "make it faster,"
       but something falsifiable: "the KDA decode kernel is register-spill-bound
       at head dimension 128," or "MoE dispatch is launch-bound below batch 8."

   2.  PREDICT the outcome using the relevant module's roofline or cost model
       BEFORE writing the optimization.

   3.  SPECIFY THE CORRECTNESS CONTRACT before benchmarking (Module 11 §4) —
       which of the Module 11 §2 test-matrix rows this change must still pass,
       and at what numerical tolerance.

   4.  IMPLEMENT the change.

   5.  RUN THE FULL Module 11 correctness suite. A speedup that fails
       the suite is not a result — it is a different, unvalidated model
       that happens to run faster.

   6.  MEASURE repeatedly, on a machine in a known state, and report
       the result whether or not it matches your stage-2 prediction.
       A wrong prediction that you can explain is worth more than a
       correct one you can't.
```

**A speedup survives this process only if it passes step 5.** This is the same principle [Module 11](Lecture-11.md) built its entire correctness matrix around, applied as the final gate rather than an afterthought: performance work on this architecture is not complete when the benchmark number improves, it is complete when the benchmark number improves *and* every mechanism-specific correctness test from stages 1–6 still passes.

---

## 5. What "mastery" means at the end of this ladder

You have completed this course when you can take the checkpoint configuration and, without consulting this course again:

* Reconstruct the attention schedule and FFN split from the layer count alone ([Module 01](Lecture-01.md)).
* Derive the router's selection-versus-weighting split and the clamped-SwiGLU asymmetry, and explain the specific bug each one's absence would cause ([Module 02](Lecture-02.md)).
* Write the KDA five-step update from memory, expand it to closed form, and explain why the decay and correction operators don't commute ([Module 03](Lecture-03.md)).
* Explain why decode and prefill need different KDA kernels for the same recurrence, and name every non-recurrence component of the sublayer ([Module 04](Lecture-04.md)).
* Derive MLA's absorption algebra and the 64× cache ratio, and explain why NoPE doesn't make the model order-blind ([Module 05](Lecture-05.md)).
* Compute DSA's indexer slot budget including the incomplete tail, and name the boundary lengths worth testing ([Module 06](Lecture-06.md)).
* Explain why four mHC residual streams is not four attention modules and 45 layers is not 45/4 effective layers ([Module 07](Lecture-07.md)).
* Enumerate every piece of state a speculative rollback must reconcile for this specific hybrid architecture ([Module 08](Lecture-08.md)).
* Build a per-GPU memory budget that does not naively divide every term by the tensor-parallel degree ([Module 09](Lecture-09.md)).
* State a falsifiable profiling hypothesis for each of seven distinct regions of the model, and know the difference between an exposed and an overlapped cross-GPU communication cost ([Module 10](Lecture-10.md)).
* Specify a numerical correctness contract before benchmarking, and name the invariant that catches a rollback or chunking bug that per-token output comparison alone would miss ([Module 11](Lecture-11.md)).

If you can recite the five mechanism names — MoE, KDA, MLA, DSA, mHC — but cannot derive any one of their update equations from the dimensions in the checkpoint config, you have vocabulary. The ladder exists to convert that vocabulary into the ability to safely modify this model without silently changing it.

---

## Current as of

* **Timeless:** the eight-stage build order and the correctness-before-performance discipline underlying stage 8.
* **Checkpoint- and deployment-specific:** stage 7's cost model and the case study threaded through this course (8× RTX 5090, PCIe Gen4, no P2P, NVFP4, `sparkinfer-frontier`) are tied to this specific deployment — the ladder itself (stages 1–6, and the discipline of stage 8) applies unchanged to any hybrid architecture combining sparse MoE, recurrent linear attention, compressed latent attention, selective retrieval, and multi-stream residuals, whatever its specific dimensions turn out to be.

---

*Course complete. [← Back to the course index](README.md) · [Phase 5 — ML Systems Engineering Guide](../Guide.md)*
