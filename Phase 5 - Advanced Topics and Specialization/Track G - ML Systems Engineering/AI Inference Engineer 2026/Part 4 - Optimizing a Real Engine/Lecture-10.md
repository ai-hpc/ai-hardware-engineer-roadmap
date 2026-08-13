# Part 4 · Lecture 10 — Silently Wrong: The Failure Mode Unique to Inference Engines

## Overview

Most software fails loudly. It crashes, throws, returns an error code, or produces output so obviously broken that nobody ships it. An inference engine does not have that safety net.

A language model is a function that turns any input into plausible-looking text. Corrupt a weight slice, drop an expert, skip an all-reduce, or read a tensor at the wrong stride, and the model does not crash — **it keeps writing fluent English.** Slightly worse English, from a slightly different model than the one you loaded, and you will not notice by looking.

This lecture is about that failure mode: why it is structural rather than incidental, the five families it comes in, the assertions that catch each, and the special case that should terrify you — **bugs that make the engine faster.**

By the end you should be able to look at a proposed optimization and name what it could silently break, plus the cheap check that would catch it. This is the discipline that makes every other lecture in this part safe to apply.

---

## 1. Why the safety net is missing

Three properties combine, and all three are inherent to the workload.

**The output space has no invalid values.** A corrupted image has visible artifacts; a corrupted JSON parse throws. A corrupted logit vector is *a logit vector*. Softmax it, argmax it, and you get a token. Every token is a legal token. There is no representation for "this distribution is wrong."

**The correct output is not known.** For most software you can write `assert result == expected`. For a 2.8T-parameter model at 128k context, the expected next token is whatever the model says it is. The only ground truth is another implementation of the same model — which is why [Lecture 02](Lecture-02.md) defines correctness as *agreement with a reference on identical weights and identical token ids*, and why an engine that cannot get a reference has a genuinely hard problem.

**Quality degrades smoothly and the eye is a terrible instrument.** A model whose experts are 6% dropped still writes competent prose. Humans reading samples cannot distinguish "top-1 agreement 1.00" from "top-1 agreement 0.93." The case study's own reminder: even the target quantization's top-1 against full precision is **90.4%** — so "it looks fine" has no discriminating power at the scale where bugs live.

Add the incentive from [Lecture 02](Lecture-02.md) — you are being paid for speed — and you have the setup for the worst version of this: **a bug whose only visible symptom is a better benchmark number.**

---

## 2. Anatomy of a silent bug

Here is one, in full, because reading a real one is worth more than a taxonomy. It is from [PR #148](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/148), the batched-prefill change from [Lecture 09](Lecture-09.md).

**The signature and the call.**

```text
   void attn_res_mix_f32(n_rows, act_row_stride, bank_row_stride, ...)

   all three call sites passed:
        attn_res_mix_f32(n_rows, bank_row, H)
                                 ^^^^^^^^  ^
                                 the two strides, transposed
```

So activations were read at the *bank's* pitch and banks at the *activation's* pitch. A two-argument transposition — the most ordinary bug in systems programming.

**Why nothing caught it for the life of the project.** Two independent masks, stacked:

```text
   MASK 1 — the depth mask
     bank_row = res_bank_row_elems × max_ckpt
     max_ckpt = ceil(n_layers / 12)

     at n_layers ≤ 12:   max_ckpt == 1   →   bank_row == H
                         the two strides are EQUAL, so each wrong
                         argument lands on the right value.
     bit-identical.  every short-model test passes.

   MASK 2 — the row mask
     n_rows == 1 never reads either stride.
     decode and the per-token prefill walk both pass n_rows == 1.
     → for the entire history of the engine, this code was exact.

   the chunk driver is the FIRST caller ever to pass more than one row.
```

Every existing test was in the intersection of two blind spots. The bug was not missed through carelessness; it was **unreachable** until a new feature reached it.

**How it was found.** By bisecting on layer depth against the 4-token parity probe:

```text
   layers ≤ 12   →  bit-identical
   layers = 14   →  KLD 1.64
   layers = 93   →  KLD 3.47
```

That is a beautiful diagnostic shape, and worth recognizing: **a metric that is exactly zero up to a threshold and then large is a structural bug, not numerical drift.** Precision loss accumulates gradually and roughly monotonically. A cliff at `n_layers = 13` points straight at something with a `12` in it — and `max_ckpt = ceil(n_layers / 12)` was one grep away.

> **The technique to steal.** When a parity metric is bad, do not stare at the kernel. **Sweep the structural parameters** — layers, heads, chunk width, tp_size, context depth — and find where the metric leaves zero. The threshold names the bug.

**The second bug in the same PR**, because they travel in pairs:

```c
/* the comment says DEFAULT OFF; the code says otherwise */
bool k3_kda_qkvg_batch_enabled(void) {
    const char *e = getenv("SPARKINFER_K3_KDA_QKVG_BATCH");
    return !(e && e[0] == '0');       /* unset -> TRUE. opt-OUT. */
}

/* eight lines below, the sibling gets it right */
bool k3_kda_pre_batch_enabled(void) {
    const char *e = getenv("SPARKINFER_K3_KDA_PRE_BATCH");
    return e && e[0] == '1';          /* unset -> FALSE. opt-IN. */
}
```

The feature was documented `DEFAULT OFF` in its own comment *and* in the function that consumed it, and was on. This one at least failed loudly once reached — every chunk of ≥2 tokens died with `LAUNCH FAILED at layer 0, phase Attn`.

> **Steal this.** Environment-variable gates should have exactly one helper in the codebase, used everywhere. Two idioms eight lines apart is a bug that has already happened; you are just waiting to find out which side it landed on.

---

## 3. Faster because broken

Now the part that makes this a *performance* lecture and not just a testing one.

[PR #148](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/148) first measured **169.72 tok/s** prefill. That would have been the largest single result in the project's history — a 4.2× step. It was rejected on accuracy. The honest number for the same change, once both defects above were fixed, was **98.80**.

The reason is stated exactly in the changelog, and it is the sentence to remember from this entire part:

> *"The corrupted walk was **faster** because it was reading the wrong rows."*

Reading the wrong rows means reading rows you had recently read — a cache hit instead of a cache miss. On a memory-bound workload, **incorrectness and speed are positively correlated**, because most of the cost is fetching the right bytes from the right places. Corruption in a memory-bound kernel is not a random perturbation of runtime; it is systematically *favorable*.

This inverts the intuition most engineers carry:

```text
   intuition:   a bug makes things slower or breaks them.
                a speedup is evidence the change worked.

   reality on a memory-bound engine:
                a bug that skips work, reads the wrong (nearer) bytes,
                drops an expert, or elides a collective  →  FASTER.

                the largest numbers in your sweep are the most
                likely to be wrong.
```

Three practices follow directly:

**Correctness gates before speed gates, structurally.** Not "we check both" — the accuracy result must be able to *suppress* the speed number so it never enters the record. [Lecture 02 §2](Lecture-02.md).

**A plausibility ceiling.** [PR #130](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/130) rejects a claim above 5× the reference as implausible rather than banking it. This looks paranoid until you have watched a corrupted kernel post a spectacular number. The ceiling does not assert what is achievable; it asserts that **a result too good to be true gets checked, not recorded.**

**Treat a surprising win as a bug report until proven otherwise.** If a change delivers 4× where you predicted 1.5×, your prediction was wrong *or* your code is. The prior on the second is much higher than engineers like to admit. This is the practical value of [Lecture 03 §7.1](Lecture-03.md)'s write-down-your-prediction rule: without a prediction, you have nothing to be surprised by.

---

## 4. The five families

Every silent-wrongness bug in the case study falls into one of these. The pattern is worth internalizing because the *defenses* are family-specific.

### 4.1 Configuration read wrong

The model file says one thing; the loader believes another. Nothing is corrupt — you are running a **different model** than you think.

| Trap | Mechanism | Symptom |
|---|---|---|
| `full_attn_layers` is **1-indexed** | converter tests `(il + 1) in full_attn_layers` | KDA where MLA belongs. Fluent text, wrong model. |
| MLA is **stored as MQA** | `head_count_kv = 1`, `key_length = kv_lora + qk_rope = 576`; per-layer `head_count_kv == 0` marks a KDA layer | Read as real MQA and your shard math divides 1 by 8 |
| Experts are in a **down-projected space** | `expert_latent_length 3584`, not `hidden_size 7168` | Every expert GEMM wrong by 2× |
| **Silently defaulted KV keys** | `expert_latent_length`, `attn_res.block_size`, both `situ` betas | "Loads cleanly and emits garbage" |

The defense for the last one is the best available answer to this whole family, and it lives in the *reference* engine: the pinned fork makes those four keys **required, not defaulted.** A missing key becomes a load failure instead of a wrong model.

> **Steal this.** For every model hyperparameter, ask: *if this were absent or wrong, would I get an error or a worse model?* Every parameter that answers "a worse model" should be made mandatory, with no default. A default is a silent guess.

The vision tower carries four more of these — non-square fused QKV (1536 ≠ `n_embd`), RMSNorm, bias-free, post-norm projector — each documented in the repo as *"a silent-wrong-output trap rather than a compile error."* That phrase is the correct way to annotate this class in your own code.

### 4.2 Partition arithmetic

Sharding is index arithmetic, and index arithmetic that is slightly wrong yields a model that runs.

**Bands must tile exactly.** A gap silently drops an expert's contribution; an overlap double-counts it. Either way the all-reduce combine is wrong and the output is plausible. The defense is exhaustive and cheap: `test_expert_bands_tile_exactly` walks every rank at every viable `tp_size` and asserts **each expert is owned exactly once** — and the load check proves the bands tile over real tensors with **16,351 checks**, every byte of a sharded tensor claimed by exactly one rank.

**KV heads often do not divide.** K3 stores MLA as MQA with exactly **1** KV head. At `tp_size 8` there is no way to give each rank a whole KV head, and splitting one is *incorrect* — a KV group must be visible to every query head attending through it. So when `n_kv_heads < tp_size` the KV projections are **replicated** and only the query side is sharded. The naive `n_kv_heads / tp_size` yields **0**, producing — in the repo's words — *"a model with no keys on any rank that still loads and still emits text."*

**Every axis must divide, not just the interesting one.** K3 has 896 experts and `896 % 7 == 0`, so an expert-only check happily accepts `tp_size 7` — but it also has 96 query heads and `96 % 7 != 0`. `shard_dims()` rejects the whole shape and names the offending field. The repo's note on this is refreshingly human: *"My own first draft of the test made exactly this mistake and the test caught it."*

The strategic decision behind all three is the one to copy: **the shard math is CUDA-free and unit-tested without a GPU** — 4,972 checks on `shard.cpp`, 230 on the weight plan, 44 on backend selection. The reasoning:

> *"TP bugs do not live in the collective — NCCL is correct. They live in the band arithmetic, and that is exactly the part you can verify on a laptop before burning node hours."*

### 4.3 Collective placement and count

This family is unique to distributed inference and is the most elegant of the five, because the bug is *algebraic*.

**Why the reduce sits where it does.** The routed experts are expert-sharded, so each rank's dispatch accumulator is a **partial sum** over the top-16. The next op is `ffn_routed_norm`, an RMS norm — and:

```text
   rms_norm(Σ partial)  ≠  Σ rms_norm(partial)
```

RMS norm is **not linear**, so the cross-rank sum must complete *before* it. Move the reduce two ops later, after `routed_up`, and two things go wrong at once: you skip the reduce the experts needed, *and* — because `routed_norm` / `routed_up` / `shexp` are all replicated, so every rank already holds the complete tensor — you reduce a complete tensor and **multiply the FFN output by `tp_size`.**

Multiplying an FFN output by 8 does not crash. It produces fluent text.

**And the count is asserted, not eyeballed.** `collectives/token 92` — one all-reduce per MoE layer, 93 layers minus the leading dense block. *"A missing reduce leaves a partial expert sum; an extra one multiplies a complete tensor by `tp_size`. Neither crashes, so the count is asserted rather than eyeballed."*

The general rule, which applies to any parallel forward pass:

> **A collective's position is determined by the first non-linearity downstream of the partial sum.** Derive it from the algebra, assert the count, and never place it from a diagram.

Two more defenses in this family worth naming:

* **The reduction is verified by construction.** `tp_allreduce_check` has rank *r* fill its buffer with `r + 1`, so the only correct total is `tp(tp+1)/2`. A missing rank, a double-counted rank, or a rank reducing the wrong buffer each produce a *different* number. And **every rank is checked, not just rank 0** — a collective that is correct on one rank and wrong on another is a real bug, and checking only rank 0 is how it ships.
* **Precision is part of correctness.** K3 runs an f32 residual stream deliberately. Routing it through a bf16 all-reduce would truncate to ~8 mantissa bits **at every layer boundary**, undoing the executor's numerics. So `make_collective(..., need_f32=true)` downgrades a bf16-only fast backend to NCCL *before* the 20-minute weight load, rather than failing at the first collective. Fail early, at the cheap moment.

The positive result is what this buys: TP and layer-split pipeline outputs agree to **1.85e-09**, and the all-reduce is exact to **1.127e-07** of peak — below f32 epsilon, one ulp.

### 4.4 Work that silently does not happen

The nastiest family, because the code is correct and simply is not running.

The canonical case is [PR #33](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/33): *"MLA decode attention **silently stops launching** past ~11.7k context."* A launch configuration exceeded a limit, the launch failed, the return code was not checked, and the forward pass continued with a stale or zeroed buffer. Under ~11.7k it worked perfectly. Past it, the attention output was garbage — and the engine kept generating.

The fix shipped alongside [PR #49](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/49): *"**fail the forward on a failed launch**." One line of error checking, and an entire class of bug becomes loud.

The same shape appears throughout the harness, and this is the tell that a project has understood the pattern:

| PR | The silent pass |
|---|---|
| [#88](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/88) | "report a `llama-tokenize` failure instead of exiting silently" |
| [#100](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/100) | "the baseline wrote a slot the harness never reads" |
| [#112](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/112) | "three harness defects that make a broken run look like a good one" |
| [#125](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/125) | "harness silent-pass on empty `--ids` and failed logits write" |

Read that list as one bug repeated: **an empty result and a successful result were indistinguishable.** Empty input, unwritten output, unread slot, failed launch — every one produced a green run.

> **The rule.** Every stage must be able to distinguish "did nothing" from "did the right thing." If your pipeline's success condition is "no error was raised," you do not have a success condition. Assert on the *presence and shape of output*, not the absence of failure.

### 4.5 Numerics that are correct but not the numerics you specified

Milder, but it is what a parity gate spends most of its time on. The case study's founding example is [PR #7](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/7): *"llama.cpp parity — beta sigmoid + KDA decay axis (**KLD 5.34 → 4e-3**)."*

A KL of 5.34 nats is not a subtle drift — it is a different model. The causes were a wrong activation variant and a **decay applied along the wrong axis**. Both produce fluent output; both are invisible without a reference; both were found by the gate rather than by reading samples.

And the counterpart worth holding onto: **not all divergence is a bug.** Mean KLD in this engine sits ~400× above the 1e-5 you would expect from two runs of the same implementation, from an identified cause — K3 keeps f32 activations where ggml quantizes them before a quantized mat-vec. The instruction to contributors is exactly right: *"Do not go hunting it as a bug; do not make it worse."*

> Knowing your **irreducible** divergence, and its cause, is what lets you treat any *change* in divergence as signal. A team that has not characterized its floor cannot use its gate.

---

## 5. The defense stack

Ordered by cost. The cheap ones catch most of it.

```text
   FREE, no GPU
     · shard / partition math as pure functions, exhaustively tested
       (4972 + 230 + 44 checks here, all on a laptop)
     · every axis validated to divide, with the offending field named
     · required-not-defaulted config keys
     · one env-gate idiom, used everywhere
     · lint EVERY script, not a hand-maintained list

   CHEAP, one run
     · check every launch's return code; fail the forward
     · assert the collective COUNT per token
     · assert bands tile exactly over real tensors
     · verify the reduce by construction (r+1 ⇒ tp(tp+1)/2), on EVERY rank
     · compute-sanitizer clean: 0 errors

   PER-CHANGE
     · parity vs reference at multiple depths, worst-of not average
     · bit-identity claims PROVED bit-identical
     · structural-parameter sweep when parity degrades (§2)
     · plausibility ceiling on the speed number

   ONGOING
     · correctness gate ordered BEFORE the speed gate
     · known irreducible divergence documented, with its cause
     · pin-drift audit on external references
```

Note how much of the top tier is free. The most expensive bugs in this case study — expert bands, KV-head division, reduce placement — are all catchable by pure functions on a laptop. **The GPU is where you validate plumbing, not arithmetic.**

One more, from the same family and easy to overlook: [PR #97](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/97) found that the copycat guard *"has never run — **100 of 100 runs were startup failures**."* A hundred green rounds, zero coverage. A check whose failures look like passes is worse than no check, because it retires the worry.

> **Assert your assertions ran.** A guard needs a positive signal that it executed and evaluated something. "No failure reported" is the same output as "never started."

The CRLF story is the same lesson in miniature: a script committed with CRLF had heredoc terminator `PY\r`, which never matches `PY`, so `bash` reported "unexpected end of file" and **the script could not execute at all.** Found *"within seconds of linting every script instead of four."* Hand-maintained lists of what to check are a way of not checking things.

---

## 6. Negative results, and the revert that was wrong

A ladder with no failures has been edited. Here is what the real distribution looks like.

### 6.1 Merged, reverted, reapplied

The most valuable three-PR sequence in the repository:

```text
   #81   MERGED, eval:s
         "quantise each activation once, not once per projection
          (+4.8%, bit-identical)"

   #84   MERGED
         "Revert #81 — 31.5% regression on current main"

   #94   MERGED
         "Reapply #81 — THE REVERT WAS BASED ON A BAD MEASUREMENT"
```

A real optimization was reverted on a bad measurement and had to be re-landed. Three lessons, in increasing order of discomfort:

1. **Your regression detector is a measurement too, with the same failure modes as the thing it grades.** A 31.5% "regression" is a large number, and large numbers feel authoritative. This one was noise, box variance, or a bad build.
2. **Revert is not a free action.** It is a change to `main`, and it deserves the same standard of evidence as the change it undoes. Reverting on one bad reading cost two extra PRs and left the ladder temporarily wrong.
3. **You can only recover if the history is intact.** Reapplying correctly required going back to the sealed receipts and establishing which of two contradictory measurements was real. This is [Lecture 02](Lecture-02.md)'s payoff.

### 6.2 The `eval:none` catalog

`eval:none` means measured, real, and **not big enough to score** — a gain inside the 2% significance gate.

| PR | Claimed | Outcome |
|---|---|---|
| [#113](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/113) | five decode-path changes, +15.4% at 128k | `eval:none`, closed |
| [#104](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/104) | MoE act-fuse + packed IQ1_S + MLA pipe, +10.4% at 128k | `eval:none`, closed |
| [#123](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/123) | let the RMS norm emit its own Q8_0 | `eval:none`, closed |
| [#129](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/129) | stage MLA broadcast shared reads, +3.4% bit-identical | closed |
| [#71](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/71) | widen the last single-block launch | `eval:none` — **twice** |

Two things to read off this table.

**A double-digit claim can measure zero.** [#113](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/113) and [#104](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/104) each claimed >10% and scored `none`. This is the normal outcome of a self-measured microbenchmark meeting an interleaved same-box harness — and it is why the contribution rules require an *end-to-end* improvement rather than an isolated kernel benchmark.

**The same idea scored `none` twice and then `s`.** [#71](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/71) was "widen the last single-block launch"; it scored `none` on two attempts. The identical idea later merged as [#115](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/115) with an `s` tier and a memorable title — *"327 norms/token were running on 128 threads"* ([Lecture 04](Lecture-04.md)).

What changed was **the rest of the engine.** The significance gate is 2% *of the frontier*, so it is an absolute bar in tok/s that rises as you improve. A win too small to measure at one frontier can be measurable at another — or, in the other direction, a fixed-size win becomes proportionally *harder* to score as you get faster. The repo's own note: *"A real but small win now scores `none` — see #71, twice."*

> **A rejected optimization is rejected against a state, not for all time.** Keep the branch. Re-measure after the frontier moves.

### 6.3 What to record

For your own artifact, from [the Part 4 README](README.md):

* Every change, including the ones that measured nothing.
* The claim *and* the measured result, so the gap is visible.
* Reverts, and whether the revert was later found to be wrong.
* Bit-identity claims and how they were proved.
* Retracted numbers, with what produced them.

The case study's changelog contains a section literally headed **"Note on a retracted number,"** documenting the 169.72 alongside the real 98.80. That is what an auditable record looks like, and it costs one paragraph.

---

## 7. The checklist

Before merging any optimization, in order:

1. **What could this silently break?** Name the family from §4. If you cannot name one, you do not understand the change yet.
2. **Does anything assert it did not?** Not "did no error occur" — is there a positive check on the shape and presence of output?
3. **If it claims bit-identical, is it proved bit-identical?** Same inputs, byte-compared outputs. "Should be" is not a measurement.
4. **What structural parameter would expose it?** Layers, heads, chunk width, `tp_size`, context depth. Test at more than one value of each the change touches (§2).
5. **Did the win match the prediction?** A surprise is a bug report until proven otherwise (§3).
6. **Is it too good to be true?** If it beats your plausibility ceiling, check before banking.
7. **Was the parity gate exercised at depth?** A change to KV cache, attention, routing, or the LM head is exactly the class the shallow probes miss.
8. **Did every guard actually run?** Look for the positive signal, not the absence of red (§5).

---

## Lab — build a corruption suite

Goal: prove your gate catches silent wrongness, by writing the bugs yourself.

1. **Write five deliberate corruptions**, one per family in §4. Suggested: shift a config index by one; introduce a one-element gap in a partition; move a collective past a non-linearity; make a kernel launch fail without checking; change an activation variant.
2. **For each, record three things**: does it crash? does the output look plausible to you by eye? what is the worst-depth KL against your reference?
3. **Time each corrupted build.** Note which ones are *faster* than correct. Expect at least one.
4. **Confirm your gate rejects all five.** Any that pass, fix the gate — that is the real output of this lab.
5. **Sweep a structural parameter** on the corruption you made depth- or width-dependent, and find the threshold where parity leaves zero. Confirm the threshold points at the bug (§2).
6. **Assert your assertions.** Pick your most important guard, make it *not run*, and check whether your CI still goes green. Fix it so it cannot.
7. **Write `CORRUPTIONS.md`** — the five bugs, their symptoms, their timings, and the check that catches each. Commit it. It is the most useful test documentation you will write.

Pass criterion: a corrupted build that is measurably *faster* than `main` and is rejected by your harness before its speed is ever reported.

---

## Self-check

1. Why can a language model not detect that its own weights are 6% wrong? Answer in terms of the output space.
2. A parity metric reads exactly 0 for `n_layers` ≤ 12, then 1.64 at 14 and 3.47 at 93. What class of bug is this, and what expression would you grep for?
3. Your all-reduce sits after the FFN up-projection instead of before the routed norm. Both versions run and produce fluent text. Explain algebraically what each computes and why the wrong one is off by a factor.
4. Your engine keeps an f32 residual stream and your fast collective backend is bf16-only. Describe the corruption, its magnitude, how many times per token it occurs, and where the check belongs.
5. A PR claims "bit-identical, +4.8%." What exactly do you require before believing the first half?
6. A change measures 4.2× where you predicted 1.4×. List the checks you run before celebrating, in order.
7. An optimization scored `none` twice and merged the third time with no code change. What moved, and what does that imply about keeping rejected branches?
8. Your CI has been green for 100 runs. Give two distinct reasons that is not evidence your guard works.
9. `n_kv_heads = 1` and `tp_size = 8`. Give the naive shard result, the output it produces, and the correct handling.

---

## References

* **SparkInfer-K3 correctness record** — the stride transposition and retracted 169.72 in [`CHANGELOG.md`](https://github.com/gittensor-ai-lab/sparkinfer-k3/blob/main/CHANGELOG.md) ("Note on a retracted number"); the config traps in [`docs/technical.md`](https://github.com/gittensor-ai-lab/sparkinfer-k3/blob/main/docs/technical.md); the partition and collective-placement traps in [`docs/tensor-parallel.md`](https://github.com/gittensor-ai-lab/sparkinfer-k3/blob/main/docs/tensor-parallel.md).
* **The revert cycle** — [#81](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/81) → [#84](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/84) → [#94](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/94). Worth reading in order.
* **Silent-launch class** — [#33](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/33) (stops launching past ~11.7k), [#49](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/49) (fail the forward on a failed launch), [#112](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/112), [#125](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/125).
* **compute-sanitizer** — [docs.nvidia.com/compute-sanitizer](https://docs.nvidia.com/compute-sanitizer/) — `memcheck`, `racecheck`, `initcheck`, `synccheck`. Required clean in this repo.
* **KL divergence and top-1 agreement as a correctness gate** — [Logprobs, Perplexity & KL Divergence — Lecture 05](../../Logprobs,%20Perplexity%20and%20KL%20Divergence/Lecture-05.md), which derives the metrics §4.5 relies on.
* **"Which Quantization Should I Use?"** — [arXiv:2601.14277](https://arxiv.org/abs/2601.14277) — intrinsic metrics are necessary but not sufficient; relevant to what a parity gate cannot see.

Cross-references:

* [Lecture 02 — The scoreboard](Lecture-02.md) — gate order, the plausibility ceiling, the sealed receipts that made §6.1 recoverable.
* [Lecture 03 — Diagnosis](Lecture-03.md) — the write-down-your-prediction rule that makes §3 operational.
* [Lecture 07 — Sharding 896 experts](Lecture-07.md) — the reduce-placement algebra of §4.3 in its performance context.
* [Lecture 09 — Batched prefill](Lecture-09.md) — the change whose two defects §2 dissects.

---

## Current as of 2026-08

SparkInfer-K3 at `7689cc7`. Parity gate: top-1 ≥ 0.95, mean KL ≤ 0.05, seven depths, worst-of; merged runs sit at 0.004–0.008. Known irreducible divergence ~400× the same-implementation floor, cause identified (f32 activations vs pre-quantized mat-vec). TP/pipeline agreement 1.85e-09; all-reduce exact to 1.127e-07 of peak. CUDA 12.8+, `compute-sanitizer` clean required. The five families and the defense stack are the durable content.

---

## Next

* This is the last lecture of Part 4. Up: [Part 4 — Optimizing a Real Engine](README.md)
* Previous: [Lecture 09 — The phase you forgot: batched prefill](Lecture-09.md)
* Course home: [AI Inference Engineer 2026](../README.md)
