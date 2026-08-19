# Module 12 — Research Methodology: Ablations That Actually Prove Something

**Collection:** [Hardware-Aware LLM Quantization](README.md) | **Previous:** [← Module 11](Lecture-11.md) | **Next:** [Capstone →](Lecture-13.md)

---

Every number in this course came from a measurement, and every measurement can be wrong in ways that look exactly like a result. This module is about the discipline that separates the two.

It matters more here than in most performance work, because quantization results are **doubly confoundable**: you are changing both a speed variable and a quality variable at once, on hardware whose clocks move under you, evaluated with metrics that have real variance. A sloppy protocol will produce a publishable-looking speedup from thermal drift alone.

---

## Learning objectives

By the end of this module you should be able to:

1. Lock a GPU into a reproducible measurement state.
2. Design a one-variable-at-a-time ablation and recognize when you have violated it.
3. Report a result with the variables that make it interpretable.
4. Identify the five invalid comparisons that account for most published quantization claims.
5. Recognize the failure mode where a **bug improves the benchmark**.

---

## 1. Lock the machine first

An unlocked GPU is not a measurement instrument. Clocks move with temperature, power, and neighbouring work, and the drift is easily larger than the effect you are hunting.

```bash
# Persistence mode: keep the driver resident (avoids first-call initialization skew)
sudo nvidia-smi -pm 1

# Lock clocks. Pick a value the card can sustain INDEFINITELY, not a boost peak.
nvidia-smi -q -d SUPPORTED_CLOCKS | head -40
sudo nvidia-smi --lock-gpu-clocks=2400,2400
sudo nvidia-smi --lock-memory-clocks=14001,14001      # GDDR7: check your part's value

# Verify under load, not at idle:
nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,\
clocks_throttle_reasons.active --format=csv -l 1
```

```text
   THERMAL STATE IS A VARIABLE.
   ────────────────────────────────────────────────────────────────
   run 1 (cold, 42 °C)  :  84.1 tok/s
   run 2 (warm, 71 °C)  :  81.6 tok/s          ← a 3 % "regression" from physics
   run 3 (hot,  79 °C)  :  78.9 tok/s          ← now you are throttling

   Fix: fixed warmup, then measure. Same thermal state for EVERY configuration.
```

The protocol:

```python
BENCH = dict(
    warmup_iters   = 20,        # discard; brings clocks and caches to steady state
    measure_iters  = 100,
    repeats        = 5,         # separate PROCESSES, not just loops
    interleave     = True,      # A,B,A,B,A,B — never AAAA then BBBB
    report         = "median + IQR",   # not mean; one outlier should not move it
)
```

**`interleave = True` is the single highest-value line here.** Running all of configuration A and then all of configuration B confounds your comparison with everything that drifts over time — temperature, background processes, memory fragmentation, another user's job. Interleaving converts a drift confound into noise, which averaging can handle.

---

## 2. One variable at a time

```text
   INVALID                                    VALID
   ────────────────────────────────           ──────────────────────────────
   baseline:  BF16, K=3, ctx 4K, vLLM         baseline:  NVFP4, K=3, ctx 4K, vLLM
   candidate: NVFP4, K=5, ctx 8K, TRT-LLM     candidate: NVFP4, K=5, ctx 4K, vLLM
              ▲      ▲       ▲       ▲                          ▲
              └──── four changes ────┘                    one change
        "NVFP4 gave us 2.3×"  ← attributable to nothing
```

The variables that must be **held fixed** across a quantization comparison, and the reason each one bites:

| Variable | Why it confounds |
|---|---|
| **Draft depth `K`** | changes τ directly ([Module 10](Lecture-10.md)) |
| **The drafter itself** | acceptance measures target-vs-drafter; moving both is uninterpretable ([Module 08](Lecture-08.md)) |
| **Context length** | up to 45 % throughput swing ([Module 09](Lecture-09.md)) |
| **Batch size / concurrency** | moves you along the roofline ([Module 01](Lecture-01.md)) |
| **Runtime + version** | kernel coverage changes between releases ([Module 03](Lecture-03.md)) |
| **Sampling parameters** | temperature and top-p change acceptance |
| **Prompt set** | acceptance is strongly content-dependent |
| **Clocks / thermal state** | §1 |

**The drafter one is the subtle one and the most commonly violated.** If you quantize the whole checkpoint including the MTP head and then compare acceptance to the BF16 baseline, you have changed both `p` and `q`. The acceptance delta no longer measures target drift, and the [Module 08](Lecture-08.md) identity `α = 1 − TV(p,q)` no longer isolates anything.

---

## 3. Paired comparison and effect size

Run configurations on **identical prompts in identical order** and compare per-prompt, not in aggregate. Pairing removes prompt-difficulty variance, which is usually the largest noise source:

```python
def paired_compare(cfg_a, cfg_b, prompts, repeats=5):
    """Per-prompt paired deltas. Returns effect size and a confidence interval."""
    deltas = []
    for p in prompts:
        for r in range(repeats):
            a = run(cfg_a, p, seed=r)          # same seed, same prompt, both configs
            b = run(cfg_b, p, seed=r)
            deltas.append(b.tok_s - a.tok_s)   # PAIRED delta
    d = np.array(deltas)
    se = d.std(ddof=1) / np.sqrt(len(d))
    return {
        "mean_delta": d.mean(),
        "ci95":       (d.mean() - 1.96 * se, d.mean() + 1.96 * se),
        "significant": abs(d.mean()) > 1.96 * se,
        "n":          len(d),
    }
```

And hold yourself to the acceptance sample sizes from [Module 08 §5](Lecture-08.md):

```text
   claim a 1-point acceptance difference    →  ≥ ~8,000 draft tokens per config
   claim a 0.4-point difference             →  ≥ ~47,000 draft tokens per config

   Below that, "2.886 → 2.871" is a coin flip with a decimal point.
```

---

## 4. The five invalid comparisons

These account for most quantization claims you will read — including, if you are not careful, your own.

### 4.1 Comparing against a badly-configured baseline

```text
   "Our NVFP4 build is 3.1× faster than BF16."
   ...where the BF16 baseline used a naive kernel, no CUDA graphs, and batch 1
      while the NVFP4 build had all three.
```

**Fix:** the baseline must be as well-optimized as the candidate. If you tuned one, tune both — or state plainly that you did not.

### 4.2 Reporting throughput without context length

[Module 09](Lecture-09.md): 81.6 tok/s at 0 context, 45.1 tok/s at 262 K, same model. A tok/s number without a context length is not a measurement.

**Fix:** report tok/s at your p50 **and** p99 serving context.

### 4.3 Measuring quality at short context only

Every metric in [Module 08](Lecture-08.md) is typically run at 2–4 K. [Module 07's](Lecture-07.md) RoPE argument predicts Q/K damage *grows* with context. A configuration that passes at 4 K can fail at 262 K, and nothing in a short-context evaluation will warn you.

**Fix:** include a long-context probe in the gate. Always.

### 4.4 Comparing checkpoint sizes instead of `B_token`

```text
   "We cut the model 34 % and got 4 % more throughput. Quantization
    doesn't deliver what it promises."
```

It delivered exactly what the ledger predicted. The 34 % included embeddings and a vision tower that are never read during decode ([Module 04](Lecture-04.md)).

**Fix:** report `B_token`, and report achieved bandwidth as `B_token × tok/s`.

### 4.5 Ignoring the acceptance tax

```text
   "+1.9 % throughput" — while τ fell from 2.792 to 2.546.
```

[Module 10](Lecture-10.md): the byte win was 11.8 % and the acceptance tax consumed 84 % of it. Reporting only the net throughput hides both that the change was worth far more than it delivered *and* that it damaged the model.

**Fix:** report `B_token`, τ, and tok/s together. Always all three.

---

## 5. When a bug makes the benchmark better

This is the failure mode that survives every protocol above, because the numbers are real — they are just measuring a different computation than you think.

```text
   Symptoms of a benchmark improved by a bug:
   ───────────────────────────────────────────────────────────────
   ✗ speedup EXCEEDS the byte-ledger prediction        ← physics violated
   ✗ throughput improves and NOTHING got smaller
   ✗ acceptance length goes UP after quantizing the target
   ✗ perplexity improves after quantization
   ✗ long-context results improve while short-context are unchanged
```

Real causes behind each:

| Symptom | Common cause |
|---|---|
| Beats the byte-ledger bound | some layers silently skipped; a shape mismatch fell back to a no-op |
| Acceptance rises after quantizing the target | draft and target accidentally sharing weights → they agree trivially |
| Perplexity improves | evaluation contaminated by calibration data; or the eval is running the *reference* model |
| Long context improves | the cache is silently truncating; you are not attending to the full context |
| Faster with nothing smaller | output tokens got shorter (EOS emitted early) — check tokens generated, not just rate |

**The defence is the ledger.** [Module 04](Lecture-04.md) gives you an *a priori* bound:

```text
   predicted_tok/s  =  BW_peak × 0.92 / B_token

   If measured > predicted, you have not found an optimization.
   You have found a bug. Physics does not have a fast path.
```

That check has no false negatives worth worrying about, costs nothing, and is the reason to build the ledger before running the experiment rather than after.

A second, cheaper check: **always verify output token counts and a sample of the actual generated text.** A model that emits `<eos>` immediately has spectacular tok/s.

---

## 6. The reporting standard

Every configuration in your ablation table carries these fields. If a field is missing, the row is not interpretable:

```yaml
# ─── identity ────────────────────────────────────────────
config_name:      mixed-fp8-nvfp4-v3
git_commit:       a1b2c3d
quant_manifest:   sha256:...          # Module 05 §5

# ─── the model ───────────────────────────────────────────
allocation:       {mlp: NVFP4, o: NVFP4, q: FP8, kv: FP8, lm_head: NVFP4}
B_token_GB:       15.81               # Module 04 — REQUIRED
checkpoint_GB:    20.19               # for reference; NOT the throughput driver

# ─── the environment ─────────────────────────────────────
gpu:              RTX 5090 (sm_120)
driver / cuda:    ...
runtime:          vllm 0.x.y          # Module 03: kernel coverage moves with version
sm_clock_MHz:     2400 (locked)
mem_clock_MHz:    14001 (locked)
temp_steady_C:    68

# ─── the workload ────────────────────────────────────────
context_length:   4096                # Module 09 — REQUIRED
batch_size:       1
draft_depth_K:    3                   # Module 10 — REQUIRED
drafter:          mtp_head @ BF16 (FIXED across all configs)
sampling:         {temperature: 0.7, top_p: 0.95}
prompt_set:       sha256:...
n_draft_tokens:   48000               # Module 08 §5 — REQUIRED for acceptance claims

# ─── results ─────────────────────────────────────────────
tok_s_median:     82.0
tok_s_iqr:        [81.4, 82.7]
predicted_tok_s:  82.0                # from B_token — must BOUND the measurement
acceptance_tau:   2.871
alpha:            0.783
mean_kl:          0.038
p99_kl:           0.21
top1_agreement:   0.984
achieved_BW_pct:  72.3                # B_token × tok/s / peak
```

Three fields do the heavy lifting: **`B_token`** makes the speed claim checkable, **`predicted_tok_s`** makes a bug detectable, and **`n_draft_tokens`** makes the acceptance claim believable. Most reports omit all three.

---

## 7. The experiment protocol, end to end

```text
   1. WRITE THE HYPOTHESIS DOWN FIRST
      "Moving Q from NVFP4 to FP8 will cost 1.81 GB of B_token (+11 % traffic)
       and recover ≥ 0.02 nats of KL and ≥ 0.15 of acceptance length."
      ↑ Specific and falsifiable. If you cannot write this, you are not running
        an experiment — you are browsing.

   2. PREDICT THE OUTCOME from the ledger and the solver.  (Modules 04, 11)

   3. FIX EVERY OTHER VARIABLE.                             (§2)

   4. RUN INTERLEAVED, WITH ENOUGH SAMPLES.                 (§1, §3)

   5. CHECK AGAINST THE PREDICTION.
      measured ≈ predicted   →  the model holds; you understand the system
      measured ≪ predicted   →  something else is binding; profile   (Module 03)
      measured ≫ predicted   →  BUG. Do not celebrate.               (§5)

   6. REPORT ALL THREE AXES: B_token, tok/s, and behavior. Never one alone.

   7. WRITE THE NEGATIVE RESULTS DOWN TOO.
      "Q → NVFP4 gains 1.9 % and costs 8.8 % acceptance" is one of the most
      valuable findings in this entire course, and it is a negative result.
```

---

## Checkpoint

You should now be able to:

1. Lock a GPU into a reproducible state and verify it under load.
2. List the eight variables that must be held fixed, and explain the drafter one.
3. Run a paired comparison and state whether a delta is significant.
4. Name the five invalid comparisons and the fix for each.
5. Use the ledger bound to detect a bug masquerading as a speedup.
6. Fill out the reporting standard completely for one configuration.

---

## Ship it

Take one result you already believe — from this course or your own work — and **re-run it under this protocol**. Locked clocks, interleaved, paired, adequately sampled, fully reported.

Then answer honestly: **did it survive?** If it did, you now have a result you can defend. If it did not, you have learned something more valuable than the original number, and you found it before someone else did.

---

## Current as of

* **Timeless:** all of it. Measurement discipline does not depend on hardware generation.
* **2026 tooling pins:** `nvidia-smi --lock-gpu-clocks` / `--lock-memory-clocks` syntax; memory-clock values are part-specific — query `SUPPORTED_CLOCKS` rather than copying the example.
* **Related:** [AI Inference Engineer 2026 — Part 4, Lecture 02](../AI%20Inference%20Engineer%202026/Part%204%20-%20Optimizing%20a%20Real%20Engine/Lecture-02.md) builds a scoreboard that cannot be gamed, and [Lecture 10](../AI%20Inference%20Engineer%202026/Part%204%20-%20Optimizing%20a%20Real%20Engine/Lecture-10.md) covers the bug-improves-the-benchmark failure on a real engine.

---

**Next:** [Capstone — TurboQuant →](Lecture-13.md)
