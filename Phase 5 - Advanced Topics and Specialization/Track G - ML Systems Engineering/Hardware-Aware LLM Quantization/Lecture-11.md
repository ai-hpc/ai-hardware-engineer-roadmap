# Module 11 — Hardware-Aware AutoQuant

**Collection:** [Hardware-Aware LLM Quantization](README.md) | **Previous:** [← Module 10](Lecture-10.md) | **Next:** [Module 12 →](Lecture-12.md)

---

Ten modules have produced measurements. This one turns them into a decision.

The question — *which tensor gets which format?* — has been answered ad hoc up to now ("quantize MLP, keep Q/K wide"). It is actually a well-posed **multiple-choice knapsack problem**, and posing it properly produces allocations that no amount of intuition finds. The result at the end of this module is the course's thesis in a single line: **the same throughput at less than half the behavioral damage, by reallocating precision rather than adding more of it.**

---

## Learning objectives

By the end of this module you should be able to:

1. State precision allocation as a constrained optimization with the right objective and constraints.
2. Explain why the format set is **discrete** and why that is a feature, not a limitation.
3. Implement a greedy Lagrangian solver and know when it is optimal.
4. Validate the additivity assumption the solver depends on.
5. Read a solver trace and explain each decision from Modules 04–10.

---

## 1. The problem, stated properly

```text
   maximize     BW_eff      τ(α)
                ────────  ×  ──────────
                B_token      1 + K·c

   over         f_i ∈ F   for each tensor group i

   subject to   Σ_i  ΔKL_i(f_i)   ≤  ε            behavior budget   (Module 08)
                Σ_i  n_i·b(f_i) + KV(L) ≤ V       VRAM budget       (Module 09)
                F = {BF16, FP8, NVFP4}             hardware-native   (Module 03)

   where        B_token = Σ_{i ∈ streamed}  n_i · b(f_i)             (Module 04)
```

Every term traces to an earlier module. The pieces:

| Symbol | Meaning | Source |
|---|---|---|
| `n_i` | parameters in group `i` | Module 04 ledger |
| `b(f)` | bytes/param for format `f` | Module 02 (BF16 2.0, FP8 1.0, NVFP4 0.5625) |
| `ΔKL_i(f)` | behavioral cost of format `f` on group `i` | Module 07 sweep, Module 08 grading |
| `ε` | behavior budget in nats | your product decision |
| `τ(α)` | acceptance length | Module 10 |

**The critical modelling choice is that `F` is discrete and small.** [Module 03](Lecture-03.md) eliminated every non-native format, which collapses a continuous "how many bits?" search into a three-way choice per group. That is what makes the problem exactly solvable in milliseconds instead of approximately solvable in GPU-days.

---

## 2. Why greedy works here

This is a **multiple-choice knapsack**: each group picks exactly one format. It is NP-hard in general, but two facts make it easy in practice:

* **The instance is tiny.** 5–8 groups × 3 formats. Exhaustive search is `3^8 = 6561` evaluations — under a second. **Just enumerate it if you want provable optimality.**
* **The greedy Lagrangian solution is near-optimal and interpretable.** It processes candidates in descending order of

```text
                    ΔB_token_i(f)        bytes saved
   value_i(f)  =  ─────────────────  =  ───────────────      [GB per nat]
                     ΔKL_i(f)           behavior spent
```

That ratio is the **exchange rate between speed and behavior**, and reading the solver's trace in those units is how you develop judgement about which trades are good ones.

> Enumerate for the final answer; run greedy to *understand* the answer. The trace is more valuable than the allocation.

---

## 3. The solver

```python
BPP = {"BF16": 2.0, "FP8": 1.0, "NVFP4": 0.5625}     # Module 02

def allocate(groups, budget_kl, norms_GB=0.06, formats=("BF16", "FP8", "NVFP4")):
    """groups: {name: (params_billions, {format: dKL_vs_BF16})}
       Greedy Lagrangian over a multiple-choice knapsack."""
    alloc = {g: "BF16" for g in groups}
    b_token = lambda a: sum(groups[g][0] * BPP[f] for g, f in a.items()) + norms_GB
    total_kl = lambda a: sum(groups[g][1][f] for g, f in a.items())

    trace = []
    while True:
        best = None
        for g, (n, kls) in groups.items():
            for f in formats:
                if BPP[f] >= BPP[alloc[g]]:
                    continue                                  # only downgrades
                dB = n * (BPP[alloc[g]] - BPP[f])             # GB saved
                dK = kls[f] - kls[alloc[g]]                   # nats spent
                if total_kl(alloc) + dK > budget_kl:
                    continue                                  # would blow the budget
                value = dB / max(dK, 1e-9)
                if best is None or value > best[0]:
                    best = (value, g, f, dB, dK)
        if best is None:
            break                                             # nothing affordable left
        value, g, f, dB, dK = best
        alloc[g] = f
        trace.append(dict(group=g, fmt=f, dB=dB, dK=dK, value=value,
                          b_token=b_token(alloc), kl=total_kl(alloc)))
    return alloc, trace
```

Three guard rails that belong in any production version:

```python
FORBIDDEN = {
    # Module 03: no native path on sm_120 → never propose it
    "any": {"INT3", "INT2", "FP6_nonnative"},
    # Module 07: measured, not assumed — but a sane default floor
    "k_proj": {"NVFP4"},
    # Module 10: the drafter's only job is agreeing with the target
    "mtp_head": {"NVFP4"},
}
```

---

## 4. A worked allocation

Inputs: group sizes from the [Module 04](Lecture-04.md) reconstruction; `ΔKL` values with the *ordering* established in [Module 07](Lecture-07.md) (K > Q ≫ MLP > lm_head > O > V). **The magnitudes below are illustrative — you must measure your own.** Budget `ε = 0.05` nats.

| Group | Params | ΔKL → FP8 | ΔKL → NVFP4 |
|---|---:|---:|---:|
| MLP | 16.36 B | 0.0015 | 0.012 |
| O | 3.22 B | 0.0008 | 0.006 |
| Q | 3.22 B | 0.0040 | 0.030 |
| K + V | 0.81 B | 0.0060 | 0.045 |
| lm_head | 1.27 B | 0.0012 | 0.010 |

**Solver trace** (starting from all-BF16, `B_token = 49.81 GB`):

| Step | Move | Saved | Cost | `B_token` | Σ KL | **Value (GB/nat)** |
|---:|---|---:|---:|---:|---:|---:|
| 1 | MLP → FP8 | −16.36 GB | +0.0015 | 33.45 | 0.0015 | **10,907** |
| 2 | O → FP8 | −3.22 GB | +0.0008 | 30.23 | 0.0023 | 4,025 |
| 3 | lm_head → FP8 | −1.27 GB | +0.0012 | 28.96 | 0.0035 | 1,058 |
| 4 | Q → FP8 | −3.22 GB | +0.0040 | 25.74 | 0.0075 | 805 |
| 5 | MLP → NVFP4 | −7.16 GB | +0.0105 | 18.58 | 0.0180 | 682 |
| 6 | O → NVFP4 | −1.41 GB | +0.0052 | 17.18 | 0.0232 | 271 |
| 7 | K+V → FP8 | −0.81 GB | +0.0060 | 16.37 | 0.0292 | 134 |
| 8 | lm_head → NVFP4 | −0.56 GB | +0.0088 | 15.81 | 0.0380 | 63 |

```text
   FINAL:  MLP → NVFP4   O → NVFP4   lm_head → NVFP4   Q → FP8   K+V → FP8

   B_token  =  15.81 GB        Σ ΔKL  =  0.0380
```

### The comparison that matters

| Configuration | `B_token` | Σ ΔKL | tok/s @ 1296 GB/s |
|---|---:|---:|---:|
| All BF16 | 49.81 GB | 0.000 | 26.0 |
| **Shipped** (body NVFP4, `lm_head` BF16) | **15.88 GB** | **0.093** | **81.6** |
| **Solver** (mixed) | **15.81 GB** | **0.038** | **82.0** |

```text
   ┌──────────────────────────────────────────────────────────────────┐
   │   Same throughput (82.0 vs 81.6 tok/s).                          │
   │   LESS THAN HALF the behavioral damage (0.038 vs 0.093 nats).    │
   │   Zero additional hardware. Zero additional bits.                 │
   │   Pure REALLOCATION.                                              │
   └──────────────────────────────────────────────────────────────────┘
```

**This is the entire course in one table.** The uniform-format instinct — "the body is NVFP4, so quantize the body" — spends its behavior budget on the tensors least able to afford it (Q, K) while leaving a 2.54 GB BF16 `lm_head` sitting on the critical path.

### Reading the trace

Every decision traces back to an earlier module:

* **Steps 1–2 (MLP, O → FP8) are nearly free** — 10,907 and 4,025 GB/nat. These are [Module 07's](Lecture-07.md) Path A tensors: error averages down as `η/√K`. Take them immediately.
* **Step 3 puts `lm_head` on the list before Q** — exactly [Module 04's](Lecture-04.md) finding. It is 16 % of `B_token` and was untouched in the shipped config.
* **Step 5 (MLP → NVFP4) is the single biggest byte win**, −7.16 GB, and still returns 682 GB/nat. The MLP is 58 % of the ledger; it should be the most aggressively quantized tensor in the model.
* **Q and K+V stop at FP8 and never reach NVFP4.** Their NVFP4 value ratios (~107 and ~18 GB/nat) are far below everything else, so the budget is exhausted before the solver reaches them. **The solver rediscovers [Module 07's](Lecture-07.md) rule from the numbers alone** — nobody told it that Q/K are sensitive.
* **Step 8 spends the last of the budget on `lm_head` → NVFP4** at 63 GB/nat, which is a marginal trade. In a speculative deployment you would likely stop at step 7: [Module 10's](Lecture-10.md) acceptance tax means the final 0.56 GB is partly paid back through α.

---

## 5. The additivity assumption, and how to check it

The solver assumes **`ΔKL` values add across groups**:

```text
   ΔKL(quantize A and B)  ≈  ΔKL(A)  +  ΔKL(B)
```

This is a first-order approximation and it is not exactly true — errors can compound (a damaged Q amplifies a damaged K) or partially cancel. **Verify it, do not assume it:**

```python
def check_additivity(groups, fmt, budget_pairs=10):
    """Compare measured joint KL against the additive prediction."""
    solo = {g: measure_kl(quantize(model, {g: fmt})) for g in groups}
    rows = []
    for a, b in itertools.islice(itertools.combinations(groups, 2), budget_pairs):
        joint     = measure_kl(quantize(model, {a: fmt, b: fmt}))
        predicted = solo[a] + solo[b]
        rows.append((a, b, joint, predicted, joint / predicted))
    return rows                     # ratio ≈ 1.0 → additive; > 1.2 → compounding
```

| Observed ratio | Interpretation | Action |
|---|---|---|
| 0.9 – 1.1 | additive | greedy solver is trustworthy |
| > 1.2 | errors compound | shrink `ε`, or enumerate with measured joint costs |
| < 0.8 | errors partially cancel | you are being conservative; you can spend more |

In practice additivity holds well for tensors in *different* layers and less well for tensors in the *same* attention block. Since Q and K live in the same block and are the two you most want to model correctly, **measure the Q/K joint cost explicitly** rather than summing.

---

## 6. Putting the budget in product terms

`ε` is not a number you can derive — it is a product decision. Anchor it to [Module 08's](Lecture-08.md) thresholds:

| `ε` (nats) | Expected top-1 agreement | Use case |
|---:|---|---|
| 0.01 | > 99 % | quality-critical; indistinguishable from reference |
| 0.05 | ~97–99 % | general serving — **the default** |
| 0.15 | ~93–97 % | throughput-first, tolerant workloads |

Then sweep it. The `B_token(ε)` curve is the artifact your product team can actually reason about:

```text
   B_token
     ▲
  50 │●  all BF16
     │ ╲
     │  ╲
     │   ╲___
  20 │       ╲──●────●─────●──────────────  knee: the cheap wins are gone
     │            0.02  0.04   0.10
     └──────────────────────────────────────▶  ε (nats)

   Ship at the KNEE. Past it you are buying single-digit
   throughput percentages with real behavioral cost.
```

---

## Checkpoint

You should now be able to:

1. State the allocation problem with objective, variables, and all three constraints.
2. Explain why a discrete native-format set makes the problem tractable.
3. Implement the greedy solver and say when to enumerate instead.
4. Interpret a value ratio in GB/nat and use it to accept or reject a move.
5. Test the additivity assumption and react to the result.
6. Explain why the solver's allocation beats the uniform one at equal `B_token`.

---

## Ship it

Run the full pipeline on your model:

1. Ledger the groups ([Module 04](Lecture-04.md)).
2. Measure `ΔKL` per group per format ([Modules 07](Lecture-07.md), [08](Lecture-08.md)).
3. Check additivity on the 5 highest-value pairs (§5).
4. Solve, and **enumerate to confirm** the greedy answer is optimal.
5. Sweep `ε` and plot the `B_token(ε)` curve; mark the knee.
6. Build the winning configuration and **verify the predicted `B_token` and tok/s against measurement.**

Step 6 is not optional. If the built artifact does not match the solver's prediction, either the ledger or the toolkit's actual export is wrong — and finding out which is worth more than the allocation.

---

## Current as of

* **Timeless:** the optimization statement, the multiple-choice knapsack framing, the value-ratio exchange rate, the additivity test.
* **Case-study pins:** group sizes from the Module 04 reconstruction. **The `ΔKL` table in §4 is illustrative** — chosen to reproduce the ordering measured in Module 07, not measured values. The conclusion (reallocation beats uniform quantization at equal `B_token`) is robust to the magnitudes; the specific allocation is not.
* **Refresh surface:** `F` is defined by [Module 03](Lecture-03.md). If a future runtime adds a native FP6 path on `sm_120`, add it to `F` and re-solve — the solver is unchanged.

---

**Next:** [Module 12 — Research Methodology →](Lecture-12.md)
