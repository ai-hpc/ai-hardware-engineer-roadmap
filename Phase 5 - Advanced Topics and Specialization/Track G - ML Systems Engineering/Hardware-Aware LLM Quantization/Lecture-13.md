# Capstone — TurboQuant: Build the Policy Engine

**Collection:** [Hardware-Aware LLM Quantization](README.md) | **Previous:** [← Module 12](Lecture-12.md) | **Next:** [Course index](README.md)

---

The twelve modules produced a method. This capstone turns it into a **reusable subsystem another engineer can run on a model you have never seen** — Level 5 on the roadmap's [artifact ladder](../../../Curriculum-Authoring-Guide.md).

**The benchmark to beat:** `155.75 tok/s` at acceptance length `2.886` on an RTX 5090, from the published [DSpark NVFP4 build](https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-DSpark-NVFP4). Not by shrinking the file. By **allocating precision correctly and fixing what the ledger says is actually binding.**

---

## 1. Where the headroom is

Before building anything, decompose the target with the tools from Modules 04 and 10. Start from the non-speculative measurement:

```text
   BW_eff / B_token  =  1296 / 15.88  =  81.6 tok/s        (non-speculative, verified)
```

Then invert the speculative speedup model against the observed 155.75:

```text
                 τ                              81.6 × 2.886
   tok/s  =  base × ──────────   ⟹   1 + K·c  =  ────────────  =  1.512
                 1 + K·c                            155.75

   at K = 3   ⟹   c  =  0.171
```

Now compare that against what the byte ledger says the drafter *should* cost:

```text
   MTP head = 0.85 GB,  target = 15.88 GB   ⟹   c_bandwidth  =  0.054

   measured c = 0.171     ≈  3.2× MORE EXPENSIVE than its bytes justify
```

```text
   ┌──────────────────────────────────────────────────────────────────┐
   │  The draft step costs 17 % of a target pass while moving only     │
   │  5 % of the bytes. That gap is launch overhead and kernel         │
   │  inefficiency in the drafting path — not physics.                 │
   │                                                                   │
   │  Fixing it alone:  81.6 × 2.886 / (1 + 3×0.054)  =  202.9 tok/s   │
   └──────────────────────────────────────────────────────────────────┘
```

**That is a +30 % finding derived from arithmetic, before touching a single weight.** It is also the second time the ledger has found headroom that is not a quantization problem — the first was [Module 04's](Lecture-04.md) 72 %-vs-92 % bandwidth gap.

### The staged target ladder

| Stage | Change | Factor | Cumulative |
|---|---|---:|---:|
| — | baseline | — | **155.8 tok/s** |
| 1 | kernel efficiency: 72 % → 92 % achieved BW ([Mod 03](Lecture-03.md), [04](Lecture-04.md)) | ×1.273 | 198.3 |
| 2 | `lm_head` BF16 → NVFP4 ([Mod 04](Lecture-04.md), [11](Lecture-11.md)) | ×1.130 | 224.1 |
| 3 | drafter path: `c` 0.171 → 0.08 ([Mod 10](Lecture-10.md)) | ×1.220 | 273.5 |
| 4 | reallocation recovers acceptance ([Mod 11](Lecture-11.md)) | ×1.022 | **279.5** |

**Set your commitment at Stage 2 (~224 tok/s) and your stretch at Stage 4 (~280).** Stages 1 and 3 are kernel work and may be gated by what your runtime exposes; stages 2 and 4 are yours regardless.

Note what is *not* on this ladder: quantizing the transformer body further. It is already NVFP4, it is 58 % MLP which is already the right choice, and going below 4 bits is ruled out by [Module 03](Lecture-03.md). **The remaining wins are elsewhere, and the ledger is what told you so.**

---

## 2. System architecture

```text
   ┌─────────────────────────────────────────────────────────────────────┐
   │                            TurboQuant                                │
   ├─────────────────────────────────────────────────────────────────────┤
   │                                                                      │
   │   [1] LEDGER          checkpoint ──▶ traffic classes, B_token,      │
   │       (Module 04)                     predicted ceiling, gap check   │
   │              │                                                       │
   │              ▼                                                       │
   │   [2] PROBE           per-group activation stats  (Module 06)        │
   │                       per-group ΔKL / Δτ sweep     (Modules 07, 08)  │
   │                       additivity check             (Module 11 §5)    │
   │              │                                                       │
   │              ▼                                                       │
   │   [3] SOLVER          multiple-choice knapsack over {BF16,FP8,NVFP4} │
   │       (Module 11)     hardware guard rails         (Module 03)       │
   │              │        ε sweep → B_token(ε) curve                     │
   │              ▼                                                       │
   │   [4] BUILDER         emit the quantized checkpoint + manifest       │
   │       (Module 05)     AWQ / GPTQ per the error-mode diagnosis        │
   │              │                                                       │
   │              ▼                                                       │
   │   [5] VERIFIER        locked clocks, interleaved, paired  (Mod 12)   │
   │       (Modules 8,12)  tok/s · τ · α · KL · p99 · agreement           │
   │                       measured vs PREDICTED ── bug check             │
   └─────────────────────────────────────────────────────────────────────┘
```

The contract between stages is what makes it reusable:

```python
@dataclass
class Ledger:      # [1] → [3]
    groups: dict[str, int]          # name → parameter count
    traffic_class: dict[str, str]   # A_streamed / B_gathered / C_conditional / D_state
    b_token_GB: float
    predicted_tps: float
    measured_tps: float | None
    gap: float | None               # predicted / measured — >1.2 ⇒ profile, do not quantize

@dataclass
class Sensitivity: # [2] → [3]
    d_kl:   dict[tuple[str, str], float]   # (group, format) → ΔKL vs BF16
    d_tau:  dict[tuple[str, str], float]   # (group, format) → Δ acceptance length
    additivity_ratio: float                # ~1.0 ⇒ solver assumption holds

@dataclass
class Allocation:  # [3] → [4]
    fmt: dict[str, str]
    predicted_b_token_GB: float
    predicted_kl: float
    predicted_tps: float            # [5] MUST check against this
```

---

## 3. Build order

### Phase 1 — Ledger and the gap check *(start here, always)*

Implement [Module 04's](Lecture-04.md) analyzer. Compute `B_token`, predicted ceiling, and the gap.

```text
   gap = predicted / measured

   ≈ 1.0   →  you are at the bandwidth wall; proceed to Phase 2
   > 1.2   →  STOP. Profile kernels first (Module 03 §5).
              For the case study, gap = 1.27 — Stage 1 of the ladder.
```

**Deliverable:** ledger table, gap, and a written verdict naming your next action.

### Phase 2 — Probe

Run [Module 06's](Lecture-06.md) activation profiler (diagnosis → method selection) and [Module 07's](Lecture-07.md) leave-one-out sweep (per-group ΔKL and Δτ). Check additivity on your five highest-value pairs.

**Deliverable:** the sensitivity table, with Q/K measured at both short and maximum context (the [Module 07 §2](Lecture-07.md) prediction).

### Phase 3 — Solve

Run the [Module 11](Lecture-11.md) solver with your measured values. Enumerate all `3^n` to confirm greedy optimality. Sweep `ε` and plot `B_token(ε)`; mark the knee.

**Deliverable:** allocation, solver trace in GB/nat, `B_token(ε)` curve.

### Phase 4 — Build

Emit the checkpoint with the manifest from [Module 05 §5](Lecture-05.md). **Verify the built artifact's actual `B_token` matches the solver's prediction** — a toolkit that silently ignores a per-group format request is common and will invalidate everything downstream.

**Deliverable:** checkpoint, manifest, predicted-vs-actual `B_token`.

### Phase 5 — Verify

Run [Module 12's](Lecture-12.md) protocol. Locked clocks, interleaved, paired, ≥8,000 draft tokens per configuration, full reporting standard.

**Deliverable:** the ablation table, and the bug check (`measured ≫ predicted` ⇒ bug).

---

## 4. The required ablation grid

Not a search — a **grid designed so each row isolates one claim from the course**:

| # | Configuration | Isolates | Predicted from |
|---|---|---|---|
| 0 | shipped baseline | reference | — |
| 1 | + `lm_head` → FP8 | traffic ≠ size | [Mod 04](Lecture-04.md) |
| 2 | + `lm_head` → NVFP4 | format ladder | [Mod 02](Lecture-02.md) |
| 3 | Q, K → FP8 (from NVFP4) | acceptance recovery | [Mod 07](Lecture-07.md), [10](Lecture-10.md) |
| 4 | embeddings → FP8 | **must be ~0 tok/s** (control) | [Mod 01](Lecture-01.md) |
| 5 | vision tower evicted | **must be ~0 tok/s, −0.86 GiB** (control) | [Mod 01](Lecture-01.md) |
| 6 | solver allocation | the thesis | [Mod 11](Lecture-11.md) |
| 7 | row 6 at 262 K context | long-context inversion | [Mod 09](Lecture-09.md) |
| 8 | `K` sweep at fixed quantization | `K*` re-tuning | [Mod 10](Lecture-10.md) |

**Rows 4 and 5 are the most important rows in the grid.** They are negative controls: the course *predicts* they produce zero throughput change. If they do not, your measurement apparatus is broken and every other row is suspect. A grid without controls is a demo.

For each row report: `B_token`, tok/s, predicted tok/s, τ, α, mean KL, p99 KL, top-1 agreement, and the **acceptance tax** from [Module 10](Lecture-10.md).

---

## 5. Success criteria

**Minimum (the course worked):**

- [ ] Ledger reproduces resident bytes to within 1 % and predicts non-speculative tok/s within 10 %.
- [ ] Negative controls (rows 4, 5) show no throughput change — proving the apparatus.
- [ ] Solver allocation achieves **≤ the shipped `B_token` at strictly lower KL** ([Module 11](Lecture-11.md)'s result, reproduced on your hardware).
- [ ] Every claim meets the [Module 12](Lecture-12.md) reporting standard.

**Target (the artifact is portfolio-grade):**

- [ ] **≥ 224 tok/s** (Stage 2) at acceptance **≥ 2.886** and mean KL **≤ 0.05**.
- [ ] The `B_token(ε)` curve with the knee identified and a defended `ε`.
- [ ] Long-context row at 262 K, with the feasibility analysis from [Module 09](Lecture-09.md).
- [ ] At least one **documented negative result** with its mechanism explained.

**Stretch (a real contribution):**

- [ ] **≥ 273 tok/s** (Stage 3) — requires fixing the drafter path.
- [ ] The drafter-efficiency finding confirmed and fixed: `c` from 0.171 toward 0.054.
- [ ] Q/K sensitivity-vs-context curve, testing [Module 07 §2](Lecture-07.md)'s RoPE prediction.
- [ ] TurboQuant runs end-to-end on a **second, different model** with no code changes.

That last one is what separates a case study from a tool.

---

## 6. Deliverables

```text
   turboquant/
   ├── README.md                 the finding, up front, in three sentences
   ├── ledger.py                 [1]  Module 04
   ├── probe.py                  [2]  Modules 06, 07, 08
   ├── solver.py                 [3]  Module 11
   ├── builder.py                [4]  Module 05
   ├── verify.py                 [5]  Modules 08, 12
   ├── configs/                  one YAML per ablation row (Module 12 §6 standard)
   ├── results/
   │   ├── ledger.md             traffic classes, B_token, gap
   │   ├── sensitivity.md        per-group ΔKL/Δτ, additivity check
   │   ├── ablation.csv          the grid, raw
   │   ├── b_token_vs_eps.png    the ε curve with the knee marked
   │   └── traces/               Nsight reports for the top kernels
   └── REPORT.md                 hypotheses, predictions, measurements, negatives
```

`REPORT.md` is the artifact that gets read. Structure it as:

```text
   1. The claim, in one sentence, with the number.
   2. The ledger: where the bytes are and where the headroom was.
   3. The allocation and why the solver chose it (the GB/nat trace).
   4. The grid, including the negative controls.
   5. What did NOT work, and the mechanism.
   6. The next hypothesis.
```

---

## 7. On negative results

The most valuable finding in the case study is a negative one:

```text
   Quantizing Q/K:  +1.9 % throughput,  −8.8 % acceptance,
                    +0.082 TV distance,  84 % acceptance tax.
                    ⟹ DO NOT SHIP.
```

Nobody publishes that, which is exactly why so many people rediscover it the expensive way. Your report should contain at least one such result stated as plainly.

The same goes for the two headroom findings this course produced by arithmetic alone — **the 72 %-vs-92 % bandwidth gap** and **the 3.2×-too-expensive draft step**. Neither is a quantization result. Both are worth more throughput than any quantization decision available on this model.

> That is the real lesson of the course: **the byte ledger is a better instrument than the intuition it replaces**, and it earns its keep most often by telling you *not* to quantize.

---

## 8. Exit criteria

You have completed this course when you can take an unfamiliar checkpoint on unfamiliar silicon and, in a single afternoon:

1. Produce its byte ledger and predict the batch-1 decode ceiling within 10 %.
2. Say whether it is bandwidth-bound, and refuse to quantize if it is not.
3. Name the three highest-value tensors, citing traffic and sensitivity separately.
4. Name the formats with a native path on that silicon, and the break-even threshold for anything else.
5. State the behavior budget in KL and acceptance terms **before** running the experiment.
6. Solve the allocation, build it, and verify the result against the prediction.
7. Detect the case where your speedup came from a bug.

If you can recite quantization algorithms but cannot say which bytes a token reads, you have vocabulary. **The point of this course is the allocation decision — and the discipline to prove it.**

---

## Current as of

* **Timeless:** the five-stage architecture, the build order, the ablation-with-controls design, the exit criteria.
* **Case-study pins:** 155.75 tok/s @ τ 2.886; derived `1 + K·c = 1.512`, `c = 0.171` at `K = 3` versus `c_bandwidth = 0.054`; the staged ladder to ~280 tok/s. The ladder's stage factors are **predictions from the course's models, not measurements** — verifying or refuting them is the capstone.
* **Related:** [AI Inference Engineer 2026 — Part 4](../AI%20Inference%20Engineer%202026/Part%204%20-%20Optimizing%20a%20Real%20Engine/README.md) runs the same discipline over ~96 PRs on an 8× H200 engine, and is the best companion read for this capstone.

---

*Course complete. [← Back to the course index](README.md) · [Phase 5 — ML Systems Engineering Guide](../Guide.md)*
