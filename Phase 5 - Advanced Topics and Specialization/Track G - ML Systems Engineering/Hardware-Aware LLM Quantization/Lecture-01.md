# Module 01 — Inference Physics

**Collection:** [Hardware-Aware LLM Quantization](README.md) | **Previous:** [← Course index](README.md) | **Next:** [Module 02 →](Lecture-02.md)

---

This is the most important module in the course, and it contains no quantization algorithms at all.

Before asking *"what quantization method should I use?"* you must answer *"what resource is limiting this workload?"* Quantizing a compute-bound workload and quantizing a bandwidth-bound workload are **different optimization problems with different correct answers**. Get this backwards and every downstream decision inherits the error.

---

## Learning objectives

By the end of this module you should be able to:

1. Compute **arithmetic intensity** for a decode step and place it on a roofline.
2. Derive the batch-1 decode ceiling `tok/s ≈ BW_eff / B_token` and use it to predict throughput within ~10 %.
3. Distinguish **resident bytes** from **per-token-read bytes**, and explain why conflating them overstates bandwidth utilization.
4. Apply the `Opportunity = Traffic × Compressibility × HardwareSpeedup × BehaviorTolerance` framework to rank quantization targets.
5. Explain why "minimize bits" and "maximize tok/s" are both wrong objectives.

---

## 1. One token through an LLM

Autoregressive generation is a loop, and the loop body is the entire model:

```text
   token N ─▶ embedding ─▶ layer 0 ─▶ layer 1 ─▶ ... ─▶ layer L-1 ─▶ final norm
                                                                        │
                              token N+1 ◀── sample ◀── logits ◀── lm_head
                                    │
                                    └────────── repeat ──────────┐
                                                                  ▼
```

For **every single generated token**, the GPU must apply billions of parameters. At batch size 1 and concurrency 1, those weights are not reused across simultaneous tokens — you load a weight, you use it once, you throw it away, and next token you load it again.

That produces the defining condition of local LLM decode:

> **The GPU spends more time transporting weights than doing arithmetic on them.**

Prefill is the opposite. Prefill processes hundreds or thousands of tokens through the same weight-load, so weights get reused and the workload becomes compute-bound. **Prefill and decode live on opposite sides of the roofline, in the same model, in the same request.** Almost every mistake in this field comes from applying a decode intuition to prefill or vice versa.

---

## 2. Arithmetic intensity

The tool that formalizes this is **arithmetic intensity** — operations performed per byte moved:

```text
              FLOPs performed
   AI  =  ───────────────────────        [FLOP / byte]
              bytes transferred
```

Two regimes follow, separated by the machine's **ridge point** (`peak FLOP/s ÷ peak bytes/s`):

```text
   AI < ridge  →  MEMORY-BOUND        T ≈ bytes_moved / bandwidth
                  more tensor-core throughput does ~nothing
                  the fix is: MOVE FEWER BYTES

   AI > ridge  →  COMPUTE-BOUND       T ≈ FLOPs / (FLOP/s)
                  more bandwidth does ~nothing
                  the fix is: DO LESS MATH, or use faster math units
```

Now compute AI for a batch-1 decode GEMV. A weight matrix of `N×K` elements does `2·N·K` FLOPs (one multiply, one add per element) and moves `N·K·bytes_per_weight` bytes:

```text
                2 · N · K                 2
   AI_decode = ─────────────────  =  ────────────
               N · K · bytes_w        bytes_w
```

The matrix dimensions cancel. **Arithmetic intensity at batch 1 depends only on the weight format:**

| Weight format | bytes/weight | AI (FLOP/byte) |
|---|---|---|
| BF16 | 2.0 | 1.0 |
| FP8 | 1.0 | 2.0 |
| NVFP4 | 0.5625 | 3.56 |

The RTX 5090's ridge point for BF16 tensor math is roughly `419 TFLOP/s ÷ 1792 GB/s ≈ 234 FLOP/byte`. Batch-1 decode sits at **1.0**.

```text
   FLOP/s
     ▲
     │                        ┌──────────────── compute-bound roof
     │                       ╱
     │                      ╱
     │                     ╱  ridge ≈ 234 FLOP/byte (BF16)
     │                    ╱
     │  ┌────────────────╱
     │  │  memory-bound slope = 1792 GB/s
     │  │
     └──┴──▲─────────────────────────────────────▶ AI (FLOP/byte)
           │
        AI = 1.0
     batch-1 decode is ~234× BELOW the ridge
```

More generally, at batch size `B` the intensity is `2B / bytes_w`, so the batch needed to reach the ridge is:

| Format | AI at batch B | B needed to reach ridge |
|---|---|---|
| BF16 | `B` | ~234 |
| NVFP4 | `3.56 · B` | ~263 |

**You would need a batch of roughly 250 concurrent sequences before decode stops being memory-bound.** Single-user local inference is not close, is not going to get close, and every optimization decision should follow from that.

---

## 3. The measurement — and a unit trap

Here is a real measurement from a ~27 B NVFP4 model on an RTX 5090:

```text
   Resident weights   :   18.80 GiB
   Decode throughput  :   81.6  tok/s
   RTX 5090 peak BW   :   1792  GB/s   (32 GB GDDR7, 512-bit @ 28 Gbps)
```

The obvious move is to multiply resident bytes by tokens per second. Do it carefully, because **GiB and GB differ by 7.4 % and this is where most published bandwidth numbers go wrong**:

```text
   18.80 GiB  ×  1024³ / 10⁹   =   20.19 GB          ← convert FIRST
   20.19 GB   ×  81.6 tok/s     =   1647 GB/s
   1647 / 1792                  =   91.9 %  of peak
```

Skip the conversion and you get `18.80 × 81.6 = 1534`, report "1.53 TB/s / 86 %", and understate the machine by 7.4 %. Bandwidth is quoted in **GB/s (decimal)**; `nvidia-smi` and most allocators report memory in **GiB (binary)**. Convert once, at the boundary, and label every number.

So the headline result: the decode path appears to run at **~92 % of theoretical peak memory bandwidth**. That is an extraordinarily strong signal:

> *"I do not primarily need more arithmetic throughput. I need fewer bytes per generated token."*

Hold that number. Section 5 is going to correct it, and the correction is the whole point of this course.

---

## 4. The fundamental equation

For a strongly bandwidth-bound decode workload:

```text
              BW_effective
   tok/s  ≈  ──────────────
                B_token
```

where `BW_effective` is achievable bandwidth (never the datasheet number — expect 85–93 % of peak on a well-written kernel) and `B_token` is bytes that **must** be fetched to produce one token.

Work an example. Suppose `BW_eff = 1650 GB/s` and `B_token = 20 GB`:

```text
   tok/s ≈ 1650 / 20   =  82.5
```

Now remove two gigabytes of per-token traffic:

```text
   tok/s ≈ 1650 / 18   =  91.7      →  +11 % throughput
```

You did not add a CUDA core. You did not write a faster GEMM. You stopped moving two unnecessary gigabytes, sixty times a second. **That is what hardware-aware quantization is.**

Rearranged, the equation is also a *design tool*. If you have a throughput target, it tells you your byte budget:

```text
   B_token_required  =  BW_effective / tok/s_target

   want 120 tok/s at 1650 GB/s   →   B_token must be ≤ 13.75 GB
```

That is a hard, checkable engineering constraint, and you can evaluate it against a checkpoint **before writing any code**.

---

## 5. Checkpoint size ≠ bytes per token

Here is the checkpoint inventory for the same model. About **6.91 GB (33.6 %) is still BF16**:

```text
   Embeddings      2.54 GB  BF16
   lm_head         2.54 GB  BF16
   Vision tower    0.92 GB  BF16
   MTP head        0.85 GB  BF16
   norms / misc    0.06 GB
   ───────────────────────
                   6.91 GB  BF16   (remaining 13.28 GB is NVFP4 transformer body)
```

The beginner's conclusion is "great — 6.91 GB of low-hanging fruit, quantize all of it." That conclusion is wrong for three of the four tensors, and the reason is that **these four have completely different runtime behavior.**

### Embeddings — 2.54 GB, read ~16 KB per token

An embedding lookup is a **gather**, not a matrix multiply:

```cpp
// you touch ONE row of a [vocab × d_model] table
const __nv_bfloat16* row = embedding + (size_t)token_id * d_model;
```

With `d_model = 8192` at BF16 that is `8192 × 2 = 16 KB` — about **0.0008 %** of the 2.54 GB table. Quantizing embeddings is a *capacity* optimization worth 1.3 GB of VRAM. Its effect on decode throughput is indistinguishable from zero.

### Vision tower — 0.92 GB, read 0 bytes per token

During text-only decode the vision encoder never runs. Quantize it, evict it, delete it — VRAM changes, `tok/s` does not.

### MTP head — 0.85 GB, conditional

If speculative decoding is off, it is not on the decode path at all. If it is on, it runs once per draft step and becomes some of the hottest memory in the system. **Its traffic is a function of your serving configuration, not of the checkpoint.** Module 10 handles this properly.

### lm_head — 2.54 GB, read in full, every token

Completely different. The output projection computes `logits = W_vocab · h` over the whole vocabulary. There is no gather; **every element of the matrix participates in every token**. It is 2.54 GB squarely on the per-token critical path — and it is still BF16 while the transformer body is already NVFP4.

That is the real target hiding in the inventory, and it is *not* the biggest tensor group.

### Correcting the utilization number

Now redo Section 3's arithmetic with only the bytes the decoder actually reads:

```text
   resident                        20.19 GB
   − embeddings (gathered)         − 2.54
   − vision tower (inactive)       − 0.92
   − MTP head (spec. disabled)     − 0.85
   ─────────────────────────────────────────
   B_token                        ≈ 15.88 GB
```

```text
   real traffic  =  15.88 GB × 81.6 tok/s  =  1296 GB/s   =  72.3 % of peak
```

**The 92 % figure was an overestimate.** It charged the memory system for 4.3 GB of bytes that are never fetched during text decode. Actual bandwidth utilization is closer to **72 %**, and that changes the engineering conclusion completely:

```text
   at 92 % utilization  →  the byte count is the only lever; you are at the wall
   at 72 % utilization  →  ~20 points of headroom exist that are NOT a bytes problem
                            (kernel efficiency, launch gaps, tail effects, scheduling)

   throughput if the SAME 15.88 GB were moved at 92 % of peak (1650 GB/s):
        1650 / 15.88  =  103.9 tok/s     ← vs. 81.6 measured
```

There are roughly **22 tok/s available before you remove a single additional bit.** A team that believed the 92 % number would have spent that quarter on quantization and found nothing, because the bytes were never the binding constraint at that operating point.

> **This is the single most valuable habit in the course:** never compute bandwidth utilization from checkpoint size. Compute it from a per-token byte ledger. [Module 04](Lecture-04.md) builds that ledger properly.

---

## 6. Hardware-aware quantization vs. ordinary quantization

A naive compression strategy asks **"which tensor is large?"** A better one asks **"which tensor produces the most runtime memory traffic?"** The best one scores four factors at once:

```text
   Opportunity_i  =  Traffic_i  ×  Compressibility_i  ×  HardwareSpeedup_i  ×  BehaviorTolerance_i
```

| Factor | Question | Failure if ignored |
|---|---|---|
| **Traffic** | bytes moved per generated token | you quantize a vision tower for 0 tok/s |
| **Compressibility** | how many bits can this tensor actually give up? | you force FP4 on a tensor that needed FP8 |
| **HardwareSpeedup** | does this silicon *execute* the compressed form natively? | you ship 3-bit and get a dequant kernel that is slower than 4-bit |
| **BehaviorTolerance** | how much error can this tensor absorb? | you quantize Q/K and lose a quarter of your acceptance rate |

The third factor is the one people skip, and on Blackwell it is decisive:

```text
   4 bits (NVFP4)  →  fewer bytes  +  NATIVE block-scaled tensor-core path
   3 bits          →  fewer bytes  +  no native path → unpack to a wider type first
   2 bits          →  fewest bytes +  unpack overhead can exceed the bandwidth saved
```

> **Lowest bit-width does not mean fastest inference.** A format is only fast if the tensor cores can consume it without an unpacking detour. [Module 03](Lecture-03.md) makes this precise for `sm_120`.

---

## 7. Accuracy is not one variable either

Consider two candidate profiles:

```text
   Profile A :  148 tok/s,  acceptance 2.79
   Profile B :  151 tok/s,  acceptance 2.55
```

B is faster on the headline metric. Would you ship it? Almost certainly not — B degraded the model's agreement with its own drafter by 9 %, which means the target distribution moved. It bought 2 % throughput with a behavioral regression.

This is not hypothetical. Measured on the case-study model:

```text
   MLP + O quantized          147.87 tok/s     acceptance 2.792
   MLP + O + QKV quantized    150.73 tok/s     acceptance 2.546
                              ──────────       ─────────────────
                              +1.9 % speed     −8.8 % acceptance
```

Adding QKV to the quantization set bought **1.9 % throughput** and cost **8.8 % acceptance**. That is a bad trade, and it is invisible if your only metric is tok/s.

So the objective is never `min(bits)` and never `max(tok/s)`. It is a **constrained optimization**:

```text
   maximize    tok/s

   subject to  BehaviorLoss  <  ε           (KL vs. reference, acceptance length)
               VRAM          <  32 GB
               context       =  262 144 tokens
               format ∈ {hardware-native formats on sm_120}
```

[Module 11](Lecture-11.md) solves this problem explicitly. Everything between here and there is about measuring its terms.

---

## 8. The optimization hierarchy

Attach a behavior profile to every tensor group, not a size:

```text
                         Large?   Read every token?   Speed target?
   ────────────────────────────────────────────────────────────────
   Embeddings             YES           NO                 NO
   Vision tower           YES           NO                 NO
   MTP head               YES        conditional        conditional
   lm_head                YES           YES                YES
   MLP weights            HUGE          YES                YES
   Attention weights      YES           YES           YES (but see Mod. 07)
   KV cache             grows           YES          context-dependent
```

Which gives a priority order that **changes with context length**:

```text
   short context                      very long context (262 K)
   ─────────────                      ─────────────────────────
   1. MLP weights                     1. KV cache traffic
   2. lm_head                         2. MLP weights
   3. attention weights               3. lm_head
   4. (KV cache — small)              4. attention weights
```

At 262 K tokens the KV cache stops being a footnote and becomes a co-dominant traffic source. That is why long-context targets introduce an optimization problem that simply does not appear in short-context benchmarks — [Module 09](Lecture-09.md).

---

## 9. Exercise — rank these before writing any code

| Tensor | Size | Read every token? | Current precision |
|---|---:|---|---|
| Embeddings | 3 GB | No (gather) | BF16 |
| MLP | 10 GB | Yes | FP4 |
| lm_head | 2.5 GB | Yes | BF16 |
| Vision | 1 GB | No | BF16 |
| QKV | 1 GB | Yes | FP4 |

**Bad target — `Vision → FP4`.** Large storage win, zero text-decode win. `Traffic = 0` zeroes the whole opportunity product.

**Best target — `lm_head BF16 → FP8`.** Modest checkpoint reduction (2.5 → 1.25 GB) but it is 2.5 GB of *per-token* traffic, and FP8 has a native path. Expected effect using the equation from §4, with `B_token = 13.5 GB`:

```text
   before:  1650 / 13.5   =  122.2 tok/s
   after :  1650 / 12.25  =  134.7 tok/s      →  +10.2 %
```

**Dangerous target — `QKV → lower precision`.** Only 1 GB of traffic (7 % of the ledger), so the ceiling on the win is ~7 %, while [Module 07](Lecture-07.md) shows the behavioral cost is disproportionate. Low reward, high risk — the worst quadrant.

Notice the ranking is driven almost entirely by the **Traffic** and **BehaviorTolerance** terms, and not at all by tensor size. Embeddings are the largest BF16 tensor in the case study and rank last.

---

## 10. The mental model

Stop seeing this:

```text
   Qwen3.8-27B  →  27 B parameters
```

Start seeing this:

```text
   Qwen3.8-27B
   │
   ├── read EVERY token  ────────────────  the decode critical path
   │   ├── MLP weights
   │   ├── attention weights
   │   └── lm_head
   │
   ├── read SPARSELY  ───────────────────  capacity cost only
   │   └── embeddings (one row per token)
   │
   ├── CONDITIONAL  ─────────────────────  traffic depends on serving config
   │   ├── vision tower (multimodal requests only)
   │   └── MTP head (speculation enabled only)
   │
   └── GROWS WITH CONTEXT  ──────────────  the long-context term
       └── KV cache
```

and attach to every node:

```text
   bytes  ·  precision  ·  bytes/token  ·  kernel path  ·  quant error  ·  behavior sensitivity
```

That annotated tree — not the parameter count — is the object you optimize.

---

## Checkpoint

You should now be able to explain why all four of these are simultaneously true:

1. A **3-bit model can be smaller but slower** than an NVFP4 model. *(No native tensor-core path → unpack overhead exceeds the bandwidth saved.)*
2. Removing a **1 GB vision tower saves 1 GB of VRAM and 0 tok/s**. *(Traffic = 0 during text decode.)*
3. Quantizing a **2.5 GB `lm_head` matters more for decode than quantizing a larger embedding table**. *(Full participation vs. single-row gather — a ~150,000× difference in per-token bytes.)*
4. Quantizing **Q/K yields a tiny speed win but disproportionate behavioral damage**. *(Small traffic share; softmax amplifies logit error — Module 07.)*

And one more, which is the module's real lesson:

5. **A 92 %-of-peak bandwidth number computed from checkpoint size can hide 20 points of non-bandwidth headroom.** Ledger the bytes, then decide.

---

## Ship it

Before continuing, produce a one-page **byte ledger** for a model you actually run:

- resident bytes (converted to GB, labelled)
- per-token-read bytes, itemized by tensor group
- measured tok/s
- implied `BW_effective` and its percentage of datasheet peak
- predicted tok/s if the same bytes moved at 90 % of peak

If the gap between measured and predicted is large, **your next task is not quantization.** That is the module working.

---

## Current as of

* **Timeless:** arithmetic intensity, roofline reasoning, `tok/s ≈ BW_eff / B_token`, the resident-vs-per-token distinction, the opportunity framework.
* **2026 hardware pin:** RTX 5090 (GB202, `sm_120`) at 1792 GB/s / 32 GB GDDR7; BF16 ridge ≈ 234 FLOP/byte. Re-derive both for any other part — [Module 03](Lecture-03.md).
* **Case-study measurements** (18.80 GiB resident, 81.6 tok/s, 155.75 tok/s @ 2.886 acceptance) are from the published NVFP4 and DSpark builds linked in the [course index](README.md).

---

**Next:** [Module 02 — Quantization Mathematics →](Lecture-02.md)
