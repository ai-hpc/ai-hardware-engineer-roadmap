# Module 02 — Quantization Mathematics

**Collection:** [Hardware-Aware LLM Quantization](README.md) | **Previous:** [← Module 01](Lecture-01.md) | **Next:** [Module 03 →](Lecture-03.md)

---

[Module 01](Lecture-01.md) established *why* fewer bytes means more tokens per second. This module is about what actually happens to a number when you take its bits away — the formats, the scaling machinery that makes 4 bits usable at all, and the exact arithmetic of one activation landing on the NVFP4 grid.

The central fact of low-precision inference: **a 4-bit float has a dynamic range of 12×.** Not 12 orders of magnitude — a factor of twelve. Nothing in a neural network fits in that range. Everything that makes FP4 work is the scaling machinery wrapped around it.

---

## Learning objectives

By the end of this module you should be able to:

1. Read any floating-point format from its `ExMy` name and enumerate its representable values.
2. Explain the range-vs-precision trade encoded in the exponent/mantissa split, and why BF16 beat FP16 for training.
3. State NVFP4's full encoding — E2M1 + 16-element blocks + E4M3 block scale + FP32 global scale — and compute its **effective 4.5 bits per weight**.
4. Quantize and dequantize a block **by hand**, and predict which values in it get annihilated.
5. Explain why NVFP4's E4M3 scale beats MXFP4's E8M0 scale, and what block size buys.
6. Distinguish W4A16 / W8A8 / W4A4 and say which one the hardware actually rewards.

---

## 1. What a floating-point format is

Every format here splits its bits three ways:

```text
   ┌───┬─────────────┬──────────────┐
   │ S │  exponent   │   mantissa   │
   └───┴─────────────┴──────────────┘
     1       E bits       M bits

   value = (−1)^S × (1 + mantissa/2^M) × 2^(exp − bias)        [normal]
   value = (−1)^S × (mantissa/2^M)     × 2^(1 − bias)          [subnormal, exp = 0]

   bias = 2^(E−1) − 1
```

The split is the entire design decision:

* **exponent bits → dynamic range** (how far apart the largest and smallest representable magnitudes are)
* **mantissa bits → relative precision** (how finely spaced values are within one binade)

| Format | Bits | E | M | Max finite | Min normal | Dynamic range | Relative step |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP32 | 32 | 8 | 23 | 3.4e38 | 1.2e−38 | ~2e76 | 1.2e−7 |
| **BF16** | 16 | 8 | 7 | 3.4e38 | 1.2e−38 | ~2e76 | 7.8e−3 |
| FP16 | 16 | 5 | 10 | 65504 | 6.1e−5 | ~1e9 | 9.8e−4 |
| **FP8 E4M3** | 8 | 4 | 3 | 448 | 1.6e−2 | ~2.9e4 | 6.3e−2 |
| FP8 E5M2 | 8 | 5 | 2 | 57344 | 6.1e−5 | ~9.4e8 | 1.3e−1 |
| **FP4 E2M1** | 4 | 2 | 1 | **6** | **1.0** | **12** | **2.5e−1** |

Two rows explain a decade of practice:

**BF16 vs FP16.** Same 16 bits, opposite choices. BF16 keeps FP32's 8 exponent bits and spends only 7 on mantissa — same range as FP32, worse precision. FP16 has 3.5× better relative precision but overflows at 65504. Training gradients span enormous dynamic range and care little about the 4th significant digit, so BF16 won and loss scaling largely became unnecessary. **Range beat precision.**

**E2M1.** Look at the last row again. Max finite value 6, min normal 1.0. That is the entire representable universe of a 4-bit float.

---

## 2. E2M1 in full

Four bits, sixteen codes, eight distinct magnitudes. There are no infinities and no NaN — every code is a finite number:

```text
   exp=00 (subnormal, ×2⁰)  :  0.0,  0.5
   exp=01 (×2⁰)             :  1.0,  1.5
   exp=10 (×2¹)             :  2.0,  3.0
   exp=11 (×2²)             :  4.0,  6.0

   full grid:  ±{ 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0 }
```

Plotted, the non-uniformity is the point:

```text
   0    0.5   1.0  1.5   2.0        3.0        4.0             6.0
   ├─────┼─────┼────┼─────┼──────────┼──────────┼───────────────┤
     0.5   0.5  0.5   0.5     1.0        1.0          2.0
                        ← gap size grows with magnitude →
```

Gaps grow proportionally to magnitude — that is exactly what a float is *for*. The relative step stays near 25 % across the range, whereas an INT4 grid (uniform spacing) would give fine absolute resolution near the maximum and catastrophic relative error near zero. **For weight and activation distributions, which are roughly bell-shaped and centered on zero, the float grid's constant relative error is the better match.** This is the core reason FP4 outperforms INT4 at equal bit-width, and it is why the industry moved to microscaling float formats rather than pushing integer quantization lower.

---

## 3. Block scaling — the machinery that makes 4 bits work

A dynamic range of 12 is useless on its own. The fix is to store a **shared scale** alongside a small group of values, so each group gets its own window into the number line:

```text
   real value  ≈  scale_block  ×  code_E2M1        (code ∈ ±{0,0.5,1,1.5,2,3,4,6})
```

The design space is one parameter — **block size** — and it is a pure trade:

```text
   small blocks  →  each scale fits a tighter local range  →  less error
                 →  more scales stored                     →  more bits/element

   large blocks  →  fewer scales                           →  fewer bits/element
                 →  one outlier ruins a wider group        →  more error
```

Effective bits per element is exact arithmetic:

```text
   bits/elem  =  4  +  (bits_per_scale / block_size)
```

| Format | Block | Scale type | Scale bits | Effective bits/elem |
|---|---:|---|---:|---:|
| **NVFP4** | 16 | FP8 **E4M3** | 8 | **4 + 8/16 = 4.50** |
| **MXFP4** (OCP) | 32 | **E8M0** (power-of-two) | 8 | 4 + 8/32 = 4.25 |

> **NVFP4 is 4.5 bits per weight, not 4.** Every capacity and bandwidth calculation in this course uses **0.5625 bytes/element**. Using 0.5 understates your checkpoint by 12.5 % — and that error propagates straight into the byte ledger from [Module 01](Lecture-01.md).

### Why E4M3 scales beat E8M0 scales

This is the sharpest distinction between the two formats, and it is not about block size.

**E8M0** is 8 exponent bits and **zero mantissa bits** — it can only represent powers of two. Your block scale must be rounded to `2^k`. Suppose a block's ideal scale is 3.95: E8M0 must choose between 2.0 and 4.0.

**E4M3** has 3 mantissa bits, so it represents 3.75, 4.0, 4.25… — it can land near 3.95 with ~3 % error instead of up to 41 %.

```text
   ideal block scale = 3.95

   E8M0 (MXFP4)  →  must pick 4.0 (or 2.0)  →  quantization grid is coarse-stepped
   E4M3 (NVFP4)  →  picks 4.0 exactly here, and 3.75 / 4.25 elsewhere
```

A badly-rounded scale wastes representable codes: if the scale is too large the block's values crowd into the low codes; too small and they clip. NVFP4 pays 0.25 extra bits per element for a scale that lands accurately, plus a 2× smaller block. **It trades a little capacity for materially lower error, which is usually the right side of the trade at 4 bits.**

### The two-level scale

NVFP4 in practice is *two* scales:

```text
   real ≈  global_scale_FP32  ×  block_scale_E4M3  ×  code_E2M1
           └── per tensor ──┘   └── per 16 elems ─┘  └─ per elem ─┘
```

The FP32 global scale exists because E4M3 itself tops out at 448. If a tensor's block scales would exceed that, the global scale pre-normalizes the whole tensor into range. It costs 4 bytes per tensor — numerically free — and it is why NVFP4 handles tensors with wildly varying block magnitudes.

---

## 4. One block, by hand

This is the exercise that makes the format concrete. Take a 16-element activation block whose absolute maximum is **23.7**, containing among others the values `0.9`, `4.1`, and `−11.2`.

**Step 1 — choose the block scale.** Map absmax onto the largest E2M1 code (6.0):

```text
   s_raw = 23.7 / 6 = 3.95
```

**Step 2 — round the scale to E4M3.** Near 3.95 the representable E4M3 values step by 0.25 (`2.0, 2.25, … 3.75, 4.0`):

```text
   s = 4.0
```

**Step 3 — quantize each element** (`x/s` → nearest E2M1 code → `× s`):

| x | x / s | nearest code | reconstructed | abs error | rel error |
|---:|---:|---:|---:|---:|---:|
| 23.70 | 5.9250 | 6.0 | 24.000 | +0.300 | 1.27 % |
| 4.10 | 1.0250 | 1.0 | 4.000 | −0.100 | 2.44 % |
| −11.20 | −2.8000 | −3.0 | −12.000 | −0.800 | 7.14 % |
| **0.90** | **0.2250** | **0.0** | **0.000** | **−0.900** | **100 %** |

Look at the last row. `0.225` is nearer to `0` than to `0.5`, so **the value is annihilated — it becomes exactly zero.**

```text
   block scale set by ONE large value (23.7)
                    │
                    ▼
   ├──────────────────────────────────────────────┤   representable window
   0                                            24.0
   ▲        ▲
   │        └─ smallest nonzero code = 0.5 × 4.0 = 2.0
   │
   └─ everything below 1.0 in real units rounds to ZERO

   0.9 is not "slightly wrong". It is GONE.
```

**This single fact drives Modules 05, 06, and 07.** A block's scale is hostage to its largest element. One outlier in a group of 16 destroys the resolution of the other 15. Every technique you will meet — clipping, AWQ's scale migration, SmoothQuant's outlier transfer, keeping sensitive layers wide — is an answer to this one behaviour.

### The clipping trade-off, quantified

The obvious fix is to stop letting the outlier set the scale. Clip absmax to 12.0 instead of 23.7 (so `s = 2.0`) and re-run the same block:

| x | unclipped (s = 4.0) | **clipped (s = 2.0)** |
|---:|---:|---:|
| 23.70 | 24.000 (1.27 %) | **12.000 (49.37 %)** ← clipped hard |
| 4.10 | 4.000 (2.44 %) | 4.000 (2.44 %) |
| −11.20 | −12.000 (7.14 %) | −12.000 (7.14 %) |
| **0.90** | **0.000 (100 %)** | **1.000 (11.11 %)** ← rescued |

```text
   NO CLIPPING          :  outliers exact,  bulk annihilated
   AGGRESSIVE CLIPPING  :  outliers mangled, bulk preserved

   the optimum is in between, and it depends on which error the
   NEXT operator amplifies  ──────────▶  Modules 05 and 07
```

There is no universally right answer here — only an error metric and a search. That search *is* calibration ([Module 05](Lecture-05.md)).

---

## 5. Quantization error, formally

For a scalar `x` quantized to `Q(x)`, the error is `e = Q(x) − x`, and it decomposes into two mechanisms with completely different behavior:

```text
   e_total  =  e_rounding   +   e_clipping
               │                 │
               │                 └── x fell outside [−s·6, +s·6]; grows without bound
               └── x landed between two codes; bounded by half the local gap
```

For a uniform quantizer with step `Δ`, rounding error is approximately uniform on `[−Δ/2, +Δ/2]`, giving the classic result:

```text
   σ²_round  =  Δ² / 12
```

which yields the familiar **~6.02 dB of SNR per bit**. E2M1 is not uniform, so the useful statement is in relative terms: with 1 mantissa bit, the worst-case relative rounding error inside a binade is 25 %, and the RMS relative error across a well-scaled block lands around 8–12 % — *per element*.

That sounds fatal. It is not, and the reason is the next section.

### Why 10 % element-wise error does not mean a 10 % wrong model

A dot product of length `K` sums `K` independent-ish errors. If per-element errors are zero-mean with standard deviation `σ`, the **sum's** error grows as `√K` while the **signal** grows as `K`:

```text
   relative error of the dot product  ≈  σ / √K
```

With `K = 8192` and `σ = 0.10`:

```text
   0.10 / √8192  ≈  0.0011      →  ~0.11 % error on the output activation
```

**Error averaging is the entire reason low-bit inference works.** Two corollaries you will use constantly:

1. **Zero-mean matters more than small.** A quantizer with a systematic bias does not average out — the bias accumulates linearly with `K`, not as `√K`. This is why stochastic rounding and bias correction appear in good quantization pipelines.
2. **Operators that break the averaging assumption are dangerous.** Softmax does not average errors; it exponentiates them. That is [Module 07](Lecture-07.md)'s subject and the reason Q/K behave unlike every other tensor.

---

## 6. Where the bits go: W, A, and KV

"4-bit model" is ambiguous. Three tensors can each be quantized independently:

```text
                weights (W)          activations (A)         KV cache
   W4A16        4-bit  ◀── dequant to BF16 ──▶  16-bit       16-bit
   W8A8         8-bit                            8-bit        8-bit
   W4A4         4-bit                            4-bit       4/8-bit
```

| Scheme | Weight traffic | Tensor-core input | Decode benefit | Difficulty |
|---|---|---|---|---|
| **W4A16** | 4× less | BF16 (weights upconverted) | **most of it** — decode is weight-bound | easy; the default |
| **W8A8** | 2× less | FP8 native | moderate | moderate |
| **W4A4** | 4× less | **FP4 native** | most + compute | **hard** — activations have outliers |

The key insight for batch-1 decode, straight from [Module 01](Lecture-01.md): **weights dominate `B_token`, activations are a rounding error.** A single token's activations are a few hundred KB; the weights are ~16 GB. So:

> At batch 1, **W4A16 captures nearly all of the available throughput win.** W4A4's extra benefit is compute-side, which you are 250× away from needing.

W4A4 becomes worth its difficulty when you are batching hard enough to approach the ridge point, or when the activation-side memory traffic in a fused kernel starts to matter. On a single-user RTX 5090, **quantizing activations to FP4 buys little and risks a lot** — a conclusion that falls directly out of the roofline, not out of any experiment.

This also explains why activation quantization gets its own module ([06](Lecture-06.md)): it is the part with a bad reward/risk ratio at batch 1, so you need to know precisely when it pays.

---

## 7. Format cheat sheet

Storage for one `8192 × 8192` weight matrix (67.1 M elements):

| Format | bytes/elem | Matrix size | vs BF16 |
|---|---:|---:|---:|
| BF16 | 2.0 | 134.2 MB | 1.00× |
| FP8 E4M3 | 1.0 | 67.1 MB | 2.00× |
| **NVFP4** | **0.5625** | **37.7 MB** | **3.56×** |
| MXFP4 | 0.5312 | 35.7 MB | 3.76× |
| INT4 (group 128, FP16 scale) | 0.5156 | 34.6 MB | 3.88× |

Note that NVFP4 is **not** the smallest — it is the one with the best error-per-byte on Blackwell *and* a native execution path. That second clause is what [Module 03](Lecture-03.md) is about, and it is what makes the marginally-smaller alternatives lose.

---

## Checkpoint

You should now be able to:

1. Enumerate all 8 E2M1 magnitudes from memory and state the format's dynamic range (12×).
2. Compute NVFP4's effective bits/element (4.5) and explain both terms.
3. Predict, given a block's absmax, which of its elements will quantize to zero.
4. Explain why MXFP4's E8M0 scale is numerically worse than NVFP4's E4M3 despite costing fewer bits.
5. Explain why ~10 % per-element error yields ~0.1 % output error, and name the operator that breaks the argument.
6. Justify choosing W4A16 over W4A4 for single-user decode using a roofline argument alone.

---

## Ship it

Implement `quantize_nvfp4(tensor)` and `dequantize_nvfp4(...)` in NumPy or PyTorch — reference semantics, no CUDA:

```text
   1.  reshape to [..., n_blocks, 16]
   2.  global_scale = amax(tensor) / (6 × 448)          # keep block scales inside E4M3
   3.  block_scale  = amax(block) / 6 / global_scale  →  round to E4M3
   4.  codes        = round_to_nearest_E2M1(block / (block_scale × global_scale))
   5.  dequant      = codes × block_scale × global_scale
```

Then report, for one real weight tensor: RMS relative error, **fraction of elements annihilated to zero**, and the error histogram. Repeat with block size 32 and with an E8M0 scale to reproduce the NVFP4-vs-MXFP4 gap yourself. That table is your artifact.

---

## Current as of

* **Timeless:** IEEE-style float decomposition, block scaling arithmetic, `Δ²/12`, the `σ/√K` averaging argument, the clipping/rounding trade.
* **2026 format pins:** NVFP4 = E2M1 + block 16 + E4M3 block scale + FP32 per-tensor scale. MXFP4 (OCP Microscaling) = E2M1 + block 32 + E8M0 scale. Both are current; the OCP MX specification is the reference for the latter.
* **Watch:** microscaling variants continue to appear (6-bit MXFP6, mixed block sizes). The `4 + scale_bits/block_size` arithmetic generalizes to all of them.

---

**Next:** [Module 03 — Blackwell Hardware →](Lecture-03.md)
