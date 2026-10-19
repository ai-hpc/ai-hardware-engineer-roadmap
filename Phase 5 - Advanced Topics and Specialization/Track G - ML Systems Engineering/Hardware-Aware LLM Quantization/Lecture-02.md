# Module 02 — Quantization Mathematics

**Collection:** [Hardware-Aware LLM Quantization](README.md) | **Previous:** [← Module 01](Lecture-01.md) | **Next:** [Module 03 →](Lecture-03.md)

---

[Module 01](Lecture-01.md) established the conditions under which fewer transferred bytes can improve token throughput. This module is about what actually happens to a number when you take its bits away — the formats, the scaling machinery that makes 4 bits usable at all, and the exact arithmetic of one activation landing on the NVFP4 grid.

The central fact of low-precision inference: **a 4-bit float has a dynamic range of 12×.** Not 12 orders of magnitude — a factor of twelve. Nothing in a neural network fits in that range. Everything that makes FP4 work is the scaling machinery wrapped around it.

---

## Learning objectives

By the end of this module you should be able to:

1. Read exponent/mantissa widths and consult the format specification for bias, special values and subnormals.
2. Explain the range-vs-precision trade encoded in the exponent/mantissa split, and why BF16 beat FP16 for training.
3. State NVFP4's full encoding — E2M1 + 16-element blocks + E4M3 block scale + FP32 global scale — and compute its **effective 4.5 bits per weight**.
4. Quantize and dequantize a block **by hand**, and predict which values in it get annihilated.
5. Compare scale precision, scale range and block size without assuming one format always wins.
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

The name alone is incomplete: bias, finite-only encodings, special values, rounding and subnormals also matter. The formula above uses the usual bias for the named formats; verify a new format rather than inferring all semantics from its name.

Two important choices are:

* **exponent bits → dynamic range** (how far apart the largest and smallest representable magnitudes are)
* **mantissa bits → relative precision** (how finely spaced values are within one binade)

| Format | Bits | E | M | Max finite | Min normal | Max/min normal | Relative spacing at 1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP32 | 32 | 8 | 23 | 3.4e38 | 1.2e−38 | ~2e76 | 1.2e−7 |
| **BF16** | 16 | 8 | 7 | 3.4e38 | 1.2e−38 | ~2e76 | 7.8e−3 |
| FP16 | 16 | 5 | 10 | 65504 | 6.1e−5 | ~1e9 | 9.8e−4 |
| **FP8 E4M3FN** | 8 | 4 | 3 | 448 | 1.6e−2 | ~2.9e4 | 1.25e−1 |
| FP8 E5M2 | 8 | 5 | 2 | 57344 | 6.1e−5 | ~9.4e8 | 2.5e−1 |
| **FP4 E2M1** | 4 | 2 | 1 | **6** | **1.0** | **6** | **5e−1** |

Two rows explain a decade of practice:

**BF16 vs FP16.** Same 16 bits, opposite choices. BF16 keeps FP32's 8 exponent bits and spends only 7 on mantissa — same range as FP32, worse precision. FP16 has 8× finer normal spacing but overflows at 65504. Training gradients span enormous dynamic range and care little about the 4th significant digit, so BF16 won and loss scaling largely became unnecessary. **Range beat precision.**

**E2M1.** Look at the last row again. Max finite value 6, min normal 1.0. There is also a nonzero subnormal 0.5: max/min-positive is 12, while max/min-normal is 6.

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

Normal floating-point gaps grow with magnitude, while subnormals have fixed spacing. Relative rounding error near zero can be 100%; it is not constant. An integer grid and a floating grid allocate their limited codes differently. Which yields lower weight, output or task error depends on the distribution, scale, clipping and kernel. FP4 does not universally outperform INT4 at equal bit-width.

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

Leading payload-plus-block-scale bits for complete blocks are:

```text
   bits/elem  =  4  +  (bits_per_scale / block_size)
```

| Format | Block | Scale type | Scale bits | Effective bits/elem |
|---|---:|---|---:|---:|
| **NVFP4** | 16 | FP8 **E4M3** | 8 | **4 + 8/16 = 4.50** |
| **MXFP4** (OCP) | 32 | **E8M0** (power-of-two) | 8 | 4 + 8/32 = 4.25 |

> **NVFP4 is 4.5 bits per weight, not 4.** The leading full-block cost is **0.5625 bytes/element**; add global scales, row padding, headers/alignment and retained wide tensors for an artifact. The 0.5625 value is 12.5% larger than 0.5 (and omitting scales understates this combined storage by 11.1%) — and that error propagates straight into the byte ledger from [Module 01](Lecture-01.md).

### Compare E4M3 and E8M0 scales

E8M0 represents powers of two over a wide exponent range. E4M3 has fractional precision but less range. Around 4 its values include 3.75, 4.0 and **4.5**, not 4.25. An ideal scale 3.95 rounds to 4 in both schemes, so that example does not demonstrate an E4M3 advantage. At 3.5, E4M3 is exact while the nearest power of two is 4.

NVFP4 also uses smaller blocks than MXFP4, so comparing them changes more than the scale grid. Measure the combined effect of block size, rounding, dynamic range and payload overhead; there is no universal error ordering.

### The two-level scale

NVFP4 in practice is *two* scales:

```text
   real ≈  global_scale_FP32  ×  block_scale_E4M3  ×  code_E2M1
           └── per tensor ──┘   └── per 16 elems ─┘  └─ per elem ─┘
```

The FP32 global scale exists because E4M3 itself tops out at 448. If a tensor's block scales would exceed that, the global scale pre-normalizes the whole tensor into range. It costs 4 bytes per tensor in this reference convention, negligible for large tensors but material for tiny ones. This is one reason NVFP4 handles tensors with wildly varying block magnitudes.

---

## 4. One block, by hand

For this hand calculation set the global scale to 1. The executable lab derives a tensor-global scale instead, so its exact reconstructed values can differ. Take a 16-element activation block whose absolute maximum is **23.7**, containing among others the values `0.9`, `4.1`, and `−11.2`.

**Step 1 — choose the block scale.** Map absmax onto the largest E2M1 code (6.0):

```text
   s_raw = 23.7 / 6 = 3.95
```

**Step 2 — round the scale to E4M3.** Near 3.95 the representable E4M3 values step by 0.25 (`2.0, 2.25, … 3.75, 4.0`):

```text
   s = 4.0
```

**Step 3 — quantize each element** (`x/s` → nearest E2M1 code → `× s`):

| x | x / s | nearest code | reconstructed | signed error | absolute relative error |
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

**This single fact drives Modules 05, 06, and 07.** A block's scale is hostage to its largest element. One outlier can reduce the resolution available to other values in its block. Every technique you will meet — clipping, AWQ's scale migration, SmoothQuant's outlier transfer, keeping sensitive layers wide — is an answer to this one behaviour.

### The clipping trade-off, quantified

The obvious fix is to stop letting the outlier set the scale. Clip absmax to 12.0 instead of 23.7 (so `s = 2.0`) and re-run the same block:

| x | unclipped (s = 4.0) | **clipped (s = 2.0)** |
|---:|---:|---:|
| 23.70 | 24.000 (1.27 %) | **12.000 (49.37 %)** ← clipped hard |
| 4.10 | 4.000 (2.44 %) | 4.000 (2.44 %) |
| −11.20 | −12.000 (7.14 %) | −12.000 (7.14 %) |
| **0.90** | **0.000 (100 %)** | **1.000 (11.11 %)** ← rescued |

```text
   LARGER SCALE         :  preserves range, some small values round to zero
   AGGRESSIVE CLIPPING  :  sacrifices tail values, may resolve small values better

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

under a locally uniform, uncorrelated rounding-error approximation, away from clipping. The familiar ~6.02 dB/bit scaling needs additional fixed-range/high-resolution assumptions. E2M1 is nonuniform and very low resolution; its RMS error depends on the actual values and scales. Do not assign it a universal 8–12% RMS error.

That sounds fatal. It is not, and the reason is the next section.

### Why 10 % element-wise error does not mean a 10 % wrong model

For a dot product `y=Σ x_i w_i`, replacing weights by `w_i+ε_i` changes the output by `δy=Σ x_i ε_i`. If errors are independent and zero-mean with common variance `σ²`, then `Var(δy)=σ² Σ x_i²`. This is an assumption to test, not a property guaranteed by quantization.

A `1/√K` **relative** error decrease additionally assumes a coherently growing signal. Dot products may cancel: with `x=(1,1)` and `w=(1,−1)`, the exact output is zero, so even tiny quantization error has undefined relative error. Correlated error, clipping, outliers, and repeated layers can defeat averaging. Report absolute and task-relevant error alongside relative error; do not predict 0.11% activation error from bit-width and K alone.

---

## 6. Where the bits go: W, A, and KV

"4-bit model" is ambiguous. Three tensors can each be quantized independently:

```text
                weights (W)          activations (A)         KV cache
   W4A16        4-bit  ◀── dequant to BF16 ──▶  16-bit       chosen separately
   W8A8         8-bit                            8-bit        chosen separately
   W4A4         4-bit                            4-bit       chosen separately
```

| Scheme | Weight traffic | Tensor-core input | Decode benefit | Difficulty |
|---|---|---|---|---|
| **W4A16** | up to 4× before scales/padding | BF16 (weights upconverted) | depends on weight traffic and conversion | runtime/shape dependent |
| **W8A8** | 2× less | FP8 native | moderate | moderate |
| **W4A4** | up to 4× before scales/padding | **FP4 native** | most + compute | **hard** — activations have outliers |

Weight traffic often dominates short-context, small-batch dense decoding, but KV/state traffic, unpacking, kernel utilization, and launch overhead can change the bottleneck. W4A16 and W4A4 specify weights and activations; neither specifies KV precision. W4A4 can unlock a different native tensor-core path even if activation storage is small, so its value cannot be ruled out using activation bytes alone. Compare supported kernels and quality on the exact model, shapes, and GPU. [TensorRT's quantization schemes](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-schemes.html) describe which combinations that runtime implements.

---

## 7. Format cheat sheet

Leading payload and block-scale storage, before the global scale/alignment/headers, for one `8192 × 8192` weight matrix (67.1 M elements):

| Format | bytes/elem | Matrix size | vs BF16 |
|---|---:|---:|---:|
| BF16 | 2.0 | 134.2 MB | 1.00× |
| FP8 E4M3 | 1.0 | 67.1 MB | 2.00× |
| **NVFP4** | **0.5625** | **37.7 MB** | **3.56×** |
| MXFP4 | 0.5312 | 35.7 MB | 3.76× |
| INT4 (group 128, FP16 scale) | 0.5156 | 34.6 MB | 3.88× |

Note that NVFP4 is **not** the smallest — it offers a particular scale/precision tradeoff and a native path on supported kernels. That second clause is what [Module 03](Lecture-03.md) is about, and its actual performance must be measured against alternatives.

---

## Checkpoint

You should now be able to:

1. Enumerate all 8 E2M1 magnitudes from memory and state the format's dynamic range (12×).
2. Compute NVFP4's effective bits/element (4.5) and explain both terms.
3. Predict, given a block's absmax, which of its elements will quantize to zero.
4. Explain the precision/range/block-size tradeoff between the two scale schemes.
5. Give a cancellation example that invalidates a universal 1/sqrt(K) relative-error claim.
6. Explain why a roofline estimate alone cannot choose between actual W4A16 and W4A4 kernels.

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

Handle zero tensors/blocks, row tails, scale underflow, clipping and round-to-nearest-even ties explicitly. The [executable CPU reference](CPU-Quantization-Lab.md) implements these conventions with actual PyTorch E4M3 casts and a teaching nibble layout; it is not a vendor checkpoint writer.

Then report, for one real weight tensor: absolute RMS error, relative error with a declared near-zero convention, **fraction of elements annihilated to zero**, and the error histogram. Repeat with block size 32 and with an E8M0 scale to measure the difference; do not assume its sign. That table is your artifact.

---

## Current as of

* **Timeless:** IEEE-style float decomposition, block scaling arithmetic, `Δ²/12`, the explicitly conditional error-variance calculation, the clipping/rounding trade.
* **2026 format pins:** NVFP4 = E2M1 + block 16 + E4M3 block scale + FP32 per-tensor scale. MXFP4 (OCP Microscaling) = E2M1 + block 32 + E8M0 scale. Both are current; the OCP MX specification is the reference for the latter.
* **Watch:** microscaling variants continue to appear (6-bit MXFP6, mixed block sizes). The `4 + scale_bits/block_size` arithmetic generalizes to all of them.

---

**Next:** [Module 03 — Blackwell Hardware →](Lecture-03.md)

**TensorRT source boundary (2026-10-18):** linked `latest` quantization pages describe TensorRT 11.4. They explain representation/support in that release, not compatibility of this case study's historical loader or a JetPack 10.x runtime. TensorRT 11.4 does not support JetPack; use [NVIDIA's platform migration guidance](https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-10x-to-11x-jetson.html) and the selected runtime's exact version when reproducing an engine.
