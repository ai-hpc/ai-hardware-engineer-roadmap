# Lecture 1 — Attention Bottleneck and Roofline

**Parent:** [FlashAttention Course](Guide.md)

**One-line purpose:** Build the IO/roofline model that explains why naive attention is memory-bound and what an "exact but IO-aware" algorithm needs to change.

**Prerequisites:** Familiarity with standard self-attention. Comfort with basic asymptotic analysis.

**Artifact:** A short Jupyter notebook that, for a fixed head config, plots arithmetic intensity vs sequence length and overlays the GPU's roofline.

---

## Why it matters

Attention has three steps that read or write large tensors:

1. **Scores:** `S = Q · Kᵀ / √d`, shape `[N, N]`.
2. **Softmax:** `P = softmax(S)`, also `[N, N]`.
3. **Output:** `O = P · V`, back to `[N, d]`.

Naive implementations materialize the full `[N, N]` matrices `S` and `P` in HBM. For `N = 8192` with bf16 that is 128 MiB per head per layer just for `S`, and another 128 MiB for `P`. On a Hopper-class GPU you can deliver ~3 TB/s of HBM bandwidth, so moving those two matrices already costs around 90 µs per head before you do any math. With many heads and layers, the model spends most of its life moving softmax intermediates through HBM rather than doing matmuls.

This is exactly the kind of workload where **roofline analysis** tells you the bottleneck is bandwidth, not compute, and where an "IO-aware" rewrite gives big speedups without changing the answer.

---

## Mental model

### The two matmuls and the elementwise stage

Per attention head with sequence length `N` and head dim `d`:

| Step | Op | FLOPs | Bytes read (bf16) | Bytes written (bf16) |
|------|----|-------|--------------------|----------------------|
| `S = QKᵀ` | matmul | `2 N² d` | `2 (2 N d)` | `2 N²` |
| `softmax(S)` | elementwise | ~`5 N²` | `2 N²` | `2 N²` |
| `O = PV` | matmul | `2 N² d` | `2 (N² + N d)` | `2 N d` |

Total compute: `4 N² d + O(N²)` FLOPs.
Total HBM traffic (naive, all intermediates materialized): `O(N² + N d)` bytes.

### Arithmetic intensity

Arithmetic intensity `I = FLOPs / bytes`. The roofline says a kernel can hit at most `min(peak_FLOPs, I · peak_bytes_per_sec)` FLOPs/s.

For the explicitly materialized bf16 path in the table, count every read and write: total traffic is approximately `8 N² + 8 N d` bytes. At large N, `I ≈ 4 N² d / (8 N²) = d/2` FLOPs/byte, or about **32 FLOPs/byte at d=64**. Ignoring the repeated reads/writes gives the wrong constant. Compare with illustrative hardware rooflines (verify the exact SKU and dense/sparse convention before benchmarking):

| GPU | Peak bf16 (TFLOPS) | HBM BW (TB/s) | Ridge point I (FLOPs/byte) |
|-----|--------------------|----------------|------------------------------|
| A100 80GB SXM | ~312 | 2.0 | ~156 |
| H100 SXM | ~989 | 3.35 | ~295 |
| H200 SXM | ~989 | 4.8 | ~206 |
| B200 SXM | ~2250 (bf16) | 8.0 | ~280 |

Below the ridge, the roofline predicts a bandwidth ceiling lower than the compute ceiling under its assumptions. The materialized pipeline at d=64 has aggregate intensity about 32 FLOPs/byte; at d=128, about 64. Its individual GEMMs and softmax can have different bottlenecks. Profile each stage: launch overhead, cache reuse, and non-matmul execution can invalidate a simple whole-pipeline diagnosis.

### What "exact but IO-aware" means

You cannot reduce the math: the answer is still `softmax(QKᵀ/√d)·V`, mathematically, with numerical differences possible from floating-point reduction order. What you can change is *how* you traverse the math: produce `O` without ever writing the full `[N, N]` matrix to HBM. If you can keep `S` and `P` tiles inside SRAM (shared memory) while you stream `Q`, `K`, `V`, you avoid storing the full quadratic score/probability matrices. **Linear auxiliary memory is not the same as linear total HBM traffic.** With SRAM capacity M measured in elements, the FA1 analysis gives an HBM-access term `O(N² d² / M)` in its stated regime, plus required inputs/outputs. Finite tiles can reread Q/K/V. Use the [paper's IO analysis](https://arxiv.org/html/2205.14135v2#S3.SS2) and measure the executed kernel before calling it compute-bound.

That is the entire premise of FlashAttention. The rest of the course is about how to make that traversal correct, fast, and general.

---

## Build it

Read the FA1 paper, Section 2 (Background) and Section 3.1 (Standard attention IO complexity). Then make this notebook:

```python
# attention_roofline.py
import numpy as np
import matplotlib.pyplot as plt

def naive_attention_bytes(N, d, dtype_bytes=2):
    # Q, K, V reads
    qkv_read = 3 * N * d * dtype_bytes
    # S = QK^T write, P = softmax(S) read+write, P read for PV
    s_write = N * N * dtype_bytes
    p_rw = 2 * N * N * dtype_bytes
    p_read_for_pv = N * N * dtype_bytes
    # O write
    o_write = N * d * dtype_bytes
    return qkv_read + s_write + p_rw + p_read_for_pv + o_write

def attention_flops(N, d):
    return 4 * N * N * d  # the two matmuls dominate

def intensity(N, d):
    return attention_flops(N, d) / naive_attention_bytes(N, d)

def flash_attention_bytes(N, d, dtype_bytes=2):
    # Stream Q, K, V from HBM; write O; never materialize S or P.
    return (3 * N * d + N * d) * dtype_bytes

GPUS = {
    "A100 80GB":  (312e12, 2.0e12),
    "H100 SXM":   (989e12, 3.35e12),
    "H200 SXM":   (989e12, 4.8e12),
}

d = 128
Ns = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
print(f"head_dim={d}, dtype=bf16")
print(f"{'N':>6} {'naive_I':>10} {'fa_I':>10}")
for N in Ns:
    fa_I = attention_flops(N, d) / flash_attention_bytes(N, d)
    print(f"{N:>6} {intensity(N, d):>10.1f} {fa_I:>10.1f}")
```

Run it, save the table, and write down for each GPU which `(N, d)` combinations fall below the ridge with the naive layout vs the FlashAttention layout.

---

## Use it in the real stack

Open `torch.nn.functional.scaled_dot_product_attention` (PyTorch's `SDPA`). It has three backends: `math` (the naive, materialised version), `flash` (FlashAttention), and `efficient` (xFormers). The `math` backend exists precisely so you can compare against it for correctness; it is also the kernel whose IO complexity matches your "naive" line on the roofline.

```python
import torch
import torch.nn.functional as F

q = torch.randn(1, 8, 4096, 64, device="cuda", dtype=torch.bfloat16)
k = torch.randn_like(q); v = torch.randn_like(q)

with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
    o_math = F.scaled_dot_product_attention(q, k, v)
with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.FLASH_ATTENTION]):
    o_flash = F.scaled_dot_product_attention(q, k, v)

print("max abs diff:", (o_math - o_flash).abs().max().item())
```

You should see correctness within bf16 tolerance and a large latency gap. Time both, and compare the difference to the byte-count ratio you computed in the lab.

---

## Measure it

- Use `torch.cuda.Event` for timing.
- Run at fp16 / bf16, never fp32 — these are the only realistic LLM training/inference dtypes.
- Warm up the kernel before timing.
- Report `tokens/sec` (or kernel time), not just absolute milliseconds.

For each shape, compute:

`achieved_bandwidth ≈ kernel_bytes / kernel_time`

and compare to your GPU's peak HBM bandwidth. For naive attention you should be close to peak HBM. For FlashAttention you should be far below HBM peak (because you read less) but high on TFLOPS utilisation.

---

## Ship it

Commit the notebook to your `flash-attn-course/` directory. It should produce:

1. A printed table of `(N, d, naive_I, fa_I)` for at least three `d` values.
2. A matplotlib roofline plot for one GPU you actually own / rent, with markers for naive and FA attention at `N ∈ {1024, 4096, 16384}`.
3. A two-sentence written conclusion: "Naive attention at d={...} crosses below the ridge at N={...}; FA stays above the ridge across the entire sweep."

If you can produce that, you have the only mental model you need for the rest of the course.

---

## Related pages

- [02 — Kernel Engineering](../Guide.md)
- [Lecture 2 — Online softmax and numerical correctness](Lecture%2002%20-%20Online%20Softmax%20and%20Numerical%20Correctness.md)
