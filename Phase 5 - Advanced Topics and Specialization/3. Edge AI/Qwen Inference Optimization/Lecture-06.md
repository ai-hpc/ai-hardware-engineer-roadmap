# Lecture 6: Batched GEMM vs Normal GEMM — Kernels, Declarations, and Bit-Comparable Results

## Overview

Lecture 3 showed that **decode** is GEMV-bound on edge hardware. Lecture 4 mentioned, without spending time on it, that **prefill** and **batched decode** are GEMM-bound. This lecture sits between them: it dissects the GEMM and **batched** GEMM kernels at the cuBLAS API level, shows the three forms (single, array-of-pointers, strided), explains where each one fits in the Qwen inference path, and — most importantly — shows how to get the **same numerical result** across forms when you need cross-validation.

If you've ever fed a tensor into `cublasSgemmStridedBatched` and gotten outputs that looked *almost* right but slightly off, this is your lecture.

By the end you should be able to:

* Read a cuBLAS GEMM declaration and identify the layout, leading dimensions, and batch stride.
* Choose `Gemm`, `GemmBatched`, or `GemmStridedBatched` for a given workload (and explain why).
* Reproduce a single-GEMM result from a batched GEMM (and vice versa) bit-for-bit.
* Recognize the eight or so failure modes that produce "wrong but plausible" outputs.

This lecture is CUDA / cuBLAS-centric because that's where Qwen production inference lives. The same concepts apply to rocBLAS (`rocblas_sgemm_strided_batched`), MKL (`cblas_sgemm_batch`), and Apple's `Accelerate` framework with `BNNSBatch`. The API names differ; the math doesn't.

---

## 1. Where Batched GEMM Shows Up in Qwen Inference

Three places, all important:

### 1.1 Prefill

When you feed a prompt of `seq_len = N` tokens to Qwen3-4B, every linear projection at every layer becomes a GEMM, not a GEMV:

```
Input:   X = [d_model, N]               # d_model = 2560 for Qwen3-4B
W_Q:     [d_proj, d_model]              # d_proj = 4096
Q = W_Q @ X → [d_proj, N]               # GEMM: M=4096, N=N, K=2560
```

For `seq_len = 256` this is `M=4096, N=256, K=2560` — sized for tensor cores, ~5 ms on H100, ~50 ms on Orin Nano. Prefill TTFT depends almost entirely on how fast these GEMMs run.

This is a **single GEMM** per projection — not a batched GEMM. The "batch" is one matrix; the second dimension is the sequence.

### 1.2 Per-head attention scores (this is where batched GEMM appears)

After QKV projections and RoPE, you have:

```
Q:  [n_heads, N, head_dim]      # for Qwen3-4B: [32, N, 128]
K:  [n_kv_heads, N, head_dim]   # [8, N, 128]
V:  [n_kv_heads, N, head_dim]   # [8, N, 128]
```

The score computation is one independent matrix multiply **per head**:

```
S[h] = Q[h] @ K[h//group]^T    # for each head h
                                # [N, head_dim] @ [head_dim, N] = [N, N]
```

`n_heads = 32` independent GEMMs of shape `(N, head_dim, N)`. This is a **batched GEMM** of batch size 32.

In practice **nobody actually calls cuBLAS for attention anymore** — FlashAttention fuses the GEMM + softmax + second GEMM into one tiled kernel. But the batched-GEMM-shaped problem is what FlashAttention is solving under the hood, and reference implementations (Sionna, eager PyTorch) still go through `torch.matmul` → `cublasSgemmStridedBatched`.

### 1.3 Batched serving (Qwen2.5-72B with multiple users)

When vLLM continuous-batches B requests with the same length, every projection becomes:

```
X = [d_model, B · N]            # B sequences concatenated
Q = W_Q @ X → [d_proj, B · N]   # one big GEMM, not batched
```

That's still a single GEMM with bigger N. **The batch dimension becomes the GEMM's second dimension.** This is the highest-throughput regime — tensor cores run at full utilization on `M=4096, N=B·N, K=2560` when `B·N ≥ 1024`.

The "batched GEMM" API form actually shows up when sequences have **different lengths and KV-cache layouts** — but that's where paged attention takes over with custom kernels.

---

## 2. The Three cuBLAS GEMM Forms

### 2.1 Normal GEMM — single matrix pair

```c
// y = alpha · op(A) · op(B) + beta · C
cublasStatus_t cublasSgemm(
    cublasHandle_t handle,
    cublasOperation_t transa,         // CUBLAS_OP_N (no transpose) | _T | _C
    cublasOperation_t transb,
    int m, int n, int k,
    const float *alpha,
    const float *A, int lda,           // device pointer + leading dim
    const float *B, int ldb,
    const float *beta,
    float *C, int ldc);
```

Reading the shapes: result is `[m, n]`, `A` is `[m, k]` after `transa`, `B` is `[k, n]` after `transb`. The "leading dimension" is the **stride between columns** (because cuBLAS is column-major).

### 2.2 GemmBatched — array of pointers

```c
cublasStatus_t cublasSgemmBatched(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m, int n, int k,
    const float *alpha,
    const float *const Aarray[], int lda,    // device pointer to array of device pointers
    const float *const Barray[], int ldb,
    const float *beta,
    float *const Carray[], int ldc,
    int batchCount);
```

The matrices can live anywhere — `Aarray[i]` points to A_i, which doesn't have to be contiguous with `Aarray[i+1]`. Useful when:

- Matrices come from a list of allocations.
- Per-batch matrices have **different sizes** (you'd loop over groups of same-size batches).
- Matrices are scattered across memory pools.

**Cost:** the array of pointers is itself a device array, so you pay an extra indirection per batch element. For batch sizes < ~16 this matters.

### 2.3 GemmStridedBatched — contiguous strided matrices

```c
cublasStatus_t cublasSgemmStridedBatched(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m, int n, int k,
    const float *alpha,
    const float *A, int lda, long long int strideA,   // single base ptr + stride
    const float *B, int ldb, long long int strideB,
    const float *beta,
    float *C, int ldc, long long int strideC,
    int batchCount);
```

A_i lives at `A + i * strideA`, with the same `m, k, lda` for every batch. This is what **virtually all LLM inference uses** — `Q`, `K`, `V` tensors are 3D contiguous tensors, the batch (head) dimension has a clean fixed stride.

Use this whenever you can. It's faster than the pointer-array form because cuBLAS can compute pointers from the stride at zero indirection cost.

### 2.4 The Ex variants — for mixed precision

For FP16 / BF16 / FP8 / TF32, use the `Ex` versions:

```c
cublasGemmEx(...)                   // single, mixed precision
cublasGemmBatchedEx(...)            // batched-pointer, mixed precision
cublasGemmStridedBatchedEx(...)     // strided-batched, mixed precision
```

They take separate `Atype`, `Btype`, `Ctype`, `computeType`, and `algo` parameters. **This is what production LLM inference actually calls** — FP16/BF16 weights with FP32 accumulation on tensor cores.

---

## 3. Memory Layout — Column-Major, Row-Major, and the Transpose Dance

cuBLAS is **column-major**. PyTorch/NumPy/C/Python are **row-major**. This is the **single most common source of bugs**.

A row-major `[M, K]` matrix in memory is byte-for-byte identical to a column-major `[K, M]` matrix. Same bytes; different interpretation. So when you call cuBLAS on row-major data, you give it:

```
"I have row-major X = [M, K]"   ⟺   "cuBLAS sees column-major X^T = [K, M]"
```

To compute `Y = X @ W` (row-major, where `X = [M, K]`, `W = [K, N]`):

```
PyTorch/NumPy view:
   Y[M, N] = X[M, K] · W[K, N]

cuBLAS view (after row→col reinterpretation):
   Y^T[N, M] = W^T[N, K] · X^T[K, M]

cuBLAS call (no extra transposes — the layout flip handles it):
   cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
               N, M, K,        // note: m=N, n=M
               &alpha,
               W, N,            // W^T from cuBLAS view, lda = N
               X, K,            // X^T from cuBLAS view, lda = K
               &beta,
               Y, N);           // Y^T, ldc = N
```

This is the **swap-arguments idiom** every cuBLAS-using C++ inference codebase uses. Read it once carefully; then internalize.

For batched, the strides follow the same flip:

```
strideA = M * K   (PyTorch view) ↔  K * M (cuBLAS view, but same bytes)
```

The bytes are the same; the only question is which dimension cuBLAS calls `m` vs `n`. Once you flip your mental model, everything just works.

### Leading dimension cheat sheet

| Layout | Matrix `[rows, cols]` | `ld` value |
|---|---|---|
| Column-major, not transposed | `[m, k]` | `m` |
| Column-major, transposed | `[k, m]` (stored as `[m, k]`) | `m` |
| Row-major reinterpreted as col-major | `[k, m]` in cuBLAS view | `k` |

When you set `ld > rows` it means you have **padding** between columns — useful for alignment. Most LLM inference uses `ld == rows` (no padding).

---

## 4. Producing the Same Result Across Forms

You'll often write reference code with one cuBLAS form and production code with another. The question: when do they produce **identical** results?

### 4.1 Bit-exact across forms — the strict version

For two cuBLAS calls (e.g., `Gemm` looped vs `GemmStridedBatched`) to produce bit-identical results, **all of**:

1. **Identical operand bytes.** The strides and offsets must lay out the same matrices.
2. **Identical `alpha`, `beta`, `computeType`.**
3. **Same algorithm selected.** cuBLAS will auto-tune; force consistency with `cublasGemmEx(..., CUBLAS_GEMM_ALGO0)` or whichever algo you pin.
4. **Same architecture.** Tensor core paths produce different bit patterns than CUDA-core paths because of different accumulation orders.
5. **No TF32 reduction.** Set `cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH)`. The default `CUBLAS_DEFAULT_MATH` on Ampere+ silently uses TF32 in some FP32 paths — different rounding.

When all five hold, the results match bit-for-bit on the same hardware.

### 4.2 Numerically equivalent (FP32-equivalent epsilon)

For most practical purposes you don't need bit-exact, you need "the difference is at most floating-point noise." For FP32 matmul that means roughly:

```
|result_a - result_b| / |result_a| < ~1e-6 · sqrt(K)
```

(The `sqrt(K)` comes from the random-walk error model of accumulating K products.)

So for a `K = 2560` GEMM, you should expect relative differences up to **~5e-5** in FP32 across different algorithms or accumulation orders. Anything bigger and you have a real bug.

For FP16 matmul with FP32 accumulation (the common LLM case), bound is similar in absolute terms because accumulation is FP32 — but the FP16 inputs round to ~1e-3 each.

### 4.3 What goes wrong — the eight common bugs

| Bug | Symptom | How to spot |
|---|---|---|
| Row/col-major confusion | Output looks transposed | Reshape result `[m, n]` to `[n, m]` — does it match? |
| Wrong `transa`/`transb` flags | Result is `A·B^T` or `A^T·B` instead of `A·B` | Run with `M=K=N=2` and verify by hand |
| Wrong `lda`/`ldb`/`ldc` | Output looks like junk past first column | Smaller test; print first 2×2 block |
| Strided-batched stride wrong | Some batches correct, others overlap | Output[i] uses elements from batch i+1 |
| Pointer-array points to wrong offsets | Random batch elements correct/wrong | Print pointer values; diff with manual calc |
| Mixed precision: alpha is FP32 but pointed-to value reads as FP16 | Output ~half what it should be | Print alpha value; check `computeType` |
| Tensor cores require alignment that isn't met | Falls back to slow path | Performance hint, not correctness |
| `beta != 0` but `C` not initialized | Adds random GPU memory contents | Set `beta = 0` for fresh outputs |

The last one is the absolute classic. cuBLAS does `C = alpha · A·B + beta · C`. If `beta = 1` and `C` is garbage memory, you add garbage.

---

## 5. Minimal Worked Example — Single vs Strided Batched

We'll verify `GemmStridedBatched` produces the same output as a loop of `Gemm` calls, on three identical 4×4 matrices.

```c++
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

int main() {
    const int M = 4, N = 4, K = 4, BATCH = 3;
    const size_t SIZE = M * K * BATCH * sizeof(float);

    // Host data: three 4×4 matrices A, B, C; same data for clarity.
    float h_A[M * K * BATCH], h_B[K * N * BATCH], h_C_single[M * N * BATCH], h_C_batch[M * N * BATCH];
    for (int i = 0; i < M * K * BATCH; ++i) h_A[i] = (i % 7) * 0.1f;
    for (int i = 0; i < K * N * BATCH; ++i) h_B[i] = (i % 5) * 0.2f;

    // Device pointers
    float *d_A, *d_B, *d_C_single, *d_C_batch;
    cudaMalloc(&d_A, SIZE);  cudaMalloc(&d_B, SIZE);
    cudaMalloc(&d_C_single, SIZE);  cudaMalloc(&d_C_batch, SIZE);
    cudaMemcpy(d_A, h_A, SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, SIZE, cudaMemcpyHostToDevice);
    cudaMemset(d_C_single, 0, SIZE);
    cudaMemset(d_C_batch,  0, SIZE);

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);    // force deterministic path

    const float alpha = 1.0f, beta = 0.0f;

    // 1) Looped single GEMM
    for (int b = 0; b < BATCH; ++b) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    M, N, K,
                    &alpha,
                    d_A + b * M * K, M,
                    d_B + b * K * N, K,
                    &beta,
                    d_C_single + b * M * N, M);
    }

    // 2) One strided batched GEMM
    cublasSgemmStridedBatched(handle,
                              CUBLAS_OP_N, CUBLAS_OP_N,
                              M, N, K,
                              &alpha,
                              d_A, M, (long long)(M * K),     // strideA
                              d_B, K, (long long)(K * N),     // strideB
                              &beta,
                              d_C_batch, M, (long long)(M * N), // strideC
                              BATCH);

    cudaMemcpy(h_C_single, d_C_single, SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_batch,  d_C_batch,  SIZE, cudaMemcpyDeviceToHost);

    // Compare
    float max_diff = 0.0f;
    for (int i = 0; i < M * N * BATCH; ++i) {
        max_diff = fmaxf(max_diff, fabsf(h_C_single[i] - h_C_batch[i]));
    }
    printf("Max abs diff: %.3e\n", max_diff);
    // Expected: 0.0 with PEDANTIC_MATH, ~1e-7 otherwise.

    cublasDestroy(handle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C_single); cudaFree(d_C_batch);
}
```

Compile and run:

```bash
nvcc -lcublas batched_gemm_compare.cu -o batched_gemm_compare
./batched_gemm_compare
# Max abs diff: 0.000e+00
```

With `CUBLAS_PEDANTIC_MATH`, the diff is exactly zero. Without it (commenting that line out), you may see `~1e-7` differences on Ampere/Hopper because the default math mode allows tensor-core reduction with different accumulation orders.

---

## 6. Reproducing One Form From Another — The Recipe

### 6.1 Looped single → strided batched

When you have a working `Gemm` loop and want to fuse:

```c++
// Before (loop):
for (int b = 0; b < B; ++b) {
    cublasSgemm(..., A + b*sA, ..., B + b*sB, ..., C + b*sC);
}

// After (strided batched):
cublasSgemmStridedBatched(...,
                          A, lda, sA,
                          B, ldb, sB,
                          C, ldc, sC,
                          B);
```

Conditions: every batch element has same `m,n,k,lda,ldb,ldc` and same alpha/beta. If any of these vary per batch, you must use the pointer-array form or call `Gemm` in a loop with the right per-call values.

### 6.2 Strided batched → array of pointers (when sizes vary)

When the per-batch sizes are the same but pointers are scattered:

```c++
const float *Aarray[B];
const float *Barray[B];
float       *Carray[B];
for (int b = 0; b < B; ++b) {
    Aarray[b] = my_A_pool[b];   // wherever each matrix actually lives
    Barray[b] = my_B_pool[b];
    Carray[b] = my_C_pool[b];
}
// Aarray must live on device too — copy to a device array first
cudaMemcpyAsync(d_Aarray, Aarray, sizeof(Aarray), cudaMemcpyHostToDevice);
// ... same for Barray, Carray

cublasSgemmBatched(handle, ...,
                   d_Aarray, lda,
                   d_Barray, ldb,
                   d_Carray, ldc,
                   B);
```

### 6.3 Single big GEMM → batched (when input is "actually one matrix")

Sometimes your data is a single `[M, B·N]` matrix where columns belong to different "batches" semantically. If the same `A` multiplies all of them:

```
single:  C[M, B·N] = A[M, K] @ X[K, B·N]
```

There's no reason to use batched GEMM — this is one large GEMM with bigger N, and tensor cores love it. **This is the most efficient regime for LLM serving** — batched decode with B sequences becomes one big GEMM, not B small ones.

The mistake to avoid: writing this as `cublasSgemmBatched` with `batchCount=B`. You pay launch and indirection overhead for nothing; the math is one GEMM.

---

## 7. Tensor Cores and What Changes Under the Hood

cuBLAS auto-selects between:

* **CUDA core path** — FP32 FMAs, deterministic accumulation order if you fix the algorithm.
* **Tensor core path** — `mma.sync` instructions, FP16 multiply + FP32 accumulate, reduction across a warp.

The tensor core path is enabled when:

1. `computeType` is one of `CUDA_R_16F`, `CUDA_R_32F` (with TF32 on Ampere+), or quantized types.
2. Dimensions are multiples of the MMA tile (Ampere: 16×16×16 for FP16; Hopper: 16×16×16 or 64×16×16 for WGMMA).
3. Pointer alignment is met (16-byte minimum, often 128-byte).
4. Math mode isn't `CUBLAS_PEDANTIC_MATH`.

You opt in/out explicitly with:

```c++
cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);  // allow TF32 on tensor cores
cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);         // let cuBLAS choose
cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);        // strict IEEE, no tensor cores for FP32
```

For Qwen inference: **always use tensor cores**. The FP32 accumulator catches the precision loss; the throughput gain is 5–10×. The only time you'd disable tensor cores is for debugging or for bit-exact numerical reproducibility checks.

---

## 8. PyTorch Equivalents

The cuBLAS calls under each PyTorch op:

| PyTorch | Likely cuBLAS call | Notes |
|---|---|---|
| `torch.matmul(A, B)` for 2D | `cublasGemmEx` | If shapes are MMA-friendly |
| `torch.matmul(A, B)` for 3D batch | `cublasGemmStridedBatchedEx` | Batch dim must be contiguous |
| `torch.bmm(A, B)` | `cublasGemmStridedBatchedEx` | Same as 3D matmul; legacy alias |
| `F.linear(x, W)` | `cublasGemmEx` (computes `x @ W^T`) | One of the most-called ops in LLMs |
| `torch.einsum("bnd,bmd->bnm", q, k)` | Strided batched | Common in attention reference impls |
| `torch.nn.functional.scaled_dot_product_attention` | Custom (FlashAttention or memory-efficient) | Not pure cuBLAS |

If you want to inspect the dispatch on Jetson / discrete GPU:

```bash
TORCH_SHOW_CPP_STACKTRACES=1 python -c "
import torch
A = torch.randn(4, 128, 64, device='cuda', dtype=torch.float16)
B = torch.randn(4, 64, 256, device='cuda', dtype=torch.float16)
torch.cuda.synchronize()
import torch.profiler as p
with p.profile(activities=[p.ProfilerActivity.CUDA], record_shapes=True) as prof:
    C = torch.bmm(A, B)
    torch.cuda.synchronize()
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))
"
```

You should see something like `aten::bmm` → `cublasGemmStridedBatchedEx_internal`.

---

## 9. Qwen Inference: Where Each Form Lands

Concrete mapping for **Qwen3-4B prefill at seq_len = 512** on Orin:

| Op | Form | Shape (cuBLAS view) |
|---|---|---|
| QKV projection (fused) | Single GEMM | `M=6144, N=512, K=2560` |
| Attention score `Q @ K^T` | Strided batched (or FlashAttention) | batch=32 (heads), `M=512, N=512, K=128` |
| Softmax | Custom kernel (not GEMM) | — |
| Attention value `S @ V` | Strided batched (or FlashAttention) | batch=32, `M=512, N=128, K=512` |
| Output projection W_O | Single GEMM | `M=2560, N=512, K=4096` |
| FFN gate, up (fused) | Single GEMM | `M=13824, N=512, K=2560` |
| FFN down | Single GEMM | `M=2560, N=512, K=6912` |
| LM head (final layer only) | Single GEMM | `M=151936, N=512, K=2560` |

For **Qwen2.5-72B prefill at seq_len = 4096, TP=4**: same structure, dimensions scaled. The QKV projection on each TP rank is `M=2560 (sharded), N=4096, K=8192`. Big GEMMs; the H100 spends most of its prefill time in cuBLAS calls.

For **batched decode (continuous batching)**: all matrices have N = effective batch size (sum of active sequences). Single GEMMs, no batched form needed.

The only place batched-GEMM API actually appears in the modern Qwen inference path is in **reference attention implementations** (PyTorch eager mode without FlashAttention). FlashAttention's custom kernel does what `Q @ K^T` and `S @ V` would do, but tiled.

---

## 10. Debugging Recipe — "Why Are My Numbers Wrong?"

When a batched-GEMM call gives an output that doesn't match the looped reference, work through these in order:

1. **Print shapes.** Are `m, n, k` what you think? Off-by-one is suspicious.
2. **Print first 4×4 block of each operand.** Does it match your reference?
3. **Force `CUBLAS_PEDANTIC_MATH`.** If the bug goes away, your problem was algorithm-selection non-determinism — not actually a bug.
4. **Set `beta = 0` and pre-zero `C`.** Removes the "garbage input" case.
5. **Try `batchCount = 1` of the strided form.** Should match the single `Sgemm` call exactly.
6. **Check transpose flags by computing M=N=K=2 by hand.** A 2×2 example pinpoints transpose bugs immediately.
7. **Check strides.** `strideA = lda · k` for non-transposed `A` in column-major. **Off by `lda · k vs lda · m` is a frequent mistake.**
8. **Check pointer offsets** in the array-of-pointers form. Print the device pointers; diff against manual `base + i * stride`.

If all of these check out and the numbers still differ — you have a real bug. Usually it's #7 or #8.

---

## 11. Hands-On Exercises

1. **Bit-exact validation.** Compile the §5 example. Run it with `CUBLAS_PEDANTIC_MATH` on and off. Record the max-diff in each case. Then add `cudaDeviceProp.major` to the output and run on two different GPU architectures (e.g., Orin and a desktop RTX). Report cross-arch reproducibility.

2. **Prefill GEMM benchmarking.** Take Qwen3-4B prefill QKV projection sizes (`M=6144, K=2560`) and benchmark for `N ∈ {1, 4, 16, 64, 256, 1024}`. Plot tok/s and GFLOPS. Identify the N where tensor-core utilization saturates.

3. **Strided-batched as the only form for attention reference.** Implement the attention `Q @ K^T → softmax → @ V` in pure cuBLAS — two `GemmStridedBatched` calls plus a softmax kernel of your choice. Compare output to PyTorch's `F.scaled_dot_product_attention` on the same inputs.

4. **The transpose dance.** Take a row-major PyTorch tensor `X[128, 256]` and a `W[256, 512]`. Compute `Y = X @ W` two ways: (a) via PyTorch, (b) via cuBLAS `Sgemm` with the swap-arguments idiom. Verify max-diff is < 1e-4.

5. **Strided vs pointer-array benchmark.** For a workload of 64 batched 64×64 GEMMs, benchmark `GemmStridedBatched` vs `GemmBatched` (where you populate the pointer array yourself). Quantify the indirection overhead. Repeat for batch size 4 and batch size 1024.

6. **PyTorch dispatch inspection.** Profile `torch.bmm` on shapes you'd actually see in Qwen attention (batch=32, M=512, K=128, N=512). Confirm via `nsys` that PyTorch dispatches to `cublasGemmStridedBatchedEx`. Then switch to `torch.nn.functional.scaled_dot_product_attention` and observe FlashAttention being called instead.

7. **Tensor core alignment.** Run the QKV projection GEMM at `N = 256` (clean tile) and `N = 251` (awkward). Measure tok/s. Confirm the awkward N is significantly slower because cuBLAS can't use tensor cores for the trailing tile.

---

## 12. Key Takeaways

| Takeaway | Why it matters |
|---|---|
| `GemmStridedBatched` is the form 95% of inference uses | Same alpha/beta, same shapes, contiguous strides — fits LLM attention exactly |
| `GemmBatched` (pointer array) is for scattered, same-size matrices | Pay indirection cost; only use when strided form isn't available |
| cuBLAS is column-major; the swap-arguments idiom handles row-major input | The single most common source of layout bugs |
| Set `beta = 0` whenever C is fresh output | The classic "I added random GPU memory" footgun |
| `CUBLAS_PEDANTIC_MATH` is your bit-exact debug button | Disables tensor cores and algorithm selection — slow but deterministic |
| Batched serving is one big GEMM, not many small ones | Don't reach for batched API when you have a contiguous concat batch |
| For attention, FlashAttention has replaced explicit batched GEMM in production | Reference implementations still go through it; modern serving doesn't |
| Eight debug-recipe steps catch >90% of "wrong output" bugs | Stride and transpose mistakes are everyone's favorite |

---

## Resources

* **[cuBLAS Library User Guide](https://docs.nvidia.com/cuda/cublas/index.html):** Reference for all the `Gemm*` variants.
* **[cuBLAS sample — strided batched GEMM](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/4_CUDA_Libraries/batchCUBLAS):** Minimal working example with both `Batched` and `StridedBatched`.
* **[CUTLASS — Composable Templates for GEMM](https://github.com/NVIDIA/cutlass):** When cuBLAS isn't flexible enough; templated tile-level GEMM with full control.
* **[Marlin GPTQ kernel](https://github.com/IST-DASLab/marlin):** Modern 4-bit GEMM used by vLLM; not pure cuBLAS but follows the same shape conventions.
* **["Efficient GEMM in Modern Architectures" — Anatomy of Matrix Multiplication](https://arxiv.org/abs/1808.07984):** Background on tile-level GEMM design.
* **[NVIDIA Math Mode Reference](https://docs.nvidia.com/cuda/cublas/index.html#cublasmath_t):** TF32, FP16, BF16, PEDANTIC modes explained.
* **[PyTorch `torch.bmm` documentation](https://pytorch.org/docs/stable/generated/torch.bmm.html):** The user-facing batched matmul; dispatches to cuBLAS.
* **[Phase 5 — Edge LLM Inference Internals](../Edge%20LLM%20Inference%20Internals/Lecture-01.md):** The GEMV-side companion to this GEMM lecture.
* **[Phase 4 — Track C — Kernel Engineering](../../../Phase%204%20-%20Track%20C%20-%20ML%20Compiler%20and%20Graph%20Optimization/DL%20Inference%20Optimization/02%20-%20Kernel%20Engineering/Guide.md):** Deeper compiler-side view of GEMM kernel design.
