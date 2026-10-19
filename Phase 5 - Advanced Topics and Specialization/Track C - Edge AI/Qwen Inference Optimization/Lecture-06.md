# Lecture 6: GEMM layouts, batching, and numerical validation

**Reviewed 18 October 2026.** Input: matrix multiplication and Lecture 1's projection shapes. Output: a layout-correct reference and a validation plan for a supported BLAS backend. The supplied padded, non-square matrix exercise runs on a CPU without CUDA.

## 1. Distinguish a larger matrix from a batch of matrices

A Qwen3 prefill Q projection uses weights `[4096,2560]` and token vectors `[2560,T]`, producing `[4096,T]`. That is one logical GEMM. Several requests' projection inputs may be concatenated into one token dimension because they share the same weight matrix.

Attention has independent head products: `Q[h] @ K[h//4].T` for this grouped-query model. A prefill reference can represent them as batched products; an optimized attention kernel may fuse score computation, masking, softmax and value accumulation. Packed variable-length sequences require boundaries and masks. The conceptual algebra does not dictate one specific cuBLAS call.

For one decode query, attention scores have `1×T` elements per head; for `T` prefill queries, the logical matrix is `T×T`. Do not infer kernel time from the mathematical shape without measuring the implementation.

## 2. Three BLAS interfaces, three storage contracts

All compute `C = alpha*op(A)*op(B) + beta*C`. The [cuBLAS reference](https://docs.nvidia.com/cuda/cublas/index.html) defines the exact supported types and parameters.

| Interface | What may differ across members in one call? |
|---|---|
| `cublasSgemm` | one matrix pair |
| `cublasSgemmBatched` | matrix pointers can differ; one shared shape/transpose/leading-dimension/scalar contract |
| `cublasSgemmStridedBatched` | base pointers plus fixed element strides; same shared operation contract |

Pointer-array batched GEMM does **not** permit arbitrary dimensions or different alpha/beta per member in that one call. Grouped interfaces or separate calls handle supported heterogeneous groups. Pointer arrays themselves must be accessible as required by the API; an ordinary host array of device pointers is not automatically a device pointer array.

Strided form can simplify pointer management. It is not guaranteed faster than pointer-array or looped calls for every shape/library/device. Also, ordinary GQA head mapping repeats each KV head several times; that repeated-block pattern is not necessarily expressible with one constant KV stride across all query heads. Use a compatible grouping, pointer arrangement, broadcast-capable kernel or fused implementation.

## 3. Leading dimensions describe stored memory

Classic cuBLAS interprets matrices as column-major. The element at stored row `i`, column `j` is at `base[i+j*ld]`. `ld` is in **elements**, not bytes, and refers to stored columns even when `op(A)` is transposed.

For logical output `[m,n]` and reduction width `k`:

| Operand | Stored shape | Minimum leading dimension |
|---|---|---:|
| A, no transpose | `[m,k]` | `max(1,m)` |
| A, transpose | `[k,m]` | `max(1,k)` |
| B, no transpose | `[k,n]` | `max(1,k)` |
| B, transpose | `[n,k]` | `max(1,n)` |
| C | `[m,n]` | `max(1,m)` |

A fully padded non-transposed A allocation can use `lda*k` elements; a transposed A can use `lda*m`. Batch stride is the actual distance to the next matrix and can include additional gaps. Choose disjoint output regions; overlapping writable batches are not a valid independence assumption.

For row-major `X[M,K]`, `W[K,N]`, and `Y[M,N]=XW`, reinterpret the same buffers as column-major transposes:

```text
Y^T[N,M] = W^T[N,K] X^T[K,M]
```

A compact no-transpose call is therefore:

```cpp
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K, &alpha, W, N, X, K, &beta, Y, N);
```

This snippet assumes tightly packed row-major inputs, FP32, valid device pointers and host scalar pointer mode. It is an API illustration, not a standalone compiled program. With padded views, use their actual row strides. PyTorch tensors can be noncontiguous; the language does not guarantee layout.

## 4. Run a non-square, padded example

Use the [lab guide](Lab-Guide.md). `column_major_gemm()` in [optimization.py](labs/optimization.py) implements the non-transposed operation with explicit leading dimensions and higher-precision Python arithmetic. The test uses:

\[
A=\begin{bmatrix}1&2&3\\4&5&6\end{bmatrix},\quad
B=\begin{bmatrix}7&8\\9&10\\11&12\end{bmatrix},\quad
AB=\begin{bmatrix}58&64\\139&154\end{bmatrix}.
\]

Stored A uses `lda=3`, B uses `ldb=4`, and C uses `ldc=3`. Sentinel padding is retained. Output data starts as NaN, `beta=0`, and the reference does not read old C values. The resulting flat output is `[58,139,123,64,154,123]`, where 123 marks untouched padding.

A second test sets `alpha=2,beta=3` for one-element operands `A=2,B=3,C=5`; the answer is `2*2*3+3*5=27`. With nonzero beta, valid initialized C is required. With beta zero, BLAS permits C's previous contents to be unused; a naive reference that evaluates `0*NaN` has different behavior.

A deliberately wrong leading dimension fails comparison. Non-square dimensions and padding expose mistakes that a 4×4×4 test can conceal. Extend this with two batch members containing different values and a gap between them, then verify both independently.

## 5. Validate numerical agreement without false guarantees

Compare to a separately implemented reference using

\[
|candidate-reference|\le atol+rtol|reference|.
\]

Choose tolerances for the input range, reduction length, input precision and accumulation path. Explicitly reject NaN/Inf outputs before taking maxima. `fmax(error,NaN)` can hide a broken result if used carelessly. Poison each independent implementation's output buffer before running it, so one successful implementation cannot leave values that make a later missing write look correct.

Check CUDA/BLAS return codes and asynchronous completion before comparing results. Use zeros, cancellation, dynamic range, irregular dimensions, transposes, padding, nonzero beta and multiple batches. The original square-only snippet lacked these checks and is replaced by the executable CPU reference rather than presented as a verified CUDA benchmark.

`CUBLAS_PEDANTIC_MATH` constrains numerical behavior; it does not force all APIs and algorithms into one reduction order. The [cuBLAS reproducibility contract](https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility) has a defined scope. Different algorithms/toolkits/devices need not be bit-identical. FP32 accumulation does not recover information already rounded out of FP16/BF16/TF32 inputs.

## 6. Map this back to Qwen

For 4B prefill with `T=512`, logical projection GEMMs have:

| Operation | `M,N,K` |
|---|---|
| Fused QKV | `6144,512,2560` |
| O | `2560,512,4096` |
| Fused gate/up | `19456,512,2560` |
| Down | `2560,512,9728` |
| Full vocabulary head for every token | `151936,512,2560` |

Generation runtimes may only compute logits at requested positions, so they need not execute the full last row for every prompt token. Quantized projections may use custom kernels rather than ordinary cuBLAS. A `torch.bmm` call is a strict batched-matrix operation, not simply an alias for all of `torch.matmul`'s broadcast semantics; see [PyTorch bmm](https://docs.pytorch.org/docs/stable/generated/torch.bmm.html).

For an optional hardware study, use the maintained [CUDA optimization course](../../Track%20A%20-%20GPU%20Infrastructure/Nvidia%20GPU/HPC%20Setup/CUDA-Advanced-Optimization/README.md) and your toolkit's official cuBLAS examples. Record exactly what compiled and what executed. This lesson executed the scalar layout tests and the tiny Transformers CPU model; it does not claim a cuBLAS GPU run.

**Acceptance:** hand-derive the padded result, explain every leading dimension, show a failing wrong-stride case, and state a numerical error contract. Then explain why two semantically identical GEMM forms can have different timing or floating-point results without making every discrepancy acceptable.
