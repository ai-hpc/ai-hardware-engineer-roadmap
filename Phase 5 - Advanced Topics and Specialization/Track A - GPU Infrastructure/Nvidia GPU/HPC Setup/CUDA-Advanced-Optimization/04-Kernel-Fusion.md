# 04 — Kernel fusion: reduce traffic without changing the function

**Outcome:** count data movement, detect a masked-reduction bug, and compare a fused operation with its reference. Use the [CPU and CUDA exercises](labs/README.md).

## 1. Count reads and writes explicitly

Separate kernels communicate through device memory, but that traffic may be served by caches. “Every intermediate goes all the way to HBM” is an assumption to check, not a universal fact.

Consider GELU, dropout with a fixed externally supplied mask, and a residual add over `N` elements. Ignoring the mask, random-number state and scalar parameters:

| Step | Reads | Writes | Element transfers |
|---|---|---|---|
| GELU | input | temporary A | 2N |
| Dropout | temporary A | temporary B | 2N |
| Residual add | temporary B + residual | output | 3N |
| Separate total | | | **7N** |
| Fused version | input + residual | output | **3N** |

For `N=1000`, FP32 payload traffic is 28,000 versus 12,000 bytes: **7/3 ≈ 2.33× less estimated traffic**, not a guaranteed speedup. The residual input must be counted. Actual HBM traffic depends on caching, implementation, mask generation and memory layout.

Fusion can also increase register pressure, spills, synchronization or duplicated work. The best choice follows from measurement and correctness, not a universal “fusion always wins” rule.

## 2. Use the appropriate roofline

Arithmetic intensity is operations divided by bytes at a specified memory boundary. State dtype, shape, operation convention and which arrays move. The ridge point is the applicable compute ceiling divided by the applicable bandwidth ceiling.

For a dense square FP16 GEMM of size 1024, counting two FLOPs per multiply-add, a simplified once-read A/B and once-written C model gives:

```text
FLOPs = 2 × 1024³ = 2,147,483,648
bytes = 3 × 1024² × 2 = 6,291,456
intensity ≈ 341.33 FLOP/byte
```

This excludes reading an existing C, packing, extra workspace and repeated loads. Whether it is compute- or bandwidth-limited depends on the target's relevant dense precision ceiling and achieved traffic. Comparing dense work against a sparse peak without the required sparsity is invalid.

## 3. Preserve numerical and stochastic semantics

The CPU lab compares separate and fused GELU/dropout/add using the **same explicit dropout decisions**. That isolates fusion from a change in RNG sequence. It checks dropout probabilities in `[0,1)` and shape agreement.

Changing exact GELU into `x*sigmoid(1.702*x)` changes the function. Changing RNG mapping can change individual outputs while preserving a distribution; decide which equivalence your application requires. Training-mode dropout should not silently appear in an inference example.

The CUDA harness uses an even smaller deterministic pair, affine then ReLU, and a fused equivalent. It compares new inputs and irregular lengths. No speed claim follows from the smaller kernel count alone.

## 4. Mask reductions after the transformation

Suppose a row has values `[1,2,4]`, but the kernel operates on four lanes. Filling the inactive input lane with zero makes the sum correct if you divide by the valid length. It does **not** make the variance reduction correct after subtracting the mean.

```text
valid mean = 7/3
incorrect extra variance term = (0 - 7/3)²
```

The padded lane must contribute zero to the centered-square sum as well. The CPU lab reproduces this old error: the first normalized value changes from about `-1.06904` to `-0.72627` when the invalid lane contaminates variance.

A correct Triton-style expression for the reduction is:

```python
# Explanatory fragment: inside a kernel with a valid row and n_cols > 0.
x = tl.load(X_row + offsets, mask=mask, other=0.0).to(tl.float32)
mean = tl.sum(x, axis=0) / n_cols
centered = tl.where(mask, x - mean, 0.0)
variance = tl.sum(centered * centered, axis=0) / n_cols
```

This is a mathematical fragment, not a complete compiled Triton program. A full implementation still needs layout/stride checks, valid launch sizes, parameter loads, output casting and a numerical comparison. See the [official Triton layer-normalization tutorial](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html).

## 5. Compilers may fuse, and may choose not to

Triton is an open-source language/compiler project for parallel kernels, with backend-dependent hardware support. It is not simply “NVIDIA's Python compiler”. A logical Triton tensor does not promise that every element resides in a particular physical memory level.

PyTorch Inductor may fuse supported operations according to device, shapes, layout, scheduling and cost decisions. CPU and GPU paths can differ. `torch.compile` does not promise that LayerNorm, activation, dropout and residual add become one kernel or run a fixed number of times faster.

Use the [actual CPU compiler lab](../../../../../Phase%204%20-%20Track%20C%20-%20DL%20Inference%20Optimization/00%20-%20CPU%20Compiler%20Lab/Guide.md) to see a measured example of materialization, lazy fusion and constant folding. That lab found three versus one kernels for its own shapes and pinned tinygrad runtime; the number is evidence about that experiment.

For PyTorch inspection, use supported [compiler debugging tools](https://docs.pytorch.org/docs/stable/torch.compiler_troubleshooting.html) for your release. Preserve a generated-code excerpt and an uninstrumented measurement run.

## 6. FlashAttention changes the attention IO algorithm

For sequence length `N` and head dimension `d`, conventional attention may materialize an `N×N` score/probability array. At `N=8192`, one FP16 array is 134,217,728 bytes = 128 MiB **per head**. Whether arrays from multiple heads/layers are simultaneously live depends on execution and training/storage policy; multiplying by all layers does not automatically describe inference peak memory.

FlashAttention uses tiled attention and online softmax rescaling to avoid materializing the full quadratic matrix. It still performs dense attention's quadratic arithmetic, and its IO complexity depends on tile size and available fast memory. It is inaccurate to claim all HBM traffic becomes simply linear in sequence length for every memory size.

Read the [FlashAttention paper](https://arxiv.org/abs/2205.14135) and [FlashAttention-2](https://arxiv.org/abs/2307.08691). The algorithm is more specific than concatenating three kernels.

PyTorch scaled-dot-product attention dispatches among eligible backends according to device, dtype, shape, mask and other conditions. It does not guarantee FA2 or FA3 for every call. Inspect the selected backend and keep `dropout_p=0.0` explicit for inference where appropriate. See [SDPA documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html).

## 7. Runtime-specific fusion is version-specific

A historical `trtllm-build` plugin command is not a current universal TensorRT-LLM recipe. The project migrated to its PyTorch backend and documents [legacy TensorRT backend removal](https://nvidia.github.io/TensorRT-LLM/latest/legacy/tensorrt-backend-removal.html). Select the runtime release, model support and configuration before discussing which kernels it uses.

Similarly, a split can be a view rather than a launched kernel; RoPE normally transforms query/key representations rather than being an interchangeable fusion with input token embedding. Draw the actual graph and inspect executed work.

## Exercises and completion

1. Run the CPU lab with `N=1003`; explain 28,084 and 12,036 estimated bytes.
2. Use a five-element row in an eight-lane block and demonstrate the variance-padding failure.
3. Compare a fixed-mask fused operation against the separate reference. Explain what changes if the RNG mapping differs.
4. On a GPU, run the harness and then design a timed comparison. Report cache/traffic assumptions separately from measured counters.

Submit the arithmetic, counterexample, changing-input checks and validation level. Continue with [warp specialization](05-Warp-Specialization.md).
