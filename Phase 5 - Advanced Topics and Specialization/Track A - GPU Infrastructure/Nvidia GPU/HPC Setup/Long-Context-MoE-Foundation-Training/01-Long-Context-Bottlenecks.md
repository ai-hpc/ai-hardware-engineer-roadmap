# Module 01 — Count Long-Context Work and Memory

**Parent:** [Long-Context MoE Foundation Training](README.md) · **Practice:** [CPU training lab](CPU-Training-Lab.md)

**Learn:** distinguish forward arithmetic, stored tensors, peak training allocation, and inference cache. **Time:** 60–90 minutes. **Prerequisites:** matrix multiplication and the [FlashAttention course](../../../../../Phase%204%20-%20Track%20C%20-%20DL%20Inference%20Optimization/02%20-%20Kernel%20Engineering/FlashAttention%20Course/Guide.md).

A long sequence creates two questions: can the system process it, and does the trained model use distant evidence correctly? A larger configuration limit answers neither. This lesson addresses the first question; Modules 03, 06 and 07 address the second.

## Derive the forward arithmetic

Let `N` be sequence length, `H` hidden width, `hq` query heads, `hkv` KV heads, `d` head width and `F` FFN width. Assume `H=hq*d`, no biases, one multiply-add counts as two FLOPs, and a **nongated two-matrix FFN**. These are explicit modeling choices, not a description of every checkpoint.

| Component | Forward FLOPs per sequence, per layer |
|---|---:|
| Q projection | `2*N*H*H` |
| K and V projections together | `4*N*H*hkv*d` |
| Attention output projection | `2*N*H*H` |
| Attention QK and PV | `4*pairs*H` |
| Nongated FFN, up and down | `4*N*H*F` |

For dense bidirectional attention, `pairs=N²`. For causal attention, useful pairs are `N*(N+1)/2`. Actual tiled kernels may calculate masked or padded positions, so useful FLOPs can differ from issued instructions. Softmax, normalization, activations, routing and backward are additional work.

With ordinary multihead attention (`hkv=hq`) and `F=4H`, projections cost `8*N*H²` and the FFN costs `16*N*H²`. Their sum is `24*N*H²`; **24 is not the FFN coefficient**. Dense pair work exceeds the FFN near `N=4H`, or projections plus FFN near `N=6H`. For triangular causal work, those crossovers are approximately `8H` and `12H`. GQA and gated FFNs change them again. A threshold without these assumptions is not portable.

A gated FFN with two input projections and one output projection has approximately `6*N*H*F` FLOPs before elementwise gating. An MoE with `k` selected experts multiplies the expert term by `k`, then adds router and shared-expert work. Count the architecture you actually have.

## Count bytes once

For element width `b` bytes, these are **listed tensors**, not a complete autograd memory estimate:

| Tensor set | Bytes |
|---|---:|
| Q, K, V and attention output | `b*(2*N*H + 2*N*hkv*d)` |
| One FFN intermediate | `b*N*F` |
| One full materialized score matrix | `b*hq*N²` |
| Inference K and V cache | `2*b*N*hkv*d` |

Do not multiply a formula already expressed in BF16 bytes by two again. FlashAttention avoids materializing the full score/probability matrices in HBM, while still performing quadratic dense attention arithmetic. Its backward recomputes tile intermediates. [FlashAttention paper](https://arxiv.org/abs/2205.14135).

For `H=4096`, `hq=32`, `hkv=8`, `d=128`, `F=16384`, BF16 and one layer:

| N | Useful causal pair work, TFLOPs | Nongated FFN, TFLOPs | Q/K/V/O, GiB | One score matrix, GiB |
|---:|---:|---:|---:|---:|
| 4,096 | 0.1375 | 1.0995 | 0.078125 | 1 |
| 32,768 | 8.7964 | 8.7961 | 0.625 | 64 |
| 131,072 | 140.7386 | 35.1844 | 2.5 | 1,024 |
| 1,000,000 | 8,192.0082 | 268.4355 | 19.0735 | 59,604.6448 |

The [runnable ledger](labs/ledger.py) reproduces these quantities. TFLOPs here is an **amount of work**; TFLOP/s is a rate. GiB means `2³⁰` bytes; decimal GB means `10⁹`.

## From one tensor to a training run

Peak memory also includes weights, master weights where used, gradients, optimizer states, live saved activations, communication buffers, kernel workspaces and allocator fragmentation. The training framework may save or recompute different tensors. GQA reduces KV terms, not the query-head count of the full score tensor.

Do not multiply every activation by global batch size. A pipeline stage retains activations for the microbatches currently live in its schedule and for its local layers. Gradient accumulation reuses buffers; it does not necessarily keep all accumulated microbatches' activations. Conversely, multiplying only one tensor by all layers is not the whole training footprint. Use the framework estimator, then compare with measured allocated/reserved memory and an all-rank trace.

For inference, this example at `N=32768` uses `128 MiB` of BF16 KV per layer and request. Across 32 such layers that is `4 GiB`, before weights and workspaces. Training's saved tensors and inference's KV cache are different accounting problems.

## Practice and answers

Run the ledger through the CPU lab. Change only KV heads from eight to four. Predict which terms halve before reading the output: K/V projection work and KV storage halve; Q and O projections, query pair work and FFN work do not.

Then compare `causal=True` and `False`. At `N=3`, there are six versus nine useful query-key pairs; the causal result is two thirds of dense, not exactly one half. The ratio approaches one half for long sequences.

Finally estimate a training budget with a stated backward/recompute model and a **range** of achieved rates. A fixed 50% utilization is an assumption, not a measured property. Explain how padding, communication, optimizer work and input stalls could invalidate the estimate. Save the formulas, units, assumptions and one counterexample; this is the first artifact in your capstone.

[Next: attention and parallelism](02-Long-Context-Attention.md)
