# Lecture 9 — FA3, FA4 and the current hardware/software boundary

**Course:** [FlashAttention](Guide.md) · **Reviewed:** 18 October 2026

**Goal:** connect kernel pipelines to the right architecture and source family. **Artifact:** a dated source map; GPU benchmarks are a separate optional extension.

## 1. Pin the current state

At source revision `965c9212e6be33c821ee9f2b461564c5c91a2503` (16 October 2026), FA3 lives in `hopper/`; FA4 lives in `flash_attn/cute/`. FA4 is publicly installable as a CuTeDSL distribution; PyPI reports **4.0.0b35**, a prerelease. It is no longer accurate to call it only a blog post or an unpublished prototype. This is a source/metadata review, not a claim that the course ran those kernels. [Pinned repository](https://github.com/Dao-AILab/flash-attention/tree/965c9212e6be33c821ee9f2b461564c5c91a2503) · [FA4 metadata](https://pypi.org/pypi/flash-attn-4/json).

## 2. Explain the Hopper schedule

FA3 uses Hopper's asynchronous matrix and memory operations to overlap work. Producer/consumer roles can stage the next tile while computation uses another. Inter-warpgroup and intra-warpgroup schedules can overlap softmax work and matrix operations, subject to dependency and resource constraints. Lower-precision paths add quantization/scaling and numerical questions. [FA3 paper](https://tridao.me/publications/flash3/flash3.pdf).

This does not make softmax free or promise 95% utilization at every shape. Very short sequences, small batches, masked work, head dimension and register pressure can change the limiting resource. Distributed shared memory is a hardware capability; do not claim every FA3 pipeline necessarily uses cross-CTA exchange.

The historical Hopper implementation uses CUDA C++ and **C++ CuTe**. Its `cute::Tensor` and `cute::gemm` names are C++ library constructs. That does not make FA3 a Python CuTeDSL implementation.

## 3. Explain the Blackwell motivation without inventing FP4 support

FA4 targets the imbalance between faster matrix units and other resources. Its published design discusses overlapping matrix work, exponentials and data movement, tensor memory, conditional rescaling and a revised backward schedule. The highlighted BF16 work is not evidence that an arbitrary FP4/FP6 attention API is available. Check the actual current interface/dtype path. [FA4 author explanation](https://tridao.me/blog/2026/flash4/).

**Conditional rescaling question:** if a block changes the maximum only slightly, can an implementation keep another reference scale temporarily? It can if its invariant and exponent range remain valid and all contributions use that scale consistently. That is a distinct derivation from incorrectly deleting rescaling in the basic FA2 recurrence.

The pinned FA4 README describes Hopper and Blackwell targets and a separate CUDA 13 installation extra. “GPU supports FP8” alone does not establish that an SM90-specific FA3 kernel runs there; instruction set, build and kernel dispatch must match.

## 4. Trace source before copying an import

Use the exact selected build's interface:

```python
# Source-reviewed import surfaces; not executed in this CPU course run.
from flash_attn import flash_attn_func                     # FA2 package
from flash_attn_3 import flash_attn_interface as fa3        # separate Hopper package
from flash_attn.cute import flash_attn_func as fa4          # FA4 CuTeDSL package
```

Do not use the old fictitious `flash_attn_func_v3` import from the FA2 interface. Read each callable's signature and returned objects before sharing a benchmark adapter. Select separate environments when package namespace/build combinations conflict.

A source map should identify: operand layout, data movement, producer/consumer synchronization, matrix operation, normalization and output write. Use actual file:line references from your checkout; generic guessed names like `csrc/flash_attn_hopper/` are not a current source map.

## 5. Hardware extension and answer checks

Run an identical correctness fixture on the selected FA2/FA3/FA4 paths, then time the same semantics on a compatible device. Label dtype, scale, mask, forward/backward, warmup and timing boundary. Benchmark speedup is measured per workload; no universal 1.5–2× win or peak-utilization percentage is required for completion.

**No GPU route:** explain which dependencies allow a producer to reuse a buffer and why an incorrectly placed barrier can corrupt a mathematically valid recurrence. Then compare the published pipeline with the CPU lab's explicit numerator/statistic ownership.

**Diagnostic:** does current `flash-attn-4` installation metadata validate a Jetson Orin run? **No.** Orin's architecture and software stack require a separately supported kernel/build. Do not infer that support from the shared NVIDIA name.
