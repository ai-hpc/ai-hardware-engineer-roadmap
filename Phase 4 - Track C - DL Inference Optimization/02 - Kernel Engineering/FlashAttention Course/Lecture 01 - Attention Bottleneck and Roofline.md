# Lecture 1 — Count attention work before drawing a roofline

**Course:** [FlashAttention](Guide.md) · **Reviewed:** 18 October 2026

**Goal:** distinguish computation, memory capacity and memory traffic. **Prerequisite:** matrix shapes and bytes per element. **Artifact:** a per-head budget with explicit assumptions, followed by a measurement plan.

## 1. Draw the tensors

For one head, `Q` has shape `[L,D]`, `K` is `[S,D]`, and `V` is `[S,Dv]`. Scores and probabilities have shape `[L,S]`; output is `[L,Dv]`. A multiply-add counts as two FLOPs, so the two dense matrix products cost:

```text
QKᵀ: 2 × L × S × D
PV:  2 × L × S × Dv
```

When all dimensions are `L=S=N` and `Dv=D`, this is `4N²D` FLOPs, plus softmax and masking work. Causal kernels can skip parts of the matrix; use the actual useful-pair count when reporting useful FLOPs. Padding and tile work can differ from that count.

For `N=8192`, one BF16 score matrix needs `8192²×2 = 128 MiB`. Holding scores and probabilities simultaneously needs 256 MiB just for those two matrices, before other tensors. That is a **capacity** statement.

## 2. Count each transfer

Under a deliberately simple materialized model—each input read once, output written once, and both intermediates written/read once—traffic is:

```text
Q,K,V,O: (LD + SD + SDv + LDv) × element_bytes
S/P:     4 × L × S × element_bytes
```

For square BF16 attention with equal head dimensions, total is `8N²+8ND` bytes. At large N, the intensity approaches `D/2` FLOPs/byte. This is an explicit model, not a measurement of every implementation.

The compulsory input/output traffic is only a **lower bound**. A tiled implementation can reread K/V for multiple query tiles. Linear auxiliary storage does not imply linear traffic or linear runtime. [The FA1 paper](https://arxiv.org/html/2205.14135v2) analyzes this distinction; Lecture 03 connects its memory parameter to a traversal.

## 3. Execute the counter

```bash
python3 'Phase 4 - Track C - DL Inference Optimization/02 - Kernel Engineering/FlashAttention Course/labs/attention_contract.py' --length 8192 --dim 64 --row-tile 64
```

[The counter](labs/attention_contract.py) reports materialized traffic, a query-outer traversal, compulsory input/output bytes and full dense FLOPs. Repeat at length 4096 and double the row tile. Predict which terms change.

With a fixed row tile, the query-outer K/V term is proportional to `N²`. Its traffic ratio to the materialized path tends toward a constant. Calling that ratio “speedup” would add a conclusion the model has not established. Cache reuse, launch costs, precision, parallelism and instruction throughput still matter.

## 4. Draw a roofline with a named device

The ideal roofline ceiling is `min(peak_compute, intensity × peak_bandwidth)`. Choose the exact GPU SKU, precision, dense/sparse convention, and power setting from its specifications. Use measured sustainable limits when available. A single aggregate intensity can hide a compute-heavy GEMM followed by a bandwidth-heavy softmax.

Do not infer that every naive implementation saturates HBM, every fused implementation is compute-bound, or every long-context benchmark must achieve a fixed fraction of peak. A roofline predicts a ceiling under assumptions; profiling tests the explanation. A float64 CPU correctness oracle is useful even when the deployed GPU kernel accepts FP16/BF16.

## 5. Practice and answers

1. **Does eliminating the score matrix eliminate its dot products?** No. Dense exact attention still computes the allowed Q–K interactions.
2. **Does a 128 MiB tensor cost one 128 MiB transfer?** Only for the stated operation. Writing it and reading it are separate transfers.
3. **Can you divide compulsory bytes by runtime and call it measured HBM bandwidth?** Call it an effective rate under that model. Actual traffic requires suitable device counters.

Finish with your assumptions, formula, two numerical rows and a reason the model could be wrong. Continue to the [executed CPU correctness lab](CPU-Attention-Lab.md); GPU timings are an extension, not a prerequisite for the mathematical path.
