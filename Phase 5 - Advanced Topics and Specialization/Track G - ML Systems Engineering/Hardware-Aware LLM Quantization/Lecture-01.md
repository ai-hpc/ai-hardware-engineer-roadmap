# Module 01 — Inference Physics: A Roofline Is a Model

**Collection:** [Hardware-Aware LLM Quantization](README.md) | **Previous:** [Previous lesson](README.md) | **Next:** [Next lesson](Lecture-02.md)

Before selecting a quantizer, identify the limiting resource at your workload's
batch size, context length and latency target. The same network can be limited by
weight traffic, KV/state traffic, compute, launch overhead or communication in
different regimes.

## 1. Follow one generated token

A decoder gathers an input embedding, applies its language layers, produces logits
and samples. Dense layers use their weights every relevant forward pass. MoE
layers select experts; multimodal encoders and external drafters run conditionally.
Small-batch dense decoding often streams weights too large to remain in cache,
which makes reducing their traffic valuable. This is a common operating point,
not a guarantee that every parameter is read exactly once per emitted token.

Prefill reuses weights across prompt positions and often has higher arithmetic
intensity. Short prompts, small operators, attention, recurrence and poor kernels
can still make parts of prefill memory- or latency-limited.

## 2. Derive arithmetic intensity

For a dense `[B,K] @ [K,N]` product, neglect input/output traffic and assume one
weight read per batch:

```text
FLOPs ≈ 2 B K N
weight bytes ≈ K N b
arithmetic intensity ≈ 2 B / b FLOP per byte
```

Here b includes the quantized payload and its scale overhead. These assumptions
break when metadata, padding, activation/state traffic or rereads are substantial.

| Weight representation | Leading bytes/weight | B=1 intensity |
|---|---:|---:|
| BF16 | 2 | 1 FLOP/byte |
| FP8, ignoring scales | 1 | 2 FLOP/byte |
| E2M1 + one 8-bit scale per 16 | 0.5625 | 3.556 FLOP/byte |

A roofline upper bound is `performance ≤ min(P_peak, BW_peak × intensity)`.
The ridge is `P_peak/BW_peak`, using the **actual compute path and accumulation
mode**. For illustrative BF16 peak 419 TFLOP/s and bandwidth 1792 GB/s, the ridge
is about 234 FLOP/byte. This does not mean every decode workload becomes
compute-bound at batch 234: total traffic, operator shapes and attainable compute
must be included. W4A16 and native W4A4 have different compute paths.

## 3. Predict with bytes, then measure the prediction error

If a step is strongly bandwidth dominated, `token/s ≈ BW_effective/B_token`.
Use an independently measured or explicitly assumed effective bandwidth. Fitting
`BW_effective = B_token × measured_token/s` to a run and “predicting” the same run
is circular validation.

**Worked hypothetical example:** 20 GB of read traffic at an assumed 1650 GB/s
predicts 82.5 token/s. Removing 2 GB predicts 91.7 token/s, an 11.1% gain, if
bandwidth and other costs remain unchanged. This is a testable estimate, not a
benchmark. State traffic, quantize/dequantize kernels and launch overhead can
prevent the gain.

A fuller per-step latency model is useful:

```text
T_step ≈ sum_operator_times + scheduling + communication
T_operator ≥ max(bytes/BW_peak, FLOPs/P_peak)
```

Some work overlaps and some does not. Whole-model effective rates do not tell you
which kernel to optimize without a trace.

## 4. Separate bytes on disk, in memory and on the bus

`1 GiB = 2^30 bytes`; `1 GB = 10^9 bytes`. NVIDIA memory tools often display MiB,
while bandwidth specifications use decimal GB/s. Preserve the displayed unit.
The historical case's 18.80 GiB becomes 20.19 GB. Multiplying it by a reported
81.6 token/s gives 1647 GB/s, but **resident bytes are not traffic**.

The [anatomy lesson](Lecture-04.md) illustrates why an inactive vision tower, a
gathered embedding and a draft component are not full weight reads on every text
token. Its 15.88 GB candidate streamed-weight estimate yields 1296 GB/s, or 72.3%
of peak. Neither product is a measured DRAM counter. Do not interpret the gap as
an established 22 token/s optimization opportunity.

The current target artifact differs from that historical build: its output head
is quantized and its built-in MTP removed. [Source evidence](Source-Evidence.md)
records the exact revisions and the difference.

## 5. Rank opportunities using costs and constraints

For each candidate record bytes accessed, kernel availability, expected numerical
change, task limits and runtime memory. A multiplicative “traffic × compressibility
× speed × tolerance” score can organize questions, but its terms are not independent
physical quantities and it is not a proven objective function.

| Candidate | Direct effect | What might change the conclusion |
|---|---|---|
| Quantize a dense output head | fewer vocabulary-projection bytes | shape, quality, scale overhead, supported kernel |
| Quantize gathered embeddings | smaller capacity, fewer row bytes | avoiding offload or enabling a larger batch |
| Quantize an external drafter | draft cost and agreement | target verification cost and rollback |
| Quantize KV | lower cache payload | context, scales, quality and recurrent state |

Preserving the desired task behavior is a constraint. Maximizing throughput alone
can reward a broken decoder that emits premature EOS or skips work.

## Practice, answers and completion

1. A matrix has 1024×1024 BF16 elements. At B=1, ignoring other traffic, it has
   2 MiB of weights and 2,097,152 FLOPs: intensity **1**.
2. Store its weights in block-16 FP4 with byte scales and one FP32 global scale.
   Payload is **589,828 bytes**, before headers/alignment, not exactly 0.5 MiB.
3. If that change selects a slower kernel, can it lose speed? **Yes**. Reduced
   storage does not guarantee lower execution time.

Submit one annotated roofline estimate, list its assumptions, and compare to a
measurement or mark the device measurement unexecuted. Explain a discrepancy
instead of forcing an arbitrary efficiency factor to make the numbers agree.
