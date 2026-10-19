# Part 3 · Lecture 03 — Expert parallelism: route, dispatch, and combine

Expert parallelism (EP) places different experts on different ranks. A token must reach each selected expert and the resulting outputs must be combined with the correct weights. Run the [CPU routing lab](../Part%201%20-%20Fundamentals/labs/README.md) before estimating fabric costs: it makes destinations, duplicate payloads, and rank imbalance explicit.

## Trace a two-rank example

Rank 0 owns experts 0–1; rank 1 owns experts 2–3. Token A starts on rank 0 and selects experts `[0,2,3]`. Token B starts on rank 1 and selects `[2,3]`. All vectors have hidden width 8 and two-byte elements, so each vector is 16 bytes.

An implementation sending one copy per remote token-expert assignment sends A twice to rank 1: 32 bytes. A destination-deduplicating implementation sends A once, then fans it out locally: 16 bytes. B stays local. Both perform the same five expert assignments; rank loads are `[1,4]`. Deduplication requires metadata and local bookkeeping, which the lab's byte count excludes.

The communication matrix's row is the source rank and column is the destination. Sum only remote entries for network activation payload. Combine traffic may differ because results can be locally weighted/summed before return, or sent per expert. State the implementation rather than blindly doubling one number.

## Uniform routing is an assumption, not a property

If E experts are evenly divided among P ranks and selected uniformly, an individual assignment is local with probability `1/P`. At P=8, expected local fraction is 12.5%, not 50%. For top-8 selections, expected remote assignments are seven per token under that model.

Several selected experts can share a rank. With E=256, 32 experts/rank, and 8 distinct uniformly selected experts, the probability that a particular rank receives the token is:

```text
1 - C(224,8)/C(256,8)
```

Multiply by seven remote ranks for expected distinct remote destinations. This is smaller than seven remote assignments. Actual DeepSeek routing includes grouping/scoring constraints; real traffic must be measured rather than assumed uniform.

Larger batches can improve grouped-GEMM efficiency but activate a larger union of experts and create hotspots. Report maximum tokens/rank and tokens/expert as well as averages. The slowest required rank can determine step latency.

## Use the actual communication API

NCCL 2.32.3 documents `ncclAlltoAll` with equal counts per peer. Variable-sized expert traffic can use grouped matched sends/receives or a specialized runtime backend; `ncclAlltoAllV` is not a generic API promised by that documentation. See [collective signatures](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html) and [grouped point-to-point patterns](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html).

[DeepEP](https://github.com/deepseek-ai/DeepEP) is a specialized expert communication project. Check its hardware, transport, build, and runtime integration requirements at a pinned revision. Its published benchmarks are workload-specific; this course does not assert an unmeasured 1.5–2× advantage.

## Build a latency model with visible assumptions

Separate router/top-k, packing, count exchange, dispatch, expert kernels, combine, and synchronization. `bytes/effective_bandwidth` is only a transfer lower bound. Small messages may be dominated by fixed overhead; irregular routes may underuse links; packing may contend with expert compute for SMs and memory.

For 64 tokens each sending a 7168-wide BF16 vector to seven remote assignments, one-way assignment payload is `64×7×7168×2 = 6,422,528 bytes` across the stated global batch. Do not divide that global total by a single rank's link rate and call it every rank's latency. Use the source/destination matrix and bottleneck paths.

TP, EP, data parallelism, and pipeline parallelism can use overlapping communicator groups depending on the runtime. `TP×EP×PP` is not a universal GPU-count identity. Draw a rank table and list each group's members. A claim about 64 ranks in a 72-GPU domain must explain the remaining allocation.

**Exercise.** Two settings have the same total assignments, but loads `[10,10,10,10]` and `[37,1,1,1]`. Why can the second be slower? **Answer:** a hot rank serializes more expert work and may receive more traffic; the mean load hides the critical path.

Deliver the route matrix, per-rank loads, payload assumptions, and observed trace if hardware is available. Next: [prefill/decode separation](Lecture-04.md).
