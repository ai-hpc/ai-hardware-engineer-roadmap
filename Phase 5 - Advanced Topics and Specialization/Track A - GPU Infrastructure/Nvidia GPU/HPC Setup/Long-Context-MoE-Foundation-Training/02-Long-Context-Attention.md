# Module 02 — Attention, Sequence Parallelism and Context Parallelism

**Parent:** [Course](README.md) · **Before this:** [memory ledger](01-Long-Context-Bottlenecks.md)

**Learn:** identify which tensor each technique partitions, then test correctness before interpreting throughput. **Time:** 90 minutes for CPU reasoning; cluster experiments are a separate milestone.

## Start with exact attention

For one query row, attention forms scores, normalizes them with softmax, and computes a weighted sum of values. FlashAttention tiles this calculation and maintains a running maximum, normalization sum and output accumulator. It avoids storing the full score/probability matrices in HBM. It does not make full attention linear in sequence length, and mathematical equivalence does not imply bit-identical floating-point results across schedules.

Run the existing [CPU attention lab](../../../../../Phase%204%20-%20Track%20C%20-%20DL%20Inference%20Optimization/02%20-%20Kernel%20Engineering/FlashAttention%20Course/CPU-Attention-Lab.md). Its dense oracle, tiled reference, gradients, masks and rectangular-query cases are the prerequisite for distributing attention. A GPU kernel is useful only after those semantics are fixed.

With SRAM capacity `M` **in elements**, the original FlashAttention analysis gives an HBM-access term of order `N²*d²/M`, within its stated memory regime, alongside linear input/output terms. This is an I/O model, not total FLOPs or a wall-time prediction. Tile choices, dtype, head width and hardware matter. [Original paper](https://arxiv.org/abs/2205.14135), [FlashAttention-2](https://arxiv.org/abs/2307.08691).

## Three different partitions

| Technique | What it distributes | Consequence |
|---|---|---|
| Tensor parallelism, TP | Parts of matrix products/heads | Partial results require collectives; local weight and activation shapes change |
| Megatron sequence parallelism, SP | Sequence slices of selected activations such as normalization/dropout around TP operations | Removes their replication across TP ranks |
| Context parallelism, CP | Sequence positions across the network | Attention needs remote keys/values; token-local operations use local positions |

Megatron SP uses all-gather/reduce-scatter around TP computations. Those replace or restructure communication that a non-SP TP path already performs; saying that SP always adds two entirely new collectives is misleading. SP can reduce saved activations, but its performance and legality depend on the implementation and shape. [Sequence-parallel/recomputation paper](https://arxiv.org/abs/2205.05198).

With CP, each query partition visits the needed KV partitions. For a basic ring with `P` ranks, each rank initially owns approximately `N/P` positions. A KV transfer contains roughly `2*(N/P)*hkv*d*b` bytes, and a full sweep visits up to `P−1` remote chunks. This differs from charging a full `N*H` tensor **per round**. Backward, saved state, masking, replication and specific algorithms change the complete ledger. [Megatron CP](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html).

Causal attention creates imbalance if one rank gets only early queries and another gets late queries. Balanced chunk assignments, striped layouts or specialized schedules can help. They must retain correct absolute positions and document boundaries. Summing independently normalized chunk outputs is wrong: merge their maxima, normalization sums and weighted accumulators consistently.

Ulysses begins with sequence shards, then uses all-to-all to exchange sequence partitioning for head partitioning during attention, and reverses that exchange afterward. It is not simply “head parallelism instead of sequence parallelism.” Head/KV divisibility and replication rules constrain the usable group size. [DeepSpeed Ulysses](https://www.deepspeed.ai/tutorials/ds-sequence/).

## Recomputation is a measured trade

Checkpointing keeps selected activations and recomputes omitted values during backward. Full-layer recomputation and selective module recomputation save different tensors. Once attention already uses a memory-efficient fused kernel, recomputing its core again may offer less benefit than recomputing FFN activations or model-specific projections. Fixed “20–30% memory savings” and “10–15% overhead” are not universal outcomes. [Bridge recomputation guide](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/activation-recomputation.html).

The inspected Megatron revision is `ce852bfb93f364f64a556602fc71a5d9ed2c377e`, dated October 18, 2026. Its argument validation rejects combining selective granularity with a recompute method such as `block`. These are **configuration fragments**, not complete launch commands:

```text
# selective attention recomputation
--recompute-granularity selective
--recompute-modules core_attn

# alternatively, full layer recomputation
--recompute-granularity full
--recompute-method block
--recompute-num-layers 1
```

Check the selected recipe/backend's additional restrictions. Keep the [pinned argument and configuration hashes](labs/sources.json); the live documentation and checkout can differ. A fake dataset path `mock` and an unspecified tokenizer do not create an executable training setup.

## Design the cluster experiment

Start from a supported recipe with a working tokenizer and a small valid dataset. Save container digest, repository revision, model dimensions, precision, batch/token counts, packing rules and group membership. Verify a short forward/backward run and checkpoint reload before extending context.

On eight GPUs, `TP=8,CP=1` and `TP=4,CP=2` are both eight-rank dense model groups when PP and DP are one. The change alters both TP and CP; it is not an isolated CP experiment. Keep global useful tokens and optimizer updates comparable, and explain the changed local matrix shapes.

Record peak allocated/reserved memory per rank, full-step latency, useful tokens per second, numerical/quality checks and communication traces. An OOM is a result; leave its throughput blank. A CP configuration may still OOM, and communication may remain exposed. Do not fill a table with promised improvements.

**Self-check:** why can a correct chunked forward still train incorrectly? Backward communication, gradient reductions, masks or dropout/RNG replay may differ. Forward parity is one gate, not the entire training contract.

[Next: positional encoding](03-Position-Encoding.md)
