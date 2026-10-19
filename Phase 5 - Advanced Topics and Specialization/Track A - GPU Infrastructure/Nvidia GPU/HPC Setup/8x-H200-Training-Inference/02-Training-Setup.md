# 02 — Build a training setup that fits and preserves the objective

**Outcome:** choose a memory/parallelism plan and validate a small training update before scaling. Source review: 2026-10-18. No eight-H200 training was executed for this update; the linked CPU DDP lab is executed evidence.

## Start from the training state, not only weights

Under the explicit 16-byte-per-parameter mixed-precision Adam assumption in [memory management](04-Memory-Management.md), the persistent state estimates are:

| Parameters | BF16 weights | Full training state | Eight-way fully sharded state/rank, before other memory |
|---|---:|---:|---:|
| 7B | 14 GB | 112 GB | 14 GB |
| 13B | 26 GB | 208 GB | 26 GB |
| 70B | 140 GB | 1,120 GB | 140 GB |
| 180B | 360 GB | 2,880 GB | 360 GB |

These are not “fits” labels. Local activations, gathered units, workspace and runtime memory are additional. Offloading, another optimizer/state policy, lower precision or more devices may change feasibility. First test a small model whose full lifecycle—including first optimizer step—fits.

## Learn correctness on two CPU workers

Complete the [Gloo/DDP practical](../NCCL-Deep-Dive/CPU-Collectives-Lab/Guide.md). It checks SUM/AllGather, then compares distributed gradients and an SGD update with a serial reference. Unequal local batch sizes expose the difference between averaging ranks and averaging samples. Keep that distinction for masked-token losses and gradient accumulation.

On GPUs, use [the complete NCCL smoke test](../NCCL-Deep-Dive/04-Configuration-and-Tuning.md#a-complete-cuda-smoke-test) before a training framework. Select devices with `LOCAL_RANK`; global rank is not a local device index on multiple nodes.

## Choose one parallelism dimension for a stated reason

| Plan | What it buys | Main cost/constraint |
|---|---|---|
| DDP replicas | Independent batches with synchronized gradients | Each rank holds full model/optimizer state; gradient buckets communicate |
| FSDP / ZeRO-3 | Sharded persistent training state | Parameter gathers, gradient reductions, transient full units |
| Tensor parallelism | Shards large within-layer computations/weights | Intra-layer communication and dimension divisibility |
| Pipeline parallelism | Distributes layers | Boundary transfers, schedule bubbles and live activations |

For 8 GPUs, TP×PP×DP=8 in a simple three-dimensional design. TP=8 leaves DP=1, so claiming another eight-way FSDP reduction across the same **independent** DP dimension double counts devices. Hybrid arrangements need an explicit mesh and supported implementation. TP=4,DP=2 gives two data replicas; it does not inherently mean two microbatches.

## Current FSDP and historical wrappers

PyTorch's current composable path is `fully_shard` (FSDP2). Read the pinned version's initialization, mesh, mixed-precision and checkpoint APIs. Shard sensible module units so transient gathers fit; initialize the optimizer after applying the intended sharding transformation. Validate loss/update equivalence and save/load behavior before a large checkpoint.

The older `FullyShardedDataParallel` FSDP1 wrapper still appears in historical code. Its `transformer_auto_wrap_policy` is a callable configured with `functools.partial`, not called with only a layer-class argument as though wrapping occurs immediately. `ShardingStrategy` and FSDP1 state-dict recipes must not be mixed into an FSDP2 tutorial. [FSDP2 API](https://docs.pytorch.org/docs/2.14/distributed.fsdp.fully_shard.html), [FSDP1 API](https://docs.pytorch.org/docs/2.14/fsdp.html).

Loading a 70B checkpoint independently in FP32 on every CPU rank can exhaust host RAM before GPU sharding starts. Use the framework's supported distributed/meta initialization and checkpoint loading path, with a verified immutable checkpoint and host-memory budget. A local small-model reproduction is the prerequisite, not an optional afterthought.

## Loss, accumulation and optimizer sequence

Partition the data explicitly. Track valid samples/tokens across ranks and accumulation microbatches. The final short group needs its actual denominator. With DDP, `no_sync()` encloses both forward and backward of unsynchronized microbatches; synchronize the final one, clip the correctly reduced/unscaled gradients as appropriate, then step and clear gradients. FSDP clipping needs its supported global-norm semantics, not an arbitrary local-shard norm.

BF16 autocast does not guarantee all state is BF16, and FP8 hardware alone does not validate FP8 training. Use supported Transformer Engine/runtime recipes and compare loss, gradient statistics and held-out quality. Remove numerical assumptions from peak-throughput marketing figures.

## Save a restart you can prove works

A meaningful checkpoint includes model, optimizer, scheduler/scaler, RNG, step and data-position state, plus sharding/mesh metadata. Test an uninterrupted step against a save/restart step. Distributed checkpointing must publish a complete checkpoint consistently across ranks; a rank-0 weight-only file does not prove resumable sharded Adam training.

## Scale only after the small gates pass

1. A small serial model and distributed equivalent agree within an explicit numerical tolerance.
2. Training data partition, token/sample denominator and accumulation schedule are verified.
3. First optimizer step and checkpoint/restart pass within a measured per-rank memory budget.
4. Communication smoke tests and nccl-tests correctness pass on the allocated topology.
5. A short representative run reports valid tokens/s, quality/loss, step-time distribution and maximum rank memory.

For a robot-learning model, use grouped robot/episode splits and held-out environmental shifts; distributed training throughput does not fix data leakage or poor control quality. [DDP](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html), [distributed checkpoint](https://docs.pytorch.org/docs/2.14/distributed.checkpoint.html), [Megatron Core](https://github.com/NVIDIA/Megatron-LM).

**Exit exercise:** explain why a 70B model's 140 GB BF16 weights do not imply 70B training fits on one H200, and why eight-way state sharding still leaves little room under the table's 16-byte assumption. Submit a feasible model-specific plan and the smallest executed correctness evidence.
