# 03 — Where frameworks put collectives

**Outcome:** predict communication from parameter ownership and verify gradient normalization. Run the [CPU DDP practical](CPU-Collectives-Lab/Guide.md) first. Framework APIs/source reviewed 2026-10-18; only that CPU/Gloo practical was executed here.

## DDP: replicated parameters, averaged gradients

Each DDP process owns a full model replica and its optimizer. Hooks mark gradients ready during backward; ready buckets can reduce while remaining backward computation continues. The default reducer averages gradients across ranks. `dist.all_reduce(..., SUM)` alone only sums; a custom communication hook must implement its own scaling correctly.

For local sample counts nᵣ and per-sample losses ℓᵣᵢ, the intended objective is:

`L = (Σᵣ Σᵢ ℓᵣᵢ) / (Σᵣ nᵣ)`.

Since default DDP averages rank gradients, each rank should backpropagate `world_size × local_loss_sum / global_sample_count` when batches differ. Simply averaging local mean losses weights each rank equally. For language modeling the denominator is the number of valid predicted tokens, not padded sequence length. The lab compares both choices against one serial batch and checks the subsequent optimizer update. [PyTorch DDP contract](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html).

DDP does not split your input automatically: use an appropriate sampler/partition and update the sampler epoch when shuffling. For gradient accumulation, `no_sync()` encloses forward and backward; all ranks still need compatible synchronization schedules. A global rank selects a communication identity, while `LOCAL_RANK` selects the process-local CUDA device.

## FSDP: gather what the next computation needs

With fully sharded parameters, a typical unit does:

1. AllGather its parameter shards before forward.
2. Run forward; optionally reshard afterward.
3. AllGather again before backward if parameters were resharded.
4. Compute gradients and ReduceScatter them to shard owners.
5. Update local parameter/optimizer shards.

This is a conceptual full-reshard schedule, not “exactly three calls per transformer layer.” Wrapping granularity, prefetching, mixed dtypes, parameter reuse, root behavior and `reshard_after_forward` change communication and peak memory. Keeping gathered parameters can save communication while using more memory.

The current PyTorch composable API is `torch.distributed.fsdp.fully_shard` (FSDP2), with DTensor-based parameters and explicit mesh/resharding controls. `FullyShardedDataParallel` is the earlier FSDP1 wrapper; its `ShardingStrategy` options and checkpoint recipes must not be mixed into an FSDP2 example. Start from a supported accelerator example for the pinned framework release; this chapter does not claim a CPU FSDP deployment. [FSDP2 API](https://docs.pytorch.org/docs/2.14/distributed.fsdp.fully_shard.html).

## ZeRO: distinguish persistent memory from communication

| Stage | Persistent states sharded | Typical communication consequence |
|---|---|---|
| 1 | Optimizer states | Reduced gradients reach optimizer owners; updated parameter portions must reach replicas |
| 2 | Optimizer states and gradients | ReduceScatter can leave each owner its gradient shard; parameter synchronization remains |
| 3 | Optimizer states, gradients and parameters | Parameters must be gathered when a module needs them, plus gradient reductions |

Exact implementations may use bucketed collectives or other equivalent schedules. ZeRO-3 is primarily a capacity tool; it can add traffic relative to replicated DDP. Offloading trades GPU capacity for host/storage traffic and latency.

DeepSpeed's `reduce_bucket_size`, `stage3_prefetch_bucket_size` and persistence threshold are specified in **elements**, not bytes. A 500,000,000-element BF16 buffer is about **1 GB**, before other buffers. Begin with a small verified configuration and profile peak memory. The following is a JSON configuration fragment, not an entire training script:

```json
{
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": 50000000,
    "stage3_prefetch_bucket_size": 50000000,
    "stage3_param_persistence_threshold": 100000
  }
}
```

[DeepSpeed configuration reference](https://www.deepspeed.ai/docs/config-json/), [ZeRO paper](https://arxiv.org/abs/1910.02054).

## Tensor parallelism: derive the reduction from matrix algebra

Use the row-vector convention `Y=XW`. Split columns of `W∈R^(H×3H)` into N pieces. Each rank computes `Yᵣ=XWᵣ` with shape `[B,S,3H/N]`; concatenating the pieces reconstructs Y. For a row split of an output projection, partition both X's last axis and W's first axis. Each rank computes a partial **sum** of shape `[B,S,H]`; AllReduce SUM reconstructs the output. Bias is added once after summing, or split in an explicitly equivalent way.

A simple tensor-parallel transformer can have two forward AllReduces per layer, one after attention output and one after the MLP output. Backward has additional communication. Sequence parallelism changes some patterns to AllGather/ReduceScatter. Attention-head counts, grouped-query attention, expert layout and framework implementations further constrain partitions. [Megatron Core tensor-parallel source](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/layers.py).

`async_op=True` returns a work handle; it does not make consuming unfinished data safe. Wait for completion or establish the documented stream dependency before using the output. There is no automatic overlap if the next operation needs that result. Overlap DDP's ready buckets with independent **remaining backward work**; ordinary synchronous SGD cannot use the next forward's updated parameters before the optimizer has the reduced gradients.

## Plan a trace you can interpret

Record a whole training step and identify parameter gathers, gradient reductions, compute, optimizer work and stalls. A timer that synchronizes after each region changes the schedule and removes some overlap. Use it for isolated costs, with a separate end-to-end measurement. Larger buckets reduce launch count but can delay their first communication; more prefetch can increase memory peaks.

**Worked check:** eight ranks, 100 million BF16 parameters per wrapped unit: one full parameter array is 200 MB. Each rank retains 25 MB if equally sharded, but a full gather materializes approximately 200 MB for that unit. Two gathers plus a ReduceScatter have a ring-normalized outgoing volume near `3×200 MB×7/8=525 MB` per rank under the simple schedule. Saved persistent storage is not the same as saved network volume.

**Exit exercise:** draw a 3-unit full-reshard schedule, mark which tensors are live, and explain one legal overlap and one illegal data dependency. Submit the CPU lab's equal/unequal-batch evidence plus a prediction for your future GPU trace.
