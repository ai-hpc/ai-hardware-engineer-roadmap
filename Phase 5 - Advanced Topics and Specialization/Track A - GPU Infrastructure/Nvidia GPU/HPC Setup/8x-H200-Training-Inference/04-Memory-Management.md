# 04 — Count memory by ownership and lifetime

**Outcome:** calculate a per-rank training/inference budget and explain an out-of-memory event. Source review: 2026-10-18. Capacity examples are arithmetic; no H200 run is claimed.

## Parameters are a count, bytes are a separate quantity

Let P be the **number of parameters**. In one common mixed-precision Adam configuration:

| State | Bytes per parameter |
|---|---:|
| BF16 working weights | 2 |
| BF16 gradients | 2 |
| FP32 master weights | 4 |
| FP32 first/second moments | 8 |
| Total persistent training state | **16** |

Other optimizers/implementations may omit a separate master copy or retain gradients/parameters differently. Record the actual policy. For P=70 billion, 16P is **1,120 GB**, before activations and workspace. Perfectly sharding these states eight ways leaves **140 GB per GPU**, already near the nominal 141 GB capacity. This does not establish that full 70B Adam training fits on eight H200s.

Full sharding divides the sharded states; it does **not** divide all local activations by the data-parallel world size. A better planning expression is:

`rank_peak = sharded_persistent_state + local_activations + live_gathered_parameters + gradients/buffers + workspace + reserve`.

Count only the non-overlapping portion of buffers when avoiding double counting, and take the maximum over time and ranks. [ZeRO](https://arxiv.org/abs/1910.02054), [FSDP2 controls](https://docs.pytorch.org/docs/2.14/distributed.fsdp.fully_shard.html).

## KV cache: use KV heads and actual retained tokens

For a conventional full-attention cache:

`bytes = 2 × layers × KV_heads × head_dim × retained_tokens_across_sequences × bytes_per_element`.

The first 2 counts keys and values. Query-head count is not KV-head count for grouped-query attention. With 80 layers, 8 KV heads, head dimension 128, 8,192 tokens for each of 32 sequences, and BF16, the total is **85.899 GB decimal (80 GiB)** before metadata. The old 42 GB figure omitted a factor of two.

```python
def kv_bytes(layers, kv_heads, head_dim, tokens, sequences, element_bytes=2):
    dimensions = (layers, kv_heads, head_dim, tokens, sequences, element_bytes)
    if any(not isinstance(v, int) or v <= 0 for v in dimensions):
        raise ValueError("all dimensions must be positive integers")
    return 2 * layers * kv_heads * head_dim * tokens * sequences * element_bytes

assert kv_bytes(80, 8, 128, 8192, 32) == 85899345920
```

Dividing this by TP assumes the implementation partitions KV heads/cache evenly. Some configurations replicate KV heads, and sliding-window, latent-attention, hybrid recurrent or compressed-cache models use different formulas. Use the checkpoint and runtime's actual cache layout.

Paged allocation reduces unused reserved slots, but still has block slack, metadata, reservations and runtime policies. For block size 16, a 200-token sequence needs `ceil(200/16)=13` blocks (208 slots); a 3,800-token sequence needs 238 blocks (3,808 slots). No fixed 2–4× concurrency improvement follows. Supported block sizes depend on backend/kernel/version. [PagedAttention paper](https://arxiv.org/abs/2309.06180).

## Measure allocator and device memory separately

```python
import torch

def memory_report(device=0):
    free, total = torch.cuda.mem_get_info(device)
    return {
        "device_total_GB": total / 1e9,
        "device_free_GB": free / 1e9,
        "torch_allocated_GB": torch.cuda.memory_allocated(device) / 1e9,
        "torch_reserved_GB": torch.cuda.memory_reserved(device) / 1e9,
        "torch_peak_allocated_GB": torch.cuda.max_memory_allocated(device) / 1e9,
    }
```

`total − torch_reserved` is not true free device memory: other processes, libraries and contexts allocate memory too. Reserved minus allocated can reflect useful cached blocks, not necessarily pathological fragmentation. Reset peak stats at a defined boundary and record initialization, first iteration and steady-state peaks separately.

Allocator settings have tradeoffs and backend/version support constraints. A per-process allocation fraction is a limit, not a guarantee that the remainder is enough for all other allocations. For vLLM pass its documented `gpu_memory_utilization` argument/CLI flag; do not assume an undocumented `VLLM_GPU_MEMORY_UTILIZATION` environment variable is honored. [PyTorch memory semantics](https://docs.pytorch.org/docs/2.14/notes/cuda.html#memory-management), [vLLM engine arguments](https://docs.vllm.ai/en/latest/configuration/engine_args/).

## Checkpointing and offloading trade work for space

Activation checkpointing stores selected boundaries and recomputes intermediates during backward. The saving depends on what was retained, checkpoint granularity and attention implementation. “Checkpoint expensive ops” is ambiguous: expensive-to-recompute operations may be better saved, while large cheap intermediates may be better recomputed. Validate gradients, RNG behavior and step time; there is no universal 33% overhead.

Optimizer CPU offload can keep optimizer moments on the CPU and transfer gradients/updated parameters, rather than copying every moment array across PCIe each step. Derive transfers from that implementation's schedule. PCIe Gen5 x16 has about 64 GB/s theoretical one-way throughput; a bidirectional total or Grace-Hopper C2C figure cannot be substituted into a one-way copy calculation. Host memory bandwidth, pinned buffers, CPU optimizer time and overlap can dominate. [DeepSpeed configuration](https://www.deepspeed.ai/docs/config-json/).

## Inspect a snapshot at the failure boundary

This diagnostic uses PyTorch's private snapshot API, which must be checked against the installed version:

```python
import torch

torch.cuda.memory._record_memory_history(max_entries=100000)
# Run the small reproducer here, including its failing allocation if possible.
torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
torch.cuda.memory._record_memory_history(enabled=None)
```

A PyTorch allocator snapshot does not include every library allocation. Pair it with device telemetry and all-rank context. Do not assume NCCL always consumes a fixed 300 MB. [PyTorch memory visualization](https://docs.pytorch.org/docs/2.14/torch_cuda_memory.html).

**Exercise:** a model has 10 GB weights, 8 GB cache, 2 GB workspace and 3 GB reserve. Quantizing only weights to 2.5 GB changes 23 GB to 15.5 GB, a **32.6%** total reduction, not 75%. Then double cache use: total becomes 23.5 GB. Explain why “quantization guarantees the larger batch fits” fails.

**Exit artifact:** a byte-accurate state table, a lifetime timeline, and the maximum measured rank peak. If hardware is unavailable, submit the checked arithmetic and an explicit list of unmeasured allocations.
