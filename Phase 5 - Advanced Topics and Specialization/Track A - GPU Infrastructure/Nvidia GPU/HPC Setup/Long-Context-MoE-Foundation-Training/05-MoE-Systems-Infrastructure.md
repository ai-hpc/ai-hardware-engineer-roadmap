# Module 05 — MoE Systems and Infrastructure

**Parent:** [Long-Context MoE Foundation Training](README.md)

**One-line purpose:** Make MoE training fast at multi-node scale by understanding expert parallelism, all-to-all dispatch, capacity factor under real batch sizes, and where the communication cliff is.

**Prerequisites:** Module 04 (you can write a top-k MoE forward). HPC Setup [NCCL Deep Dive](../NCCL-Deep-Dive/README.md).

**Artifact:** An all-to-all dispatch micro-benchmark across NVLink and InfiniBand domains, plus a capacity-factor sweep at realistic batch size showing dropped-token rate and per-iter time.

---

## Why it matters

A correctly-implemented MoE FFN can be slower than a dense one if the all-to-all is bad. Production MoE training spends 20–60% of its time in dispatch and combine all-to-alls; on a multi-node setup with poor topology, that figure climbs above 80%. This module is the difference between "MoE worked in a notebook" and "MoE scales to 1024 GPUs."

---

## Mental model

### Expert parallelism (EP)

The natural way to host `E` experts across many GPUs: put `E / EP` experts on each EP rank. A token routed to expert `e` must travel to whichever rank owns expert `e`. After the expert runs, the result must travel back.

- **Dispatch**: each rank sends tokens to wherever their chosen expert lives.
- **Compute**: each rank runs the experts it owns on the tokens it received.
- **Combine**: each rank sends results back to the original token's home rank.

Both dispatch and combine are **all-to-all** collectives: every rank sends a chunk to every other rank.

### Two ways to lay out MoE in a parallel mesh

A modern MoE training mesh has 5–6 dimensions: DP × TP × PP × CP × EP (× ZeRO, sometimes). EP is usually:

- **Inside a node**: EP ≤ 8, using NVLink/NVSwitch. All-to-all is fast (~600+ GB/s effective per pair).
- **Across nodes**: EP > 8. All-to-all crosses InfiniBand or RoCE; effective bandwidth ~25–50 GB/s per link, an order of magnitude slower.

Cross-node EP is unavoidable when the model needs more experts than fit on one node. The design goal is to minimize the cross-node hop count.

### Capacity factor at scale

In Module 04 you saw `capacity = ceil(capacity_factor · T · k / E)` per expert. At training-realistic batch sizes (tens of thousands of tokens per micro-batch per EP rank), capacity overflow is rare with `capacity_factor = 1.25`. But if you let it drift:

- Too high (e.g. `2.0`): you waste memory and compute on padded slots for tokens that never arrive.
- Too low (e.g. `1.0`): tokens get dropped when load is uneven, which produces noisy gradients and drives the load-balancer harder, which can oscillate.

The right value is "the smallest one that gives <2% dropped tokens at your batch size with a healthy router." Tune empirically.

### All-to-all communication cost

For `T` tokens per rank, `k` top-k, `E` experts, EP ranks `P`, hidden size `H`:

- Bytes sent per rank per all-to-all: `T · k · H · 2` (bf16).
- Number of partner ranks: `P - 1`.
- Volume per all-to-all: `T · k · H · 2 · (P - 1) / P` per rank.

For `T = 8192, k = 2, H = 4096, P = 8`:

`8192 · 2 · 4096 · 2 · 7/8 ≈ 117 MB per rank per all-to-all`.

Two all-to-alls per MoE layer (dispatch + combine). With 32 MoE layers per pass: ~7.5 GB of all-to-all traffic per micro-batch per rank. On NVLink at ~500 GB/s effective, that's ~15 ms. On InfiniBand at ~25 GB/s, that's ~300 ms.

That ratio is the entire reason EP usually stays within a node.

### Reducing all-to-all cost

- **Topology-aware all-to-all**: NCCL chooses ring vs hierarchical based on `NCCL_ALGO`. For MoE dispatch, hierarchical usually wins. Set `NCCL_ALGO=Tree,Ring` and let it pick, or pin with `NCCL_ALL_TO_ALL_PIPELINE`.
- **Packing**: gather all tokens for the same expert into a contiguous buffer before sending. Megatron-LM does this. Without it, you send `k` separate chunks per destination.
- **Overlap with compute**: launch dispatch from layer `L`, then begin attention on layer `L+1` while dispatch is in flight. Megatron's `--moe-token-dispatcher-type alltoall` + careful CUDA graph capture gets this overlap.
- **Drop the all-to-all entirely** with shared experts: a small dense FFN that runs on every token regardless of routing, in parallel with the all-to-all. Used by DeepSeek-V3.

### MoE + TP

Tensor-parallelizing the expert FFNs is straightforward (TP slices the expert's hidden dim like any other linear). The subtlety: when TP and EP coexist, every token's chosen expert lives on a `(EP_rank, TP_group)` pair. Dispatch becomes a 2D collective. Megatron handles this; you set both `--tensor-model-parallel-size` and `--expert-model-parallel-size`.

A common production layout is `TP = 8 (intra-node), EP = N_nodes`. Each expert is itself TP-parallelized across the 8 GPUs of its EP rank's node.

### Expert load imbalance at runtime

Even with aux loss and z-loss, real workloads produce minute-to-minute imbalance — one expert spikes when a long code block enters a batch. The capacity-factor padding absorbs spikes; if spikes exceed capacity, tokens drop.

Two diagnostics you watch like a stock ticker:

- **Per-expert utilisation** (rolling): should hover near `1/E` ± 30%.
- **Dropped token rate**: should stay near 0%. Sudden spikes are usually data-pipeline issues (a shard of the dataset has unusual content) rather than the router's fault.

---

## Build it

### 1. All-to-all microbenchmark

Across 8 GPUs in one node (intra-node, NVLink):

```python
# alltoall_microbench.py
import torch, torch.distributed as dist, time
dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(rank)

for size_mb in [1, 4, 16, 64, 128]:
    n = size_mb * 1024 * 1024 // 2          # bf16 elements per rank-pair
    send = torch.randn(world, n, device="cuda", dtype=torch.bfloat16)
    recv = torch.empty_like(send)

    # warmup
    for _ in range(3):
        dist.all_to_all_single(recv, send)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(20):
        dist.all_to_all_single(recv, send)
    torch.cuda.synchronize()
    t = (time.perf_counter() - t0) / 20 * 1000

    total_bytes = world * n * 2
    bw = total_bytes * (world - 1) / world / (t / 1000) / 1e9
    if rank == 0:
        print(f"size/rank-pair={size_mb:>4} MB  time={t:7.2f} ms  bus_bw={bw:6.1f} GB/s")
```

Run on a single node and capture the times. Then run the same script on a 2-node setup (16 GPUs across InfiniBand). The intra-node bus bandwidth should be 4–8× the cross-node bandwidth.

### 2. Capacity-factor sweep

Take the `minimal_moe.py` from Module 04 and wire it into a multi-GPU training step with `torchrun --nproc-per-node 8`. Vary `capacity_factor ∈ {1.0, 1.1, 1.25, 1.5, 2.0}` at a realistic per-step token count (e.g. 16K tokens × 8 ranks = 128K tokens per step). Report:

- Dropped-token rate.
- Per-iter time.
- Final task loss after `N` warmup steps.

You should see `cf=1.0` produce 5–15% drops and noisy loss; `cf=1.25` produce 0–2% drops and stable loss; `cf=2.0` produce 0% drops but waste memory.

---

## Use it in the real stack

In Megatron-LM:

```
--num-experts 64
--expert-model-parallel-size 8
--moe-router-topk 2
--moe-router-load-balancing-type aux_loss
--moe-aux-loss-coeff 0.01
--moe-z-loss-coeff 1e-3
--moe-expert-capacity-factor 1.25
--moe-token-dispatcher-type alltoall
--moe-pad-expert-input-to-capacity
--use-flash-attn
```

A 64-expert, top-2 model on 64 GPUs (8-node 8×H200) would typically run with `EP=8` (intra-node) and `DP=8` (across nodes). The cross-node communication is then for gradient all-reduce, not for MoE all-to-all — much cheaper.

DeepSpeed-MoE has analogous knobs (`moe_param_group`, `ep_size`, `min_capacity`, `top_k`). Read both libraries' MoE docs once; the names differ but the concepts map 1:1.

---

## Measure it

For your sweep:

- **All-to-all bandwidth** at multiple message sizes, intra-node and cross-node. Plot.
- **Per-iter time breakdown**: forward, backward, all-to-all, all-reduce. Megatron's `--log-throughput --log-communication-volume` produces these.
- **Token traffic per iteration** in GB. Match against your back-of-envelope from the mental-model section.

A healthy MoE training step has:

- All-to-all under 25% of step time at EP=8 intra-node.
- Under 40% at EP=16 (one cross-node hop).
- Above 50% means you should either reduce EP, add overlap, or revisit topology.

---

## Ship it

Drop into `lcm-course/`:

1. `alltoall_microbench.py` and `alltoall.csv` with intra-node and cross-node results.
2. `capacity_factor_sweep.csv` with dropped-token rate, per-iter time, and loss after warmup.
3. `moe_systems_notes.md` — one paragraph each on expert parallelism, all-to-all costs, capacity factor in production, and at least one named failure you induced (e.g. "at cf=1.0 with 16K tokens/rank, dropped rate spiked to 12% during the first 200 steps, then settled to 2%").

---

## Related pages

- [Module 04 — MoE fundamentals](04-MoE-Fundamentals.md)
- [Module 08 — Combining long-context and MoE](08-Combining-LongContext-and-MoE.md)
- [Module 09 — Distributed training infrastructure](09-Distributed-Training-Infrastructure.md)
- [NCCL Deep Dive](../NCCL-Deep-Dive/README.md)
- Megatron-LM MoE: <https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/README.md>
- DeepSpeed-MoE: <https://www.deepspeed.ai/tutorials/mixture-of-experts/>
