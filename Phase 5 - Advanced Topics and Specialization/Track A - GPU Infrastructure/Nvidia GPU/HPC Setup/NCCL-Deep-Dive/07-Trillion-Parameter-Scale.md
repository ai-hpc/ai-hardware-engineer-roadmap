# 07 — A capacity and communication budget at large scale

**Outcome:** build a consistent paper design and identify what must be measured. The **1-trillion-parameter model below is a hypothetical dense model**, not a claimed reconstruction of a proprietary training run. No large cluster execution was performed. Source review: 2026-10-18.

## First see whether the state fits

One trillion BF16 parameters occupy **2 TB decimal** before metadata. Eight 141 GB H200 GPUs provide 1.128 TB nominal capacity; weights do not fit in that node even before activations. Training also needs gradients and optimizer state. A common mixed-precision Adam accounting is 2-byte weights + 2-byte gradients + 4-byte master weights + two 4-byte moments = **16 bytes/parameter**. Actual implementation choices differ, so list them explicitly.

Choose TP=8, PP=8, DP=4: `8×8×4=256` GPUs. A model-parallel rank owns P/(TP×PP)=15.625 billion parameters. Its BF16 parameter shard before data-parallel sharding is **31.25 GB**. Full sharding across DP=4 reduces persistent 16-byte state to `16P/256=62.5 GB` per GPU under this simplified model. That leaves room to investigate activations, gathered parameters, communication buffers, workspace and fragmentation; it does not prove the run fits.

**Check:** dividing 2 TB by all 256 GPUs gives 7.8125 GB of persistent parameter bytes. This is not the input size of every gradient collective. Use the owning model-parallel shard and the actual DP operation.

## A rank map you can verify without a cluster

Let `rank = tp + TP×pp + TP×PP×dp`. TP groups share pp/dp; PP groups share tp/dp; DP groups share tp/pp. With TP=8, rank 73 has coordinates `(tp=1, pp=1, dp=1)`.

```python
TP, PP, DP = 8, 8, 4
rank = 73
tp = rank % TP
pp = (rank // TP) % PP
dp = rank // (TP * PP)
tp_group = [t + TP * pp + TP * PP * dp for t in range(TP)]
pp_group = [tp + TP * p + TP * PP * dp for p in range(PP)]
dp_group = [tp + TP * pp + TP * PP * d for d in range(DP)]
assert (tp, pp, dp) == (1, 1, 1)
assert tp_group == list(range(72, 80))
assert pp_group == [65, 73, 81, 89, 97, 105, 113, 121]
assert dp_group == [9, 73, 137, 201]
```

When creating process groups, follow the framework's required consistent creation order across participating processes. Point-to-point pipeline traffic still uses communicator/group context; “PP has no communicator” is misleading. A logical map also needs a physical node/NIC placement map.

## Derive traffic per operation, then per rank

### Data-parallel full sharding

For the 31.25 GB model-parallel parameter shard, DP=4 and a simple schedule of two parameter AllGathers plus one gradient ReduceScatter, BF16 throughout:

`outgoing bytes/rank ≈ 3×31.25×(4−1)/4 = 70.3125 GB/step`.

This assumes each parameter is gathered twice and reduced once, with equal splits and no persistence/reuse. Actual wrapping and accumulation change it. A single exclusive 25 GB/s path would need at least **2.8125 seconds** for that outgoing volume, ignoring latency and other traffic. Multiple ranks sharing one NIC cannot each claim its full bandwidth simultaneously.

### Tensor-parallel activations

For an illustrative activation shape B=4, S=2048, H=8192 at BF16, the payload is `4×2048×8192×2 = 134,217,728 bytes`. An 8-rank ring AllReduce sends `1.75×payload = 234,881,024 bytes/rank`. At an ideal 450 GB/s one-way budget this is **0.522 ms**, not 0.3 ms. Multiply by calls on **that rank's stage**, microbatches and forward/backward schedule. These activation dimensions are a separate arithmetic example, not a complete architecture for the hypothetical trillion-parameter model.

### Pipeline boundaries

A chain of 8 stages has **7 boundaries**. One boundary sends a forward activation and a backward gradient per microbatch. `7 boundaries × M microbatches × 2 directions` counts transfers across the **whole pipeline**, not traffic on every GPU. A rank normally touches only adjacent stage boundaries, and tensor/sequence sharding changes each transferred shape.

For balanced stages under a simple fill/drain model, idle fraction is `(PP−1)/(M+PP−1)`. PP=8, M=16 yields 7/23≈30.4%. A schedule that runs all forwards before all backwards is **GPipe-style**, not 1F1B; 1F1B interleaves forward/backward after warmup and reduces live activations. Increasing microbatches changes batch size or per-microbatch work; it is not free. [GPipe](https://arxiv.org/abs/1811.06965), [interleaved model parallel training](https://arxiv.org/abs/2104.04473).

## Expert parallelism adds a routing contract

MoE dispatch sends tokens to selected expert owners and returns expert outputs. A correct implementation must preserve token IDs, top-k assignment, routing weights, per-destination counts and inverse permutation. Unequal splits require matching send/receive split metadata; a zero-filled tensor and two AllToAll calls do not constitute a router.

Test dispatch/return on tiny labeled tokens before expert compute. For token t routed to experts a/b with weights 0.25/0.75, return `0.25 fₐ(t)+0.75 fᵦ(t)`, even when packed buffers change order. Capacity dropping and padding must be explicit in both quality and traffic accounting. Experts may themselves be tensor-parallel. [Switch Transformers](https://arxiv.org/abs/2101.03961).

## Shared resources limit overlap

If PP and DP compete for the same NIC, their bandwidth demands add over the interval; `max(T_PP,T_DP)` is not a sufficient lower bound. Independent links and useful compute may overlap, but dependencies, memory/SM contention and collective launch costs remain. Build a resource timeline with GPU compute, HBM traffic and each NIC's bytes. Measure the slowest rank and end-to-end step, not only the mean rank.

For a dense model, a rough model FLOP utilization calculation is `6×P×global_training_tokens / (step_seconds×GPU_count×dense_peak_FLOP_per_GPU)`. State whether attention, recomputation and padding are included. MoE needs active computation and routing-aware accounting; total parameter count alone overstates useful FLOPs.

## Read published runs without inventing missing facts

Use the [Llama 3 technical report](https://arxiv.org/abs/2407.21783) for its documented architecture, scale and training method, and preserve its date. A source's GPU count does not establish this course's per-collective latency. Unverified GPT-4 parameter counts, expert counts and training-layout guesses are not evidence and are not used as worked baselines here.

**Capstone:** submit parameter/state arithmetic, rank groups, one physical placement, per-rank and per-node traffic, a peak-memory timeline, and three measured quantities needed to accept or reject the design. Demonstrate the tiny rank-map assertions and the [CPU correctness lab](CPU-Collectives-Lab/Guide.md). Mark cluster timings unexecuted until a run artifact exists.
