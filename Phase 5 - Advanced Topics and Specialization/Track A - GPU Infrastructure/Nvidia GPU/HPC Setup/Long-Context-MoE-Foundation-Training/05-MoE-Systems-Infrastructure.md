# Module 05 — Dispatch, Combine and the Parallel Mesh

**Parent:** [Course](README.md) · **Before this:** [routing and capacity](04-MoE-Fundamentals.md)

**Learn:** follow an expert assignment across processes and back, derive a communication budget, and avoid counting parallel groups twice. **Practice:** [real two-process CPU dispatch](CPU-Training-Lab.md).

## Follow one token

Expert parallelism places different experts on different ranks. A source rank groups its selected assignments by destination. The dispatcher transfers inputs and required metadata; each destination computes its experts; the combine stage returns outputs and restores source-token order, applying the routing weights at the implementation's chosen stage.

An all-to-all exchanges different pieces with different ranks. It does not mean every token must visit every rank. Local assignments can remain local, destinations can receive zero entries, and counts may be uneven. Specialized implementations may replace generic collectives with more direct communication.

The runnable lab uses two real Gloo processes. Rank zero owns inputs `[1,2]`; rank one owns `[11,12,13]`. Expert zero computes `2x`, expert one computes `3x+1`. For the top-two fixture, each token has weights `.25,.75`, so the serial answer is `2.75x+.75`. Rank zero must recover `[3.5,6.25]`; rank one must recover `[31,33.75,36.5]`.

The code exchanges count vectors, sends packed inputs, runs the destination expert and performs the reverse exchange. Retaining the packing permutation is essential: numerically correct expert outputs returned to the wrong tokens are still wrong. A second fixture has an empty source/destination partition and skewed routing. Both are CPU correctness checks, not NCCL performance measurements. [PyTorch distributed semantics](https://docs.pytorch.org/docs/2.14/distributed.html).

## Derive bytes from assignments

Let each source rank have `T` tokens, top-k `k`, hidden width `H`, element width `b` and `P` expert ranks. Under uniform routing and equal ownership, a simple unpadded dispatcher sends approximately

`T*k*H*b*(P−1)/P`

remote payload bytes per source rank in one dispatch. The factor `(P−1)/P` removes the expected local fraction. Metadata, duplicated versus deduplicated destinations, padding, expert tensor parallelism and backward add implementation-specific costs. Combine often has a similarly sized output payload, but verify the shape and dtype.

For `T=8192,k=2,H=4096,b=2,P=8`, one dispatch sends `117,440,512` remote bytes, or `112 MiB`, per rank. Dispatch plus combine across 32 MoE layers sends `7,516,192,768` bytes per rank per forward microbatch in this model. At a **hypothetical exclusive effective path of 25 GB/s**, that takes at least `0.30065 s` if those transfers serialize on that path. It is not a measured step time; overlap, routing locality and shared links change the interpretation.

An NVSwitch fabric shares link and switch resources. A GPU's aggregate bidirectional NVLink specification is not a dedicated one-way rate to every peer. Likewise, a network port's line rate is not application payload throughput. Use measured topology and concurrency rather than assuming intra-node transfers are always four or eight times faster.

## Count dense and expert grids separately

At the inspected Megatron revision, with extra rematerialization factors set to one, the conventional PP-last cardinalities are:

```text
dense_DP  = world / (dense_TP * PP * CP)
expert_DP = world / (expert_TP * PP * EP)
```

The expert rank generator uses `CP=1`; it rearranges the same ranks into an expert grid. You do not multiply `world` by another independent EP axis. For 64 ranks, dense TP2, PP2, CP2, expert TP2 and EP4: dense DP is eight, expert DP is four. Eight experts give two experts per EP rank before expert tensor sharding. The [ledger](labs/ledger.py) checks these cardinalities; it does not construct rank placement or prove every feature combination legal. [Pinned parallel-state implementation](https://github.com/NVIDIA/Megatron-LM/blob/ce852bfb93f364f64a556602fc71a5d9ed2c377e/megatron/core/parallel_state.py).

Record actual group members. A valid arithmetic product does not show whether communication stays inside a node, whether pipeline groups align, or whether local tensors divide evenly. [Module 08](08-Combining-LongContext-and-MoE.md) develops the placement decision.

## Capacity and overlap are separate decisions

A high capacity factor allocates or computes padded slots only if that dispatcher/kernel pads to capacity. A dropless variable-size implementation makes a different trade. Large token batches do not guarantee that routing becomes balanced; a domain shift can concentrate assignments. Measure per-expert counts, admitted counts, zero-coverage tokens, task loss and allocated workspace.

Overlap is legal only when work is independent: eligible chunks, shared-expert computation or a supported interleaved schedule may overlap communication. The next layer generally needs the current layer's combined output. A flag named “overlap” does not remove this dependency or guarantee hidden communication.

Current Megatron provides generic all-to-all and specialized flexible dispatch paths. Their environment, capacity, precision, recomputation and parallelism restrictions differ. Start from a supported pinned recipe; inspect validation errors and traces instead of stacking all performance flags. [MoE implementation guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html).

## Take the exercise to GPUs

First finish the CPU dispatch and the [sample-weighted distributed-gradient lab](../NCCL-Deep-Dive/CPU-Collectives-Lab/Guide.md). The latter demonstrates why averaging unequal rank-local losses can change the objective.

For a hardware benchmark, bind the GPU using `LOCAL_RANK`, not global rank: global rank eight is not local GPU eight on an eight-GPU second node. Fix input/output split counts and payload contents, check results before timing, warm up, synchronize the device at measurement boundaries and report the slowest participating rank. Label bytes per peer versus bytes per source, and distinguish decimal MB from MiB. Do not call every derived throughput “bus bandwidth.”

Retain a correctness result, raw per-rank times, topology, precision, traffic definition, warmup and failure records. Compare a real routed training step with the communication-only microbenchmark; the latter omits routing, permutation, expert GEMMs and backward. No universal 25% or 40% communication fraction defines a healthy MoE.

**Exit answer:** improving an all-to-all benchmark can leave training unchanged if another stage dominates or the benchmark's split sizes/locality differ. Demonstrate that reasoning with one measured trace or an explicitly labeled dependency model.

[Next: adaptive data](06-Adaptive-Data-Pipelines.md)
