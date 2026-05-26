# Module 08 — Combining Long-Context and MoE

**Parent:** [Long-Context MoE Foundation Training](README.md)

**One-line purpose:** Design a parallel-mesh layout that combines tensor, sequence, context, and expert parallelism cleanly, predict where the communication bottleneck moves as you scale either context length or expert count, and pick the trade-off that matches your hardware.

**Prerequisites:** Modules 02 and 05. Comfort with NCCL collectives and bandwidth math.

**Artifact:** A decision table for a realistic 64×H200 cluster choosing TP × SP × CP × EP × DP for one model shape, plus a communication-cost breakdown by collective type.

---

## Why it matters

Long context and MoE both have substantial communication costs. Combined naively, those costs interact: more EP means more all-to-all volume per layer; more CP means more ring-attention rounds per layer; both want NVLink. If you do not lay out the mesh deliberately, you will spend most of your training time in NCCL.

---

## Mental model

### The full parallel-mesh dimensions

In a modern Megatron-style trainer:

| Dimension | Splits along | Dominant collective | Goes well on |
|-----------|--------------|---------------------|--------------|
| Data (DP) | batch | all-reduce (gradients) | any link, hidden by overlap |
| Tensor (TP) | hidden | all-gather + reduce-scatter per matmul | NVLink only |
| Sequence (SP) | sequence (non-matmul) | piggybacks on TP collectives | NVLink only |
| Context (CP) | sequence (attention) | ring all-gather of KV | NVLink for `CP ≤ 8`; InfiniBand viable for ring at larger CP |
| Expert (EP) | experts | all-to-all (dispatch + combine) | NVLink only; cross-node costs explode |
| Pipeline (PP) | layers | point-to-point activations | any link, hidden by overlap |

The constraint everyone hits: TP, EP, and CP all want NVLink; intra-node NVLink width is finite. You must choose which two of TP/EP/CP get to share the node.

### Communication volume by dimension (per layer, per micro-batch)

Let `T = tokens per rank per layer` (sequence length × micro-batch).

- **TP all-gather of activations**: `T · H · (TP - 1) / TP` bytes per matmul (typically 4 matmuls per layer).
- **CP ring of KV**: `T_local · 2 · H_kv · D · (CP - 1)` bytes per layer for the ring step.
- **EP all-to-all (dispatch + combine)**: `T · k · H · 2` per all-to-all × 2 per layer.
- **DP all-reduce of gradients**: `model_params · 4` bytes per step (fp32 grad accumulator), amortized over `micro_batches_per_step`.

For a 32K-context, 64-expert, 8B-class model at `T = 32K, H = 4096, k = 2, H_kv = 8, D = 128, layers = 32`:

| Collective | Bytes / layer / micro-batch | Total / micro-batch |
|------------|------------------------------|----------------------|
| TP all-gather | ~31 MB × 4 = 124 MB | 4.0 GB |
| CP ring (CP=4) | ~24 MB | 0.77 GB |
| EP all-to-all (EP=8) | ~104 MB × 2 = 208 MB | 6.7 GB |
| DP all-reduce | (model_params × 4) / micro_batches | varies |

EP all-to-all dominates. CP is second-tier. TP is well-overlapped with compute. This pattern holds for most realistic configs.

### Mesh layout recipes

#### Single 8×H200 node (no cross-node)

| Model size | Sequence | Layout |
|-----------|----------|--------|
| 7B dense | 32K | TP=8, SP, no CP, no EP |
| 8×7B MoE | 32K | TP=4, EP=2, SP |
| 7B dense | 128K | TP=4, CP=2, SP, recompute |
| 8×7B MoE | 128K | TP=2, EP=2, CP=2, SP (tight; consider FP8) |

#### 8-node × 8 = 64×H200 cluster (NVSwitch intra-node, InfiniBand inter-node)

| Model size | Sequence | Layout |
|-----------|----------|--------|
| 70B dense | 32K | TP=8 intra-node, PP=4 across nodes, DP=2 |
| 70B dense | 128K | TP=8, PP=4, CP=2 (CP rotates within node), DP=1 |
| 64-expert 8×7B MoE | 32K | TP=4, EP=2 intra-node, DP=8 across nodes |
| 64-expert 70B MoE | 32K | TP=8, EP=8 (each intra-node), PP=4 across nodes, DP=2 |
| 64-expert 70B MoE | 128K | TP=8, EP=8, PP=4, CP=2 — communication-tight, consider reducing EP to 4 |

The principle: **cross-node hops should serve PP and DP, not EP or CP**. PP and DP are well-overlapped; EP and CP are not (yet).

### Where the bottleneck moves as you scale

Hold model size fixed and scale **context**:

- 4K → 32K: TP+SP is enough.
- 32K → 128K: introduce CP; activation memory becomes the limiter.
- 128K → 1M: CP dominates, must overlap aggressively, consider mixed precision (FP8) for KV.

Hold model size fixed and scale **experts**:

- 8 → 32 experts: EP intra-node, all-to-all stays cheap.
- 32 → 256 experts: EP crosses nodes, all-to-all dominates. Look at expert-sharing tricks (DeepSeek-V3 shared expert) or alternative routing.
- 256+ experts: routing diversity is your bigger problem than systems; you usually need expert-parallel + topology-aware routing.

Scale **both** context and experts: each pushes the other into a tighter regime. A realistic frontier MoE long-context model balances them — `~64–128 experts` with `~32–128K` context, not `1024 experts + 1M context`.

### Overlap is the lever

The communication-cost numbers above are upper bounds — they assume the collective is on the critical path. In practice:

- **TP all-gather** overlaps with the next matmul (Megatron does this automatically with `--tp-comm-overlap`).
- **CP ring** overlaps with attention compute (the ring step is hidden behind the next chunk's FlashAttention).
- **EP all-to-all** can partially overlap with router compute (DeepSpeed's `MoE_dispatch_async`).
- **DP all-reduce** overlaps with backward of next layer (standard ZeRO + gradient bucketing).

When overlap is healthy, the **visible** communication cost is far less than the raw bytes. When overlap breaks (graph-capture issue, NCCL stream sync), it shows up as a sudden TFLOPs drop.

### Fault tolerance under combined parallelism

The more dimensions you slice, the more ways things can break. A practical floor:

- Checkpoint every 30–60 minutes (Module 09).
- Validate the mesh on a small step before launching a long run.
- Monitor per-collective time as a first-class metric — most failures appear as a single collective hanging or slowing.

---

## Build it

### 1. Mesh-layout decision exercise

For a target model + sequence + cluster, fill out:

```
Cluster: 8 nodes × 8 H200 (64 GPUs total), NVSwitch intra-node, IB inter-node
Model: 64-expert (top-2) 7B base, hidden=4096, layers=32
Sequence: 32K

Choices:
- TP =      (must be ≤ GPUs/node)
- SP =      (always on if TP > 1)
- EP =      (must divide num_experts; usually ≤ GPUs/node)
- CP =      (must divide GPUs/node, leave room for TP and EP)
- PP =      (usually = num_nodes if model fits per stage)
- DP =      (= total_GPUs / (TP × EP × CP × PP))

Constraints:
- TP × EP × CP ≤ GPUs per node = 8
- Total = TP × EP × CP × PP × DP = 64
- Per-rank HBM usage ≤ 141 GB (with recomputation/offload headroom)
```

Iterate to find a layout that satisfies the constraints and minimizes EP cross-node hops.

### 2. Communication-cost calculator

```python
# comm_cost.py
def per_layer_comm_bytes(T, H, H_kv, D, k, TP, CP, EP):
    tp_ag = T * H * (TP - 1) // TP * 4         # 4 matmuls per layer
    cp_ring = (T // CP) * 2 * H_kv * D * (CP - 1) if CP > 1 else 0
    ep_a2a = T * k * H * 2 * 2 if EP > 1 else 0  # dispatch + combine
    return dict(tp=tp_ag * 2, cp=cp_ring * 2, ep=ep_a2a)  # bytes (bf16)

def bw_for_collective(coll, intra_node_bw_GBs, inter_node_bw_GBs, crosses_node):
    return inter_node_bw_GBs if crosses_node else intra_node_bw_GBs

# Plug in your layout choices and see expected per-layer comm time
```

Print the breakdown for two or three candidate layouts; pick the one with the smallest **maximum** collective time per layer.

### 3. Sanity-run

If you have the cluster, run a 20-step training iteration with each candidate layout. Compare:

- Per-iter wall-clock.
- Per-collective time (Megatron-LM `--log-throughput --log-communication-volume`).
- Per-GPU TFLOPs achieved.

The layout with the smallest predicted comm should match the measured fastest. If it does not, your overlap assumptions are wrong somewhere — fix that before scaling up.

---

## Use it in the real stack

The Megatron-LM flags that pin a mesh:

```
--tensor-model-parallel-size <TP>
--pipeline-model-parallel-size <PP>
--context-parallel-size <CP>
--expert-model-parallel-size <EP>
--sequence-parallel
--tp-comm-overlap
--num-experts <E>
--moe-token-dispatcher-type alltoall
```

NeMo Megatron Bridge wraps these in a higher-level config; for long-context MoE the "MoE Long-Context Training" skill page gives reference configs by model size + sequence + cluster.

DeepSpeed has a different API but the same dimensions; for an EP-centric setup, DeepSpeed-MoE is sometimes cleaner.

---

## Measure it

For each candidate layout:

- **Predicted vs measured per-collective time**.
- **Achieved TFLOPs per GPU** as fraction of theoretical peak.
- **Step time breakdown** by phase (fwd compute, bwd compute, tp comm, ep comm, cp comm, dp comm).
- **Per-rank peak HBM** — must fit with headroom.

A healthy combined-parallelism run on 64×H200 holds per-GPU TFLOPs at ~45–55% of peak. If you are at 30%, you are spending too much time in collectives; if at 65%+, you are probably not using all available parallelism.

---

## Ship it

Drop into `lcm-course/`:

1. `mesh_decision_table.md` — candidate layouts, predicted comm costs, chosen layout with rationale.
2. `comm_cost.py` and its CSV output.
3. (If cluster access) `mesh_sanity_run.log` with per-layout measurements and a one-paragraph conclusion identifying the dominant collective.

---

## Related pages

- [Module 02 — Long-context attention mechanics](02-Long-Context-Attention.md)
- [Module 05 — MoE systems and infrastructure](05-MoE-Systems-Infrastructure.md)
- [Module 09 — Distributed training infrastructure](09-Distributed-Training-Infrastructure.md)
- NeMo Megatron Bridge MoE long-context skill: <https://docs.nvidia.com/nemo/megatron-bridge/nightly/skills/perf-techniques/moe-long-context/SKILL.html>
- Megatron-LM mesh README: <https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/README.md>
