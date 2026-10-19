# Module 08 — Combine context and expert parallelism correctly

Long context distributes attention work; mixture of experts distributes selected feed-forward work. The same GPU ranks participate in different groups for those operations. You must understand those groups before interpreting memory, batch size, or communication.

**Before this lesson:** [attention](02-Long-Context-Attention.md), [MoE systems](05-MoE-Systems-Infrastructure.md), and [evaluation](07-Long-Context-Evaluation.md). **Outcome:** validate one concrete mesh, calculate logical payloads, and design a measured scaling comparison. The included CPU ledger verifies arithmetic. It does not initialize a distributed Megatron job or prove a topology is fast.

## 1. Give each parallelism axis a job

| Term | Work or state distributed across the group |
|---|---|
| Tensor parallelism (TP) | Pieces of a layer's matrix operations |
| Context parallelism (CP) | Tokens from the same sequence, with attention communication |
| Pipeline parallelism (PP) | Different layers/stages of the model |
| Dense data parallelism (DP) | Independent sequence batches for the dense computation |
| Expert parallelism (EP) | Different experts and the routed tokens sent to them |
| Expert tensor parallelism (ETP) | Matrix pieces within each expert |
| Sequence parallelism (SP) | Selected activation work/storage distributed over an existing TP group |

SP is not an extra multiplier in world size. Expert and dense group layouts can overlap; multiplying TP×CP×PP×EP×DP as if all were independent axes is wrong for the Megatron layout used here.

## 2. Work a 64-GPU example from the actual group definitions

This lesson follows Megatron-LM commit `ce852bfb93f364f64a556602fc71a5d9ed2c377e`, inspected on 2026-10-18. For the standard decoder layout with the additional GTP/EGTP rematerialization factors fixed at one, the cardinalities are:

`dense_DP = world_size / (TP × CP × PP)`

`expert_DP = world_size / (ETP × EP × PP)`

The expert rank generator sets its own CP axis to one. Consequently, `expert_DP = dense_DP / EP` is not generally valid when CP or ETP differs. Custom layouts and nonunit rematerialization factors require their own derivation. [Pinned group construction](https://github.com/NVIDIA/Megatron-LM/blob/ce852bfb93f364f64a556602fc71a5d9ed2c377e/megatron/core/parallel_state.py#L930).

For 64 GPUs, TP=2, CP=2, PP=2, EP=4, ETP=2, and eight experts:

| Calculation | Result |
|---|---:|
| Dense grid: 2×2×2×dense_DP=64 | dense_DP=8 |
| Expert grid: 2×4×2×expert_DP=64 | expert_DP=4 |
| Experts hosted by each EP rank, before ETP sharding | 8/4=2 |
| Standard dense weight-replica group including CP | dense_DP×CP=16 |

The last line is a parameter/gradient replication group, **not** 16 independent input batches. CP ranks hold parts of the same sequences. With one sequence per microbatch per dense DP replica and 16 accumulated microbatches, the global batch is `1×8×16=128` sequences. At 32,768 valid predicted tokens each, that would be 4,194,304 valid tokens per update; actual loss masking can reduce it.

From the course's repository root:

```bash
cd "Phase 5 - Advanced Topics and Specialization/Track A - GPU Infrastructure/Nvidia GPU/HPC Setup/Long-Context-MoE-Foundation-Training"
python3 labs/ledger.py
python3 -m unittest discover -s labs -p 'test_ledger.py' -v
```

Inspect the `mesh` result for dense DP8 and expert DP4. To test a different mesh from this course directory:

```bash
python3 - <<'PYCODE'
import sys
sys.path.insert(0, 'labs')
from ledger import dense_mesh
print(dense_mesh(64, tp=2, pp=2, cp=2, ep=4, expert_tp=2, experts=8))
try:
    dense_mesh(60, tp=2, pp=2, cp=2, ep=4, experts=8)
except ValueError as error:
    print('rejected:', error)
PYCODE
```

Sixty GPUs cannot be divided by the eight-rank dense model group. The ledger also rejects nonintegral expert divisions. Passing the ledger is only the first check: the real framework must construct rank groups with matching pipeline groups, valid ordering, compatible layers, and supported kernels. In this source, unequal dense/expert DP imposes a PP-last ordering condition when PP>1. Print actual rank membership in your real job before reasoning about which transfers stay inside a node.

## 3. Calculate payload before claiming network bandwidth

Use explicit units and state whether a count describes one rank, one layer, or the whole job.

**MoE dispatch example.** A source rank owns T=8,192 token rows, hidden width H=4,096, uses top-k=2, and sends two-byte activations. The unpadded logical expert-input payload is:

`T × k × H × 2 = 134,217,728 bytes = 128 MiB`.

If returned expert outputs have the same width/dtype, the return payload is another 128 MiB. This is an assignment-size calculation. Actual remote bytes depend on local expert assignments, capacity/drop policy, padding, tensor sharding, metadata, quantization, and the dispatcher algorithm. EP=1 can have that logical payload with zero inter-rank dispatch traffic.

**One CP ring illustration.** A rank holds 4,096 local tokens, eight KV heads of dimension 128, and two-byte K/V elements. Its K+V block is `2×4,096×8×128×2 = 16 MiB`. In an illustrative four-rank ring that circulates each block through three peers, a rank sends three such blocks, 48 MiB, in the forward circulation. This omits backward, queries, metadata, causal scheduling, and implementation-specific compression/overlap. Other CP algorithms have different transfers.

**Collective convention.** For an ideal eight-rank ring AllGather whose final tensor is 64 MiB per rank, each rank receives 56 MiB and sends 56 MiB. Do not label the eight-MiB local contribution as the full tensor size when comparing bandwidth formulas. Use the [collectives lab and accounting](../NCCL-Deep-Dive/CPU-Collectives-Lab/Guide.md) to keep algorithm bandwidth, normalized bus bandwidth, and measured link traffic distinct.

## 4. Map groups onto hardware, then measure

A group that contains more ranks than one node has GPUs necessarily spans nodes. TP=8 and CP=2 do not make their 16-rank combined dense model slice fit inside an eight-GPU node. Which individual TP, CP, and expert transfers cross nodes depends on rank ordering and the overlapping group layouts.

Compare a few valid layouts while holding the intended objective and valid-token batch constant. Record peak memory, useful tokens per second, step-time distribution, expert assignment counts, dropped assignments, tokens receiving no expert output, and time spent on the critical path. A shorter step caused by dropping more tokens is not equivalent training work.

Communication and compute overlap share memory bandwidth, buffers, streams, and network resources. Summing isolated kernel times does not predict overlapped step time. Start with traces that identify the stalled operation and its dependency; adjust one layout or scheduling choice, then repeat the same workload.

**Model FLOPs utilization (MFU)** compares a stated model-FLOP estimate per second with a stated hardware peak. For MoE, distinguish selected-expert work from all stored expert parameters; for long context, include the intended attention computation. Explain whether recomputation is excluded from the model numerator and which precision/peak is used. There is no universal healthy MFU percentage across these workloads. Useful-token throughput plus unchanged evaluation is often easier to compare.

## 5. Configuration combinations need source checks

A launch configuration must include the model, tokenizer, data, precision, optimizer, batch contract, attention backend, mesh, checkpoint path, and framework/container revisions. A fragment with only parallelism values is a design sketch, not a runnable training recipe.

For example, the checked Megatron argument validation rejects `recompute_granularity='selective'` together with a nonempty `recompute_method`; copying `selective` and `block` from separate examples creates an invalid combination. The same code disables SP when TP=1 and requires SP for its TP communication overlap option. Read the installed version's validation instead of assuming that individually valid options compose. [Pinned argument checks](https://github.com/NVIDIA/Megatron-LM/blob/ce852bfb93f364f64a556602fc71a5d9ed2c377e/megatron/training/arguments.py#L1323).

## Exercise with answers

Keep the example's world size, TP, PP, EP, ETP, and expert count, but increase CP from two to four. **Answer:** dense DP becomes four while expert DP stays four. At microbatch one and accumulation 16, global batch falls from 128 to 64 sequences. Raising accumulation to 32 restores that sequence count, but it does not prove equal throughput or identical floating-point updates.

Submit the two grid equations, actual or planned rank membership, the global-batch calculation, and one payload ledger. Mark the mesh as **arithmetic-checked** until the real framework has initialized it; mark performance as **unmeasured** until a specified workload runs on the intended hardware.

**Continue:** [Module 09 — Distributed training and recovery](09-Distributed-Training-Infrastructure.md).
