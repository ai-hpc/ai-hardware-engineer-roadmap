# Lecture 02 — Workload Atlas and Trace Contract

[Previous: product contract](Lecture-01.md) · [Course overview](README.md) · [Next: tensor and vector tiles](Lecture-03.md)

**Outcome:** turn a robot policy into a versioned workload package that a compiler engineer, RTL designer, and robotics evaluator can all interpret. Your deliverable is a trace contract and a small shape/lifetime inventory, before choosing tensor-array dimensions.

**Prerequisites:** the mission and quality budget from Lecture 01; matrix multiplication, attention, and action chunking from [VLA Foundations](../../Track%20D%20-%20Robotics/VLA%20Foundations%20to%20Frontier%202026/README.md). Use the CPU exercise first; a GPU and robot are optional extensions.

## 1. A workload is more than its parameter count

Two policies with the same number of parameters can stress hardware differently. One may generate action tokens sequentially; another repeatedly update a continuous action chunk. Camera count changes visual-token work. Chunk horizon changes action-head shapes. Frame history changes both live state and dependencies. None of these differences is represented by “3B parameters.”

Use three complementary artifacts:

1. **Semantic contract:** what observations and actions mean, including units, frames, timestamps, normalization, and embodiment.
2. **Computation contract:** operators, shapes, dtypes, dependencies, repetitions, and tensor lifetimes.
3. **Measurement contract:** what was timed or counted, on which device and software revision, with which synchronization and quality checks.

The hardware must satisfy all three. A fast GEMM result with the wrong action normalization is an incorrect robot policy.

## 2. Choose architectural diversity deliberately

| Family | Distinct workload to investigate | Experiment status |
|---|---|---|
| SmolVLA | Flow-based continuous chunks; asynchronous policy inference and action playback | Public model/code; adapt the base policy to the evaluation task |
| openpi π₀ / π₀.₅ | Vision-language conditioning followed by a repeated flow head | Public implementations/checkpoints; pin the head, framework, and adapter |
| openpi π₀-FAST | Autoregressive action-token generation and decoding | Public implementation/checkpoints; do not assume the flow-head trace applies |
| GR00T N1.7 | Action Cascade combining a vision-language backbone and diffusion action head | Public code and checkpoint pages; gated backbone access is a dependency |
| π₀.₇, MEM, Gemini Robotics 2 | Steering, temporal memory, and future deployment requirements | Research/watchlist cases where reproducible access is not established or is restricted |

These entries are supported and qualified in the [dated source ledger](Sources-Models.md). Select two contrasting executable policies when equipment permits. Three checkpoints of the same graph provide less architectural coverage than two different action-generation mechanisms.

The [cross-accelerator study](https://arxiv.org/html/2604.24447v1) motivates profiling the backbone and action expert separately. Treat its stage bottlenecks as hypotheses. A different token count, data type, cache policy, or hardware balance can change which resource limits a stage.

## 3. Draw the graph before multiplying the work

A useful **illustrative** flow-head graph is:

```text
images + instruction ──► encoder/backbone ──► conditioning C
robot state + noise ───► expert step 0 ──► step 1 ──► ... ──► step S−1
                                              ▲                 │
                                      C is reused at each step   ▼
                                                    decode/normalize → chunk
```

This drawing is a hypothesis to verify in source code or a trace. Some implementations precompute conditioning projections; others recompute part of them. Some attention layouts permit prefix-cache reuse; others require additional dependencies. Record actual calls rather than assigning every VLA the same graph.

For this graph, operation counts follow:

\[
F_{\text{query}}=F_{\text{encoder/backbone}}+
\sum_{s=0}^{S-1}F_{\text{expert},s}+F_{\text{decode}}.
\]

The backbone is not multiplied by \(S\) unless it really executes \(S\) times. Weight storage is counted once per distinct allocation. Weight **traffic** depends on the memory boundary and reload schedule. A graph invocation does not prove a DRAM read: the operand may already reside in a cache or scratchpad.

Autoregressive decoding needs a different record: generated-token count, stop condition, attention/cache growth, and the mapping from tokens to actions. Do not rename token iterations “denoising steps.”

## 4. The manifest binds semantics to bytes

Record these fields before collecting performance:

| Group | Required fields | Why it changes the conclusion |
|---|---|---|
| Identity | checkpoint ID + immutable revision, file hashes, code commit, configuration hash | An updated checkpoint can change shapes or task quality |
| Inputs | camera order/count, resolution, crop, channel order, value range, instruction tokenizer, state fields | Identical array sizes can encode different observations |
| Time | per-sensor timestamp, clock domain, alignment rule, query ID, observation age | A fast answer can still use stale observations |
| Outputs | action dimension, units, coordinate frame, absolute/delta convention, normalization version | Actions cannot be interpreted from tensor shape alone |
| Schedule | predicted horizon, executed prefix, control period, inference trigger, denoising/token settings | Playback rate is distinct from fresh policy-query rate |
| State | cache dependencies, history length, reset/invalidation rule, random seed/noise inputs | Reuse can change behavior or violate causality |
| Runtime | framework/compiler/runtime versions, device, power/clock policy, warmup, memory allocator | A “model speedup” may come from a changed execution environment |
| Quality | episode IDs, held-out split, task success definition, allowed degradation | Calibration observations must not silently become the test set |

For `actions.shape = [1, 16, 8]`, state what each of the eight channels represents. Six pose coordinates plus gripper and auxiliary command is one possibility; eight joints is another. Shape alone does not identify degrees of freedom or physical units.

## 5. Build an operator and allocation inventory

One row per operator invocation should include `query_id`, `stage`, `iteration`, `op_id`, dependencies, input/output allocation IDs, logical shapes, physical strides/layout, storage and compute dtypes, accumulation dtype, scale layout, and device. Add start/end timestamps only when measured; never use zero as a placeholder for an unknown duration.

Track allocations separately: `allocation_id`, allocated bytes, producer, last consumer, memory tier, alias/view relationships, and persistence across queries. A view can have a logical size without owning new storage. An allocator's reserved pool is not identical to live tensor bytes.

Label byte counts with their origin:

- **Logical bytes:** derived from shape and representation.
- **Allocated bytes:** includes alignment, padding, packing, and workspace.
- **Transferred bytes:** measured counters or a named memory model, at a specified boundary.

An estimated read of `weights.nbytes` is a traffic assumption, not a hardware-counter measurement. Mark a CPU-only analytical trace `synthetic`; mark a source-derived shape inventory `static`; reserve `measured` for captured observations.

## 6. Worked example: repeated work and lifetime

Consider a **toy projection graph**, not a complete real VLA. A backbone projection uses \(M=512,K=N=1024\). An expert projection uses \(M=16,K=N=1024\) for six iterations. They have separate BF16 weight matrices. Count a multiply-add as two operations.

| Item | Calculation | Result |
|---|---|---:|
| Backbone projection | \(2\times512\times1024^2\) | 1,073,741,824 operations |
| One expert projection | \(2\times16\times1024^2\) | 33,554,432 operations |
| All expert projections | previous row × 6 | 201,326,592 operations |
| Each weight matrix | \(1024^2\times2\) bytes | 2 MiB |
| Fixed conditioning tensor | \(512\times1024\times2\) bytes | 1 MiB |
| One expert input/output tensor | \(16\times1024\times2\) bytes | 32 KiB |

If both matrices stream from external memory on every invocation, weight traffic is \(2+6\times2=14\) MiB per query. Distinct weight storage is only 4 MiB. If expert weights remain resident throughout the six steps, weight traffic for these projections becomes 4 MiB per query, assuming a cold start and one load of each matrix. Neither count includes activation traffic, scaling metadata, or other operators.

Keeping conditioning, the expert matrix, and separate input/output buffers simultaneously requires at least \(1+2+2/32=3.0625\) MiB. This excludes double buffering, workspace, allocator padding, and all other state. A 2.5 MiB scratchpad cannot satisfy that particular residency plan. A tiled plan may fit, but must expose its changed transfer schedule.

Now connect the query to time. At a 50 Hz controller period, a predicted horizon of 16 commands spans 320 ms; executing four commands before replanning spans 80 ms. Neither implies the model runs at 50 Hz. If a query takes 100 ms and no newer chunk arrives, the execution schedule must define queue depletion and stale-action handling. Expanding the chunk does not make its observations fresher.

## 7. CPU lab: an auditable trace fixture

Save this as `trace_fixture.py` in your experiment directory and run `python3 trace_fixture.py`. It performs accounting only, with no model downloads or accelerator dependencies.

```python
import json
from math import prod

MiB = 1024 ** 2
steps = 6
weight_shape = (1024, 1024)
weight_bytes = prod(weight_shape) * 2
rows = []
for stage, m, repeats in (("backbone", 512, 1), ("expert", 16, steps)):
    for iteration in range(repeats):
        rows.append({
            "evidence": "synthetic",
            "stage": stage,
            "iteration": iteration,
            "mnk": [m, 1024, 1024],
            "storage_dtype": "bf16",
            "ops": 2 * m * 1024 * 1024,
            "weight_allocation": stage + ".weight",
            "weight_bytes": weight_bytes,
            "duration_ns": None,
        })

traffic_streamed = sum(row["weight_bytes"] for row in rows)
traffic_resident = len({r["weight_allocation"] for r in rows}) * weight_bytes
assert len(rows) == 7
assert sum(r["ops"] for r in rows) == 1_275_068_416
assert traffic_streamed == 14 * MiB
assert traffic_resident == 4 * MiB
print(json.dumps({"events": rows, "assumed_streamed_bytes": traffic_streamed,
                  "assumed_resident_bytes": traffic_resident}, indent=2))
```

Extend the fixture with conditioning and input/output allocations, producer/last-consumer IDs, and a dependency edge between expert steps. Add a manifest containing a **hypothetical** embodiment and action schema. Keep missing timestamps null. Then change the action horizon to 32: identify which shapes and operation counts change and which weight sizes stay fixed.

**Optional measured route:** use one accessible checkpoint and a recorded observation, without connecting actuators. Capture profiler events plus model configuration; annotate which runtime operators map to each stage. Separate profiler-instrumented timing from the final latency run. Include warmup, asynchronous-device synchronization, per-query wall time, and the action-output validation result. Module hooks alone can miss functional or fused operations.

Use [the action-parity course](../../Track%20D%20-%20Robotics/VLA%20Optimization%20and%20Action-Parity%20Harness/README.md) to distinguish replayed-observation numerical checks from closed-loop task evaluation. A synthetic shape inventory passes this lecture's analytical route; it cannot establish policy accuracy or chip performance.

The [model-to-hardware lab](Model-to-Hardware-Lab.md) supplies the measured CPU route using the small scalar flow model trained in the VLA course. Export its actual graph, capture linear calls and profiler events, and compare them with parameter, operation, and tile-padding calculations. This is a trained synthetic model, not a pretrained VLA benchmark.

## 8. Self-check and handoff

1. **Why is storage 4 MiB while streamed weight traffic is 14 MiB?** There are two distinct matrices, but one is loaded six times under the stated schedule.
2. **Does the 3.0625 MiB figure prove a design fits in 4 MiB SRAM?** No. It is a lower bound for selected live tensors, excluding workspace, banking, padding, and concurrent work.
3. **Can conditioning be reused for the next camera observation?** Only if all dependencies are unchanged, or as an explicitly approximate method with invalidation and behavioral tests.
4. **Does a 50 Hz playback loop prove 20 ms response to a new event?** No. Reaction includes sensing, query triggering/queueing, inference, and when a fresh action becomes eligible for execution.
5. **Which graph should π₀-FAST use?** Its actual autoregressive graph, with token/cache accounting; the six-step toy flow graph is unsuitable.

**Pass gate:** another learner can reconstruct the seven-event calculation, identify every assumption, and explain the action/time schema without asking you what the columns mean. Deliver the manifest, event inventory, allocation/lifetime table, and an evidence-status label to [Lecture 03](Lecture-03.md) for compute-tile selection, then [Lecture 04](Lecture-04.md) for the capacity and traffic plan.
