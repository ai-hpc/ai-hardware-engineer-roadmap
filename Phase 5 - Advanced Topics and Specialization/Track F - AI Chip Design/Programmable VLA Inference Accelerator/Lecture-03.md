# Lecture 03 — Programmable Tensor and Vector Tiles

**Question:** what compute organization serves both a large visual-token matrix and a short action sequence without depending on one model architecture?

**Prerequisites:** [Lecture 02](Lecture-02.md), matrix multiplication, tensor shapes, and the distinction between an operator graph and its execution schedule. Review the [accelerator architecture track](../Guide.md) if scratchpads and dataflows are new.

**By the end:** turn an operator trace into tile requirements, calculate padding waste, budget vector work, and explain why independently scheduled tiles still compete for memory.

**Deliverable:** a compute-fabric decision sheet with two candidate organizations, three workload shapes, a numerical model, and one experiment capable of rejecting your preferred design.

## 1. Begin with recurring operations

A VLA policy is a program with multiple operating regimes. Image patches, language tokens, robot state, and action samples produce different dimensions and dependencies. Model names alone do not specify these dimensions.

For each traced operation, record:

| Property | Why the architecture needs it |
|---|---|
| Operator and mathematical definition | A matrix multiply, reduction, gather, and nonlinear function need different machinery |
| Shape, stride, and layout | Logical tensor size does not describe physical access order |
| Storage, operand, accumulator, and output types | Packed weights and arithmetic operands can have different widths |
| Invocation count and dependencies | Repeated action steps can dominate even when each step is small |
| Bytes at each memory level | DRAM efficiency does not establish scratchpad or register efficiency |
| Critical-path membership | Accelerating a noncritical branch may leave action latency unchanged |

Use the traces from Lecture 02 to form shape families: image processing, multimodal prefill, single-token or short-sequence work, iterative action blocks, and vector operations. These are workload descriptions, not a rule that every VLA contains all five.

## 2. Specify a programmable tile

Our **proposed teaching design** contains a tensor engine, vector engine, banked local storage, a DMA interface, and a command queue. It is a design hypothesis, not a specification of an existing commercial chip.

```text
          compiler-generated command stream
                         |
                  tile command queue
                         |
       +-----------------+-----------------+
       |                                   |
 tensor engine <---- banked SRAM ----> vector engine
       |                |                  |
 accumulators           +---- DMA / interconnect ---- external memory
```

The tensor engine evaluates tiled contractions. The vector engine handles operations such as reductions, activation functions, scaling, and rearrangement. Both need explicit access to their inputs and outputs; drawing two boxes does not give them two independent SRAM ports.

Gemmini is a useful open reference for studying configurable arrays, banked scratchpads, accumulator storage, and DMA. Its implemented capabilities should be read from a pinned configuration rather than inferred from the word “programmable.” [Gemmini architecture](https://github.com/ucb-bar/gemmini)

Compare three organizations before choosing one:

| Candidate | Potential benefit | Cost to model |
|---|---|---|
| One large tensor array | Efficient mapping for suitably large matrices | Small-shape waste; limited independent scheduling |
| Several smaller tensor tiles | More shape choices and concurrent jobs | Replicated control, interconnect traffic, distributed reductions |
| Tensor tiles plus shared vector units | Less replicated vector hardware | Queues and interference at the shared vector stage |

Keep aggregate resources comparable. Four small arrays versus one large array is not a fair comparison if one proposal has twice the multipliers, memory ports, or clock rate.

## 3. Worked example: padding is visible before RTL

For a GEMM, let \(A\in\mathbb{R}^{M\times K}\), \(B\in\mathbb{R}^{K\times N}\), and \(C=AB\). Counting a multiply and an add as two operations gives approximately \(2MKN\) useful operations.

Assume a simplified implementation pads each dimension to a tile multiple and executes every padded lane. Its fraction of useful multiply-accumulates is:

\[
U_{\mathrm{padding}}=
\frac{MKN}
{\lceil M/T_M\rceil T_M\,
 \lceil N/T_N\rceil T_N\,
 \lceil K/T_K\rceil T_K}.
\]

This is **padding efficiency**, not measured utilization. It excludes fill/drain cycles, dispatch, stalls, data movement, and frequency changes. A machine that skips tails or repartitions lanes needs a different model.

Consider two invented shapes, with \(T_K=32\):

| GEMM | Useful operations | \(128\times128\) output tile | \(32\times32\) output tile |
|---|---:|---:|---:|
| Short action block: \(M=16,N=256,K=1024\) | 8,388,608 | 12.5% | 50% |
| Visual-token block: \(M=784,N=768,K=768\) | 924,844,032 | 87.5% | 98% |

For the action block, the large tile uses only 16 of 128 row positions. The smaller tile uses 16 of 32. Neither result proves the smaller design is faster: it may execute more tiles, repeat operand transfers, or provide fewer total multipliers.

**Design decision:** retain multiple shape mappings or independently schedulable resources only if the trace-weighted benefit survives those costs. Do not optimize solely for a square GEMM benchmark.

## 4. Choose dataflow using movement counts

An output-stationary mapping keeps partial sums close to the arithmetic while traversing the reduction dimension. A weight-stationary mapping retains weights while streaming inputs. These names describe reuse choices; they do not determine every buffer size or network transaction.

For each candidate mapping, count:

1. Weight reads from DRAM, shared SRAM, and local storage.
2. Input reads and any broadcast or replication.
3. Partial-sum reads and writes when a reduction is split.
4. Output stores, format conversions, and layout changes.
5. Metadata reads, including quantization scales.

An array can reduce DRAM traffic while increasing on-chip traffic enough to stall. Conversely, keeping weights stationary may consume storage needed for a useful activation tile. Evaluate the whole mapping.

Timeloop supports joint mapping and architecture exploration with modeled performance and energy. Treat its results as estimates to calibrate against a simulator or implemented block; they are not measured silicon results. [Timeloop overview](https://timeloop.csail.mit.edu/)

## 5. Vector operations are part of the policy

Consider a chain `GEMM → normalization → activation → GEMM`. A tensor engine cannot make the next GEMM ready until the intervening work finishes.

Define the required semantics for each vector kernel:

| Kernel family | Details that must survive lowering |
|---|---|
| Normalization | Reduction axes, epsilon, accumulation type, and affine parameters |
| Softmax | Mask, scaling, stable reduction, and treatment of fully masked rows |
| Positional operation | Position indices, layout, and model-specific convention |
| Quantization | Scale axes, clipping, rounding, and packed representation |
| Tensor rearrangement | Whether a view is sufficient or an actual copy is required |

A tiny vector workload can still be expensive if every invocation requires a CPU round trip. The compiler report should identify every fallback, its transferred bytes, and its synchronization cost. A first prototype can support a limited kernel set, provided unsupported operations remain visible in end-to-end measurements.

## 6. Independent tiles do not imply independent bandwidth

Suppose two independent jobs require 600 MB and 120 MB of DRAM traffic during the same measurement window. The shared interface sustains 120 GB/s under the assumed access pattern. Use decimal units consistently.

\[
T_{\mathrm{shared\ memory}}\ge
\frac{600+120\ \mathrm{MB}}{120\ \mathrm{GB/s}}=6\ \mathrm{ms}.
\]

The second job's isolated transfer time would be 1 ms. Allocating it a sustained 40 GB/s share instead makes its transfer component at least 3 ms. Allocating more tensor units to that job does not change these byte counts.

These are simplified bounds. A guaranteed service rate additionally requires bounded arbitration, request sizes, controller behavior, and competing traffic. Software priority alone cannot establish it.

If a newer observation can be encoded while an older action chunk is generated, record which observation each job consumes. Dependency-legal overlap can improve throughput while still producing actions based on old observations. [Lecture 07](Lecture-07.md) develops the scheduling and freshness contract.

## 7. Exercise: select a candidate that can lose

Use the two shapes above and one shape from a real pinned policy trace. For the CPU-only route, use a static shape inventory derived from a pinned model's source/configuration and label it `static`; if no model source is available, extend Lecture 02's fixture and label it `synthetic`. Neither establishes measured execution frequency or traffic. Compare one large array with multiple small tiles at the same aggregate multiplier count and stated clock assumption.

For each candidate:

1. Calculate useful work and padding efficiency; name unsupported assumptions.
2. Account for operand, accumulator, scale, and layout traffic at two memory levels.
3. Add one vector kernel with an explicit dependency.
4. Run or model two independent jobs with a shared bandwidth limit.
5. Sweep one parameter: tile dimensions, vector throughput, or SRAM ports.

**Negative case:** choose a shape with \(M=1\) or a deliberately awkward tail. Show where your preferred mapping wastes resources, spills, or falls back. Hiding this case invalidates the comparison.

**Acceptance:** submit a shape table, two architecture diagrams, assumptions, a reproducible calculation, and one sentence explaining the strongest reason to reject your chosen design. Report predicted and measured values in separate columns.

## 8. Answered self-checks

**1. Does 50% padding efficiency mean 50% of peak measured throughput?** No. It describes useful versus padded arithmetic in this model. Pipeline, memory, and scheduling losses can lower throughput further; tail-skipping hardware can invalidate the padding model itself.

**2. Why can a vector engine improve end-to-end latency even when it accounts for few operations?** Its operations may serialize the critical path, require expensive data conversion, or otherwise force CPU synchronization between tensor kernels.

**3. Can two tensor tiles each achieve their isolated DRAM bandwidth simultaneously?** Only if the memory system can supply the combined demand. Shared controllers, banks, and interconnect links impose aggregate limits and interference.

## Sources and next step

- [Gemmini implementation and configuration](https://github.com/ucb-bar/gemmini): inspect a concrete open architecture before borrowing its design ideas.
- [NVDLA hardware architecture](https://nvdla.org/hw/v1/hwarch.html): compare specialized engines and explicitly managed convolution storage with this lecture's proposed programmable organization.
- [Timeloop/Accelergy](https://timeloop.csail.mit.edu/): explore mappings and hardware parameters; record tool and technology-model revisions.

Continue to [Lecture 04 — Memory Residency and Traffic](Lecture-04.md). The compute decision becomes credible only after its working set fits a feasible memory plan.

## Execute the tile contract before scaling the array

The [tiling and dataflow lab](../Labs/Tiling-and-Dataflow.md) adds a functional check to the padding calculation: every tiled result is compared with a dot-product oracle, and operand tokens carry their K index through a small systolic grid. Compare a 1×1 output on a 2×2 array with a full 2×2 output at K=3. Both modeled runs issue 12 MACs over 5 cycles, but only 3 versus 12 are useful. This separates padding loss from pipeline fill/drain loss; neither is a measured chip utilization counter.
