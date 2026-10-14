# Lecture 09 — Prototype the subsystem, then assess silicon feasibility

**Previous:** [Evaluation](Lecture-08.md) · **Next:** [Capstone](Lecture-10.md)

**Prerequisites:** a workload trace, compiler contract, and reference vectors. **Deliverable:** a bounded prototype and an implementation evidence matrix.

## 1. Choose the next uncertainty to remove

The long-term goal is a Linux-capable physical-AI SoC with an accelerator and wireless connectivity. The first implementation can be a custom accelerator beside an existing host. That isolates arithmetic, data movement, scheduling, and software integration before adding CPU, memory-controller, boot, RF, and packaging work.

| Prototype | Answers | Does not establish |
|---|---|---|
| Numerical reference | Operator and format semantics | Timing, power, physical feasibility |
| Analytical/trace model | Sensitivity and resource requirements under assumptions | Accurate cycle timing or complete contention |
| Cycle/transaction simulator | Modeled queues, stalls, commands, memory service | Fabricated area or measured power |
| RTL simulation/formal checks | Implemented logic and specified properties | Closed timing at a selected process |
| FPGA integration | Implemented datapath, interface, driver, measured prototype timing | ASIC power/performance through a scaling factor |
| Physical implementation study | Results at a named PDK/library/corner and constraints | Manufacturing readiness without full sign-off |
| Measured silicon | Actual behavior in tested conditions | Every future workload or operating condition |

Retain a model-to-measurement comparison at each step. A discrepancy should refine the model rather than disappear from the report.

## 2. A minimal useful vertical slice

Select one bottleneck subgraph: for example, quantized projection, scale reconstruction, accumulation, activation, and output layout. Implement the entire path from host input to returned output:

```text
reference vectors → compiler lowering → command descriptor → driver submit
  → DMA input → local buffers → tensor/vector execution → DMA result
  → completion/error → numerical comparison → full-path timing
```

Test odd dimensions and zero/outlier values before choosing large matrices. Include fallback execution in the timing when the selected graph requires it. A good prototype is small enough that you can explain its state machine and every memory transfer.

**Worked bound:** the selected block is 40% of a measured 50 ms path. A 4× block speedup with no new overhead reduces the path to `30 + 20/4 = 35 ms`, a 1.43× overall speedup. If the host boundary adds 6 ms, it becomes 41 ms, only 1.22×. State whether transfers already existed in the baseline before adding them.

## 3. Turn the software contract into hardware checks

Build a reference scoreboard and verify:

- Arithmetic: signedness, accumulation width, rounding, saturation, scale indexing, and tails.
- Commands: length/stride/address limits, alignment, versioning, and unsupported-operation errors.
- Flow control: no dropped or duplicated accepted work; correct completion tags; bounded queues.
- Ownership: a buffer cannot be reclaimed while a device still uses it.
- Recovery: reset, timeout, invalid descriptor, and cancelled client behavior.
- State: chunk/policy identifiers prevent an old completion from becoming a new action.

Use constrained/random tests and formal properties where they address an actual failure mode. Functional coverage should map to the contract, not merely count test cases. For Linux integration, follow the [DMA API](https://docs.kernel.org/core-api/dma-api-howto.html) and the selected bus/platform rules; coherency and ordering are separate responsibilities.

## 4. Area, power, and timing require a physical context

A proposed 64 MiB SRAM is `64 * 2^20 * 8 = 536,870,912` data bits before ECC, tags, queues, redundancy, and periphery. That is a capacity requirement, not an area estimate. Bank count, ports, word width, macro choices, and routing change feasibility.

For each implementation estimate save:

| Category | Required provenance |
|---|---|
| Process and cells | PDK/library versions, allowed use, voltage/temperature/process corners |
| Memories | Real versus placeholder macros, dimensions, ports, timing/power models |
| Constraints | Clocks, reset domains, I/O timing, false/multicycle path justification |
| Physical design | Placement/routing stage, utilization, congestion, clock tree, parasitics |
| Power | Activity source, workload coverage, leakage/dynamic split, voltage/frequency |
| Interfaces | PHY/controller IP, pads, package, signal/power integrity assumptions |

The [OpenROAD flow tutorial](https://openroad-flow-scripts.readthedocs.io/en/latest/tutorials/FlowTutorial.html) provides a reproducible path for an implementation exercise. Its results remain tied to the selected design and technology. The [generated-memory documentation](https://openroad-flow-scripts.readthedocs.io/en/latest/user/AutoMemories.html) explicitly distinguishes synthetic memory models; placeholder timing/power is not SRAM characterization. Never extrapolate an educational PDK result into a promised advanced-node product metric.

## 5. What changes when the host becomes part of the SoC?

Add a CPU with the facilities required by the selected Linux port, boot firmware, interrupt/timer architecture, debug, interconnect, and memory protection/translation where required. Define which agent can address each memory region and how DMA isolation works. A CPU ISA choice alone does not specify the accelerator ABI or a functioning Linux platform.

Integration adds clocks/resets, power domains, coherency choices, shared DRAM contention, boot order, error reporting, and firmware updates. Re-run the original numerical and scheduling tests after each boundary change.

For wireless, distinguish an external module, companion die, and monolithic RF integration. The host interface can be designed and fault-tested before RF integration. Include wireless DMA/interrupt traffic in the contention study. Time-stamped task requests, telemetry, and updates have different requirements from local actuator control.

Use the roadmap's [embedded Linux](../../../Phase%202%20-%20Embedded%20Systems/3.%20Embedded%20Linux/Guide.md), [ESP-Hosted integration](../../../Phase%202%20-%20Embedded%20Systems/3.%20Embedded%20Linux/Jetson%20ESP-Hosted%20Host%20Code/Guide.md), and [SoC capstone](../Guide.md#physical-ai-soc-capstone) for the next level of integration.

## 6. Silicon decision gate

Write a decision memo with requirements, achieved evidence, remaining risks, cost assumptions, and alternatives. Require a usable compiler/runtime and supported workloads alongside the implementation study. Include verification, test access/DFT, memory-interface feasibility, packaging, thermal design, bring-up observability, and recovery planning.

Functional safety is a separate assurance effort. A watchdog or supervision island is an architectural mechanism, not a certification. Follow the scope distinctions in [Sources-Hardware](Sources-Hardware.md) when discussing IGX as a reference architecture.

## Lab and acceptance

Choose one vertical slice and produce a requirements-to-check table. Run its reference and your available implementation level. Add an unsupported command and a reset/timeout scenario before reporting latency.

**CPU-only route:** implement a command-stream interpreter for the operator reference, validate descriptors, and log buffer lifetime transitions. Use the analytical labs to choose a tile and SRAM hypothesis. Mark FPGA, PPA, and physical power as not run.

**FPGA route:** report fabric utilization, achieved timing constraints, device/host transfer costs, and end-to-end latency. Save the bitstream/build recipe, host code, and numerical vectors. Keep ASIC estimates in a separate column.

**Acceptance:** another learner can identify exactly which claims the prototype establishes and which tool/hardware experiment would establish the next claim.

## Self-check with answers

1. **Why not multiply FPGA throughput by a clock ratio to predict ASIC throughput?** Memory ports, routing, arithmetic mapping, I/O, and workload bottlenecks can change; frequency alone is insufficient.
2. **What does an unconstrained passing timing report prove?** Very little about the intended interface. Missing clocks/I/O constraints can leave important paths outside the analysis.
3. **Why might more SRAM hurt?** It consumes area and power, changes banking/routing, and may not reduce the traffic on the actual critical path.
4. **When is a full SoC justified?** When integration benefits and product requirements support its additional engineering cost after the accelerator/software advantage is demonstrated.
