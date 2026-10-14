# Lecture 01 — Design brief, competitive baselines, and evidence gates

[Course guide](README.md) · [Next: workload characterization](Lecture-02.md) · [Hardware source ledger](Sources-Hardware.md)

**Outcome:** write a falsifiable specification for a programmable VLA inference accelerator and explain what evidence would justify building it.

**Prerequisites:** inference profiling, bytes versus operations, basic robot action representations, and the [Physical AI Design Studio](../../../Physical-AI-Design-Studio.md). You do not need a fabrication process or RTL implementation to complete this lecture.

**Research cutoff:** October 14, 2026. Specifications and external benchmark results below are attributed reports; they are not measurements made by this course.

## 1. Define the product by what the robot accomplishes

The proposed product is an accelerator for **local policy inference**. Training may remain off-device. Its purpose is to reduce compute-system energy while satisfying task-quality, observation-age, and response-time requirements.

Write the central claim before selecting a tensor array:

> For a specified family of tasks and policies, our complete compute system completes the required tasks within the response-time limit, with an acceptable task-success difference, using less energy per successful task than the strongest reproducible baseline we can evaluate.

Every phrase must become a field in the experiment manifest. “Acceptable” requires a numerical, justified quality margin; “less” requires an energy boundary and measurement method. A proposed twofold efficiency improvement is a research target until demonstrated.

The design should accommodate several policy graphs through its compiler and runtime. Supporting an additional matrix shape, normalization, or action-head schedule should usually require software work. Supporting a different arithmetic format may require hardware changes. **Programmability has boundaries: specify them.**

## 2. Establish realistic external baselines

### Module specifications are capacity constraints

NVIDIA lists the following Jetson modules. FP4 throughput is **sparse**; memory bandwidth is an advertised interface figure. Module power ranges do not include every carrier, sensor, or cooling load. [NVIDIA specifications](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/)

| Module | Advertised sparse FP4 TFLOPS | LPDDR5X | Advertised bandwidth | Module power range |
|---|---:|---:|---:|---:|
| Jetson T5000 | 2,070 | 128 GB | 273 GB/s | 40–130 W |
| Jetson T4000 | 1,200 | 64 GB | 273 GB/s | 40–70 W |

Qualcomm advertises up to **700 TOPS** for its Dragonwing IQ10 Robotics Reference Design. The cited overview does not establish the precision and sparsity conventions needed to compare this figure with sparse FP4 throughput. Treat IQ10 as a candidate platform whose reproducible policy results must be obtained separately. [Qualcomm announcement](https://www.qualcomm.com/news/onq/2026/06/dragonwing-iq10-robotics-reference-design)

### A policy benchmark is closer to the relevant question

NVIDIA's GR00T hardware guide reports these **N1.7 replanning rates, with one camera and four denoising steps**. They describe fresh action chunks, not motor-control frequency. [Reported hardware results](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/hardware_recommendation.md)

| Platform | PyTorch eager | TensorRT |
|---|---:|---:|
| Jetson AGX Thor | 8.9 Hz | 12.4 Hz |
| Jetson Orin, 64 GB | 2.9 Hz | 6.6 Hz |
| RTX PRO 6000 Blackwell | 12.8 Hz | 35.9 Hz |

The companion deployment table reports Orin eager as **354.0 ms / 2.8 Hz**, differing from the summary above. Keep that discrepancy in your evidence record. Its command defaults specify BF16 and batch size one, but defaults do not prove every parameter used for the published runs. Reproduce with a pinned checkpoint, camera resolution, action settings, power mode, clocks, software versions, and raw timing samples. [Deployment methods and results](https://github.com/NVIDIA/Isaac-GR00T/blob/main/scripts/deployment/README.md)

An action chunk may supply commands between replans. That improves scheduling flexibility, but cannot make a stale observation fresh. Measure the age of the observation behind the command actually executed.

## 3. Measure useful work with an explicit boundary

For all attempted episodes in the declared evaluation set, define:

\[
E_{\text{per success}} =
\frac{\sum_i\int_{t_{i,\mathrm{start}}}^{t_{i,\mathrm{end}}}P_{\mathrm{compute\ system}}(t)\,dt}
{N_{\mathrm{successful\ episodes}}}.
\]

Include failures and the retries specified by the protocol. Report “undefined: no successes” when the denominator is zero. Also report attempts, successes, duration, energy per policy evaluation, and the distribution of episode energy. Task selection and termination rules must be shared across systems.

For the primary compute-system metric, include the host, accelerator, DRAM, and required interconnect; declare whether carrier conversion losses, networking, and cooling are included. Report whole-robot energy separately when relevant. An accelerator-only power rail cannot establish a compute-system advantage by itself.

**Worked example — invented measurements for reasoning practice.** Two systems each attempt the same 100 simulated episodes. Their time-weighted average compute-system power across those attempts and mean episode durations are hypothetical summaries, not device specifications.

| System | Average power | Mean attempt duration | Successes | Total energy | Energy per success |
|---|---:|---:|---:|---:|---:|
| A | 30 W | 20 s | 90 | 60,000 J | 666.7 J |
| B | 20 W | 25 s | 60 | 50,000 J | 833.3 J |

System B consumes fewer joules per attempt yet more per successful task. More importantly, it fails a matched-quality comparison. A third system using 24 W for 20 s with 90 successes would yield 533.3 J per success, a **1.25×** efficiency ratio relative to A, subject to the full latency and uncertainty checks.

Do not infer matched quality from equal point estimates alone. Preserve paired task conditions, seeds where meaningful, independent repetitions, sample counts, and confidence intervals. Choose a non-inferiority margin before inspecting the results; a statistically inconclusive result is not proof of equivalence.

## 4. Distinguish response time, throughput, and control

Measure the following timestamps on a reconciled clock:

| Event | Why it matters |
|---|---|
| Sensor exposure / observation timestamp | Defines which world state the policy observed |
| Input ready | Exposes capture, transport, and preprocessing delay |
| Policy start and completion | Exposes queues and inference time |
| Validated action ready | Includes decoding, unit conversion, and action checks |
| Command execution | Exposes buffering and controller delay |

Report median and tail observation-to-action latency, deadline-miss count and denominator, maximum observed age, and fresh-chunk rate. State sensor timestamp semantics and clock uncertainty. Accelerator event timing alone cannot cover the whole path.

A measured p99 does not establish a worst-case execution bound. A queue that hides compute jitter can increase action age. Independent supervision must reject expired or invalid commands and select a validated fallback appropriate to the robot.

NVIDIA's IGX safety brief distinguishes an **FSI inside the SoC**, with its own clocks and power rail, from a **safety MCU on the carrier board**. Its Safety Extension Package connects hardware features to supervision software. This is an architecture reference; copying a watchdog or using a particular module does not certify a robot. [IGX safety product brief](https://developer.download.nvidia.com/assets/igx/robotics-product-brief-igx-thor-safety-4473375.pdf)

## 5. Run two comparisons and label them clearly

| Comparison | Hold fixed | Permitted change | Claim supported |
|---|---|---|---|
| Hardware / implementation | Checkpoint, input trace, arithmetic specification, camera setup, action horizon, iteration count, acceptance criteria | Compiler mapping, kernels, hardware, memory schedule | Benefit of this implementation for that workload |
| Model–hardware co-design | Task distribution, success definition, quality margin, latency limit, energy boundary, allowed information | Model, precision, iteration count, hardware, schedule | Benefit of the complete deployed system |

Changing ten denoising steps to four may be valuable, but the improvement includes an algorithm change. Quantizing only the proposed system has the same attribution issue. Compare an optimized baseline, and account for any unsupported-operation CPU fallback.

At minimum, keep one compact policy, one iterative action policy, and one structurally different policy in the evaluation plan. Include difficult observations and concurrent system activity. The next lecture turns those choices into traces.

## 6. Treat the starting envelope as a search space

These ranges come from the project proposal. **They are exploration bounds, not a feasible product specification, and their upper endpoints need not be achievable together.**

| Dimension | Initial sweep | Evidence needed before commitment |
|---|---|---|
| Policy size | Approximately 0.5–4B parameters | Actual stage shapes, bytes, iteration counts, quality |
| Complete compute-module power | Approximately 20–40 W | Sustained measured boundary, DRAM and host allocation, thermal design |
| External memory | 16–32 GB | Peak live working set, runtime workspace, Linux and concurrent tasks |
| Local SRAM | 16–64 MiB | Bank/port design, macro area, leakage, routing and timing |
| Fresh policy evaluations | Investigate 10–30 Hz | Specified sensors and model, action age, tail delay, task quality |
| Arithmetic | Reference path plus selected 8-bit and low-bit paths | Exact format, scales, accumulation, kernel coverage and behavioral evaluation |

Keep fabrication node and peak TOPS open until traces and physical estimates narrow the design. For example, **64 MiB is 536,870,912 storage bits**, before ECC or spare rows. A bit count is not an SRAM macro area estimate: banking, peripheral circuitry, ports, process, and floorplan matter.

Start with an accelerator subsystem attached to a known host. A complete SoC adds boot, MMU/IOMMU, interrupts, coherency, memory-controller integration, Linux drivers, connectivity, and recovery responsibilities. Include those responsibilities in the eventual product plan without making them prerequisites for the first architectural experiment.

## 7. Advance through evidence gates

| Gate | Required evidence | A reason to stop or revise |
|---|---|---|
| Workload | Reproducible reference policy and trace manifest | Task quality or graph cannot be reproduced |
| Architecture | Executable model, traffic accounting, sensitivity sweeps | Advantage disappears with credible bandwidth or overhead |
| Compiler/runtime | Correctness tests, fallback report, dependency and residency schedule | Unsupported operators dominate the critical path |
| Prototype | Verified selected datapaths and interfaces; FPGA or RTL measurements | Data hazards, queue stalls, or host cost invalidate the model |
| Physical feasibility | Technology-specific SRAM/IP assumptions, synthesis and physical estimates | Power, timing, memory interface, package or cost fails its budget |
| Product readiness | Integration verification, bring-up/test plan, fault handling, task evaluation | Claimed gain fails under thermal load or concurrent work |

Gemmini provides a concrete study of configurable matrix arrays, scratchpads, DMA, and software-visible scheduling. Ara provides a complementary programmable vector architecture. Neither is a drop-in VLA chip: investigate the uncovered operations and interfaces. [Gemmini](https://github.com/ucb-bar/gemmini), [Ara](https://github.com/pulp-platform/ara)

Timeloop/Accelergy estimates depend on mappings and technology cost models. Validate selected design points against RTL and implementation evidence. An FPGA timing or power result is not a direct ASIC prediction. [Modeling methodology](https://timeloop.csail.mit.edu/)

## 8. Lab — defend a one-page design brief

Produce `design-brief.md` and a baseline manifest. This lab is complete on a CPU-only machine; label imported measurements as external reports and unmeasured fields as unknown.

1. Choose one simulated manipulation task, a reference policy, camera configuration, and observation/action contract. Link exact checkpoint and code revisions when available.
2. Set quality margin, maximum permitted action age, energy boundary, and target response time. Explain each choice using the task, not a vendor headline.
3. Create the two comparison rows from Section 5. Identify an optimization that belongs only in the co-design row.
4. Recompute the energy example and storage-bit count. Change success count, duration, and power separately; explain which change reverses the ranking.
5. List the three largest uncertainties: for example effective bandwidth, low-bit action quality, and SRAM implementation cost. Propose the cheapest experiment that reduces each uncertainty.
6. Write a rejection criterion. A valid result may be that an existing platform already meets the requirements and the proposed accelerator has no defensible benefit.

**Pass condition:** a peer can identify the workload, distinguish reported evidence from your assumptions, recompute the arithmetic, and state what experiment would falsify the claim.

## Self-check

1. Why cannot 700 TOPS and 2,070 sparse FP4 TFLOPS rank VLA task efficiency? **Their arithmetic conventions and task measurements are not matched.**
2. Does 30 Hz action execution demonstrate 30 Hz reaction to new observations? **No; chunk playback can reuse decisions from older observations.**
3. Can a lower-power design be worse per successful task? **Yes; duration, failures, and retries affect the denominator and total energy.**
4. What is missing from “64 MiB SRAM fits the chip”? **A technology-specific implementation and its area, power, ports, routing, and timing costs.**
5. Does a watchdog establish functional safety? **No; supervision is one component of an application-specific, validated architecture.**

Continue to [Lecture 02](Lecture-02.md) with the design brief: its unknowns determine which tensor shapes, lifetimes, traffic, and timing observations you must collect.
