# How to learn this roadmap

Choose one engineering outcome, diagnose the missing prerequisites, and build a small experiment before expanding it. The phase numbers describe dependencies; they are not a requirement to finish every course. Time estimates are planning aids, not mastery tests.

The shared destination is a [custom physical-AI SoC](README.md#the-goal) built around advanced VLA workloads: a dedicated accelerator and AI compiler, embedded Linux, wireless, and interfaces for robots and drones. Treat each course artifact as evidence for a subsystem of that platform. The [SoC capstone](Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20F%20-%20AI%20Chip%20Design/Guide.md#physical-ai-soc-capstone) joins the model, compiler, hardware, and application work.

## Choose a route and a first deliverable

| Route | Start | First deliverable | Then |
|---|---|---|---|
| Inference optimization | [Phase 4C](Phase%204%20-%20Track%20C%20-%20DL%20Inference%20Optimization/Guide.md) | Correctness-checked operator benchmark with a profiler trace | [Inference course](Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20G%20-%20ML%20Systems%20Engineering/AI%20Inference%20Engineer%202026/README.md), then its DeepSeek V4.1 Flash NVFP4 part |
| Robotics and robot learning | [Robotics](Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20D%20-%20Robotics/Guide.md) | A simulated robot with validated frames, units, reset behavior, and timing | [Isaac Sim / Isaac Lab](Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20D%20-%20Robotics/Isaac%20Sim%20and%20Isaac%20Lab/README.md), then held-out policy evaluation |
| Embedded and edge systems | [Phase 2](Phase%202%20-%20Embedded%20Systems/Guide.md) | A timestamped sensor-to-application pipeline with recovery from a fault | [Jetson](Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/1.%20Nvidia%20Jetson%20Platform/Guide.md), then a thermal and latency budget |
| FPGA and accelerator design | [Phase 1](Phase%201%20-%20Foundational%20Knowledge/Guide.md) | Tested RTL with a timing constraint and a waveform explaining a failure | [FPGA](Phase%204%20-%20Track%20A%20-%20Xilinx%20FPGA/1.%20Xilinx%20FPGA%20Development/Guide.md), then [chip design](Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20F%20-%20AI%20Chip%20Design/Guide.md) |
| AI compiler and runtime | [Phase 4C](Phase%204%20-%20Track%20C%20-%20DL%20Inference%20Optimization/Guide.md) | A reference-checked compiled subgraph with memory and execution traces | Custom accelerator backend, Linux driver, and SoC integration |
| ML and agent systems | [Phase 3](Phase%203%20-%20Artificial%20Intelligence/Guide.md) | A reproducible model or agent evaluation, including failed cases | ML engineering, runtime profiling, and one deployment target |

## Diagnose before reading

Try the entry task for your next course without following its solution. If you can produce and explain the artifact, skip the introductory reading and attempt the independent exercise. If you cannot, study the prerequisite that addresses the specific failure.

| Skill | Five-to-fifteen-minute diagnostic | Return here if stuck |
|---|---|---|
| Systems | Explain why asynchronous GPU work cannot be timed by timing only the launch | [Parallel computing](Phase%201%20-%20Foundational%20Knowledge/4.%20C%2B%2B%20and%20Parallel%20Computing/Guide.md) |
| Architecture | Estimate bytes moved and operations for a matrix-vector product; state your cache assumption | [Computer architecture](Phase%201%20-%20Foundational%20Knowledge/2.%20Computer%20Architecture%20and%20Hardware/Guide.md) |
| ML | Trace tensor shapes through a forward pass and explain how to hold evaluation data out | [Neural networks](Phase%203%20-%20Artificial%20Intelligence/1.%20Neural%20Networks/Guide.md) |
| Robotics | Draw world, base, and camera frames and distinguish physics dt from policy period | [Robotics foundation](Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20D%20-%20Robotics/Advanced%20Robot%20Operating%20System/Lecture-01.md) |
| Measurement | Name the baseline, one variable you will change, and an outcome that would disprove your prediction | The experiment record below |

## A repeatable lesson cycle

Use this cycle with each course's **Practice and assessment** section. For courses that already contain detailed labs, those labs remain the main assignments.

1. **Retrieve (5 minutes):** close the notes and draw the data path or explain yesterday's failure. Mark what you cannot yet explain.
2. **Predict (10 minutes):** estimate a result with units. Write the condition under which the estimate would fail.
3. **Study (20–30 minutes):** read the relevant concept and one worked example. Trace a small input by hand.
4. **Build (45–90 minutes):** reproduce the baseline, save the raw output, and change one variable.
5. **Debug (20 minutes):** introduce a bounded fault in a test or simulator. Show that your check detects it and that the fix restores behavior.
6. **Explain (10 minutes):** compare prediction with measurement and write the next experiment. An unchanged or slower result is useful evidence.

These are session sizes, not limits on difficult labs. Split installation, experimentation, and analysis into separate sessions when needed. After two days, answer the self-check without notes; after a week, repeat the exercise with an unfamiliar input; after a month, connect it to another course's artifact. Keep an error log of misconceptions and the evidence that corrected them.

## Three levels of practice

| Level | Assistance | What you demonstrate |
|---|---|---|
| Worked example | Follow provided commands and explain each intermediate result | You can reproduce a result |
| Guided variation | Change input size, layout, timing, fault, or workload; predict first | You understand what controls the result |
| Independent transfer | Solve a related problem without copying the example's implementation | You can choose and defend a method |

Do not make every lab larger. Moving a correct idea to a new condition is often a stronger assessment than using a bigger model or more GPUs.

## Hardware and access levels

| Access | Useful work | Claim you may make |
|---|---|---|
| CPU laptop | Tensor math, metadata inspection, RTL simulation, recorded-trace analysis, experiment design | Analysis or CPU result, with assumptions |
| One compatible GPU / development board | Operator profiles, small model serving, supported small simulation tasks, board bring-up | Measurements on the named device and software versions |
| RTX simulation workstation | Isaac scenes, sensors, parallel tasks, policy evaluation | Simulation results with engine, renderer, and task settings |
| Multi-GPU node | Full large checkpoints, expert parallelism, topology and load sweeps | Deployment results for the exact checkpoint/runtime/hardware tuple |

Each lab should offer a reduced exercise when possible. A CPU replay of an Isaac trace does not certify simulator performance. A metadata-only DeepSeek exercise does not certify inference. Record **not run** for experiments without the required hardware; complete the theory route without inventing results.

## Keep an experiment record

Use one directory per experiment. Save the following fields with the artifact; a Markdown table, JSON document, or notebook is fine.

```text
question and predicted outcome:
baseline and single variable changed:
source revision / checkpoint revision / asset revision:
OS, driver, runtime, compiler, device, memory, power mode:
input shape, dataset split or seed set, workload distribution:
commands and configuration:
correctness criterion fixed before optimization:
warmup, repetitions, timing boundary, sample count:
raw output and summary statistics:
failure cases and explanation:
limitations, next experiment, reproduction instructions:
execution status: measured | analytical | fixture | not run
```

Report units and distributions. GPU kernel time, server queue time, and end-to-end latency answer different questions. Separate cold starts from steady state. For robotics, separate transitions per second, physics steps per second, and simulated time divided by wall time. Never average percentiles from separate runs and call that the combined percentile.

## Completion rubric

Score each dimension 0 (missing), 1 (works with unexplained gaps), or 2 (reproducible and explained).

| Dimension | Evidence for 2 points |
|---|---|
| Correctness | Reference comparison, valid inputs, and a test that catches a deliberate fault |
| Reasoning | A prediction, assumptions, and an explanation of disagreement with data |
| Measurement | Raw data, units, timing boundary, repetitions, and a named hardware/software setup |
| Reproduction | Another learner can rerun the work from the recorded commands and revisions |
| Transfer | The method works on a new case, with limitations explained |

A course milestone passes at **8/10 or better, with 2 in correctness and reproduction**. This is a teaching rubric, not a production certification. Hardware availability does not lower the correctness standard; it changes which milestone you can claim. Retake only the dimensions that need work.

## Phase checkpoints

| Phase / specialization | Evidence to carry forward |
|---|---|
| 1 — Foundations | Testbench, cache/bandwidth prediction, concurrency failure and fix |
| 2 — Embedded | Interface map, boot/capture log, repeatable fault recovery |
| 3 — AI | Held-out evaluation, numerical reference, data and model versions |
| 4A — FPGA | Constraints, timing/utilization results, software-visible interface |
| 4B — Jetson | Pipeline latency distribution, sustained power/thermal behavior, deployment manifest |
| 4C — Inference | Correctness gate, operator profile, measured optimization and rejected hypothesis |
| 5A/B — Infrastructure / HPC | Topology, communication baseline, scaling curve and failure diagnosis |
| 5C — Edge | Quality/latency/memory tradeoff on the target device |
| 5D — Robotics | Task contract, reset tests, held-out seeds, closed-loop outcomes and timing |
| 5E — Autonomous systems | Replay evaluation, timing/coordinate contracts, failure analysis in simulation |
| 5F — Chip design | Workload-derived architecture, reference model, verification and PPA assumptions |
| 5G — ML systems | Pinned workload, quality gate, service-level latency and useful throughput |
| 6 — Interviews | Short design explanation backed by one of the artifacts above |

Review the [authoring guide](Curriculum-Authoring-Guide.md) when contributing a lesson and the [curriculum review](Curriculum-Review.md) for the scope and limits of the repository audit.
