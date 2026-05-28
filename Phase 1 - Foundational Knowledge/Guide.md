# Phase 1: Digital Foundations

<div class="course-identity digital-foundations" markdown="1">
<div class="course-identity__icon">P1</div>
<div markdown="1">
<p class="course-identity__eyebrow">Phase 1 · Digital Foundations</p>
<p class="course-identity__title">Logic, architecture, operating systems, and parallel execution form the base of the whole roadmap.</p>
<p class="course-identity__meta">Artifact: RTL block + profiler note · Measure: timing, bandwidth, utilization</p>
</div>
</div>


> *Learn how computation is represented, executed, scheduled, and accelerated before you try to deploy AI on real hardware.*

**Layer mapping:** Primarily **L5** (hardware architecture) and **L6** (RTL / logic design), with an important bridge into **L3** (runtime behavior) through operating systems and parallel computing.

**Role targets:** RTL Design Engineer · FPGA Engineer · GPU Runtime Engineer · AI Compiler Engineer · AI Accelerator Architect

**Prerequisites:** comfort with basic command-line tooling and a working development environment

**What comes after:** [Phase 2 — Embedded Systems](../Phase%202%20-%20Embedded%20Systems/Guide.md), [Phase 3 — Artificial Intelligence](../Phase%203%20-%20Artificial%20Intelligence/Guide.md), then one of the Phase 4 tracks: [Xilinx FPGA](../Phase%204%20-%20Track%20A%20-%20Xilinx%20FPGA/1.%20Xilinx%20FPGA%20Development/Guide.md), [NVIDIA Jetson](../Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/1.%20Nvidia%20Jetson%20Platform/Guide.md), or [ML Compiler](../Phase%204%20-%20Track%20C%20-%20ML%20Compiler%20and%20Graph%20Optimization/Guide.md)

---

## Why This Phase Exists

Every later phase assumes you already understand the mechanics of computation:

- how logic turns into hardware behavior
- how processors execute instructions
- how operating systems manage memory and devices
- how parallel programs map work onto CPU and GPU hardware

If you skip this phase, later topics become tool usage instead of engineering.

---

## Phase Structure

| # | Module | What you learn | Why it matters |
|---|--------|----------------|----------------|
| **1** | [Digital Design & HDL](1.%20Digital%20Design%20and%20Hardware%20Description%20Languages/Guide.md) | Boolean logic, sequential systems, Verilog, testbenches | The language used to describe hardware |
| **2** | [Computer Architecture](2.%20Computer%20Architecture%20and%20Hardware/Guide.md) | ISA, pipelines, caches, memory systems, throughput vs latency | The design logic behind CPUs, GPUs, and NPUs |
| **3** | [Operating Systems](3.%20Operating%20Systems/Guide.md) | processes, memory, scheduling, synchronization, drivers | The software layer that manages hardware resources |
| **4** | [C++ and Parallel Computing](4.%20C%2B%2B%20and%20Parallel%20Computing/Guide.md) | SIMD, OpenMP, oneTBB, CUDA, HIP, SYCL | The execution models used by modern AI systems |

**Recommended order:** `1 → 2 → 3 → 4`

If you already know digital logic, you can move faster through Module 1. If you already know OS fundamentals, still do Module 4 carefully; it is the most important bridge into AI hardware work.

---

## What You Should Produce

This phase should leave you with visible low-level artifacts, not just notes.

- a small Verilog block plus a testbench
- an architecture explainer or comparison note for CPU vs GPU vs accelerator design
- a debugging write-up around memory, scheduling, or synchronization behavior
- at least one measured parallel program, ideally including a CUDA or GPU profiling artifact

Record those outputs in a simple engineering log, project README, or benchmark note so the work stays visible and reviewable.

---

## Exit Criteria

You are ready to move on when you can:

- read basic RTL and explain what hardware it implies
- reason about cache, memory bandwidth, and pipeline bottlenecks
- explain how the OS affects device access and concurrency behavior
- profile a simple parallel workload and describe whether it is compute-bound, memory-bound, or synchronization-bound

That is the minimum base for the rest of the roadmap.

---

## Who Should Prioritize This Phase

- **Hardware-first learners:** do the whole phase in order
- **ML engineers moving downward:** focus especially on Modules 2 and 4
- **Embedded engineers:** do Modules 2, 3, and 4 thoroughly even if Module 1 is familiar

---

## Next

→ [**Phase 2 — Embedded Systems**](../Phase%202%20-%20Embedded%20Systems/Guide.md) · [**Phase 3 — Artificial Intelligence**](../Phase%203%20-%20Artificial%20Intelligence/Guide.md)
