# AI Hardware Engineering — Roles and Market Analysis

Choose work you want to learn, then check its requirements against current employer
postings. This page maps **23 technical specializations** to example titles and
portfolio evidence. The layer codes are this course's navigation scheme, not an
official occupational classification or a ranking of careers.

**Career sources checked: 19 October 2026.** Employer requirements and openings
change; refresh your own sample before applying. This page does not estimate
niche-role pay, remote-work shares, hiring volume, or market growth.

## How To Read This Data

### What the tables mean

| Field | Meaning | How to use it |
|---|---|---|
| Focus | The engineering problem a specialization addresses | Choose a problem you enjoy investigating |
| Typical roles / titles | Search terms; employer titles vary and can span layers | Read responsibilities and required qualifications |
| Portfolio evidence | A suggested learning artifact, not an employer's universal requirement | Demonstrate a result, a failure case, and reproducible steps |

### What this data is for

Decide which subsystem to study, what evidence to produce, and which adjacent
skill would help you collaborate. A learner interested in robotics might start
with L1c perception/control and L4a timing, then add L1b deployment.

### Interpretation rules

- Distinguish required qualifications from preferred qualifications in each posting.
- Check location, work arrangement, travel, lab access, and experience directly.
  Access to a cloud GPU does not determine an employer's remote-work policy.
- A broad occupational wage or projection does not measure a niche AI role.
  A posted base-pay range is not total compensation or a guaranteed offer.
- Completing course pages does not establish a job level. Assess demonstrated
  scope, independent debugging, collaboration, and relevant experience.

## How To Use This Page With The Roadmap

Start by choosing one **primary sub-layer** and one **adjacent sub-layer**:

- Example: `L1a Inference Optimization` + `L2c Kernel Engineering`
- Example: `L4b Embedded Linux & BSP` + `L3b Linux Kernel & Drivers`
- Example: `L5a Accelerator Architecture` + `L6a RTL Design`

Then map those targets back to the roadmap:

- **L1 / L2 roles:** prioritize [Phase 3](Phase%203%20-%20Artificial%20Intelligence/Guide.md) and the relevant Phase 4 track: [Xilinx FPGA](Phase%204%20-%20Track%20A%20-%20Xilinx%20FPGA/1.%20Xilinx%20FPGA%20Development/Guide.md), [NVIDIA Jetson](Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/1.%20Nvidia%20Jetson%20Platform/Guide.md), or [ML Compiler](Phase%204%20-%20Track%20C%20-%20DL%20Inference%20Optimization/Guide.md)
- **L3 / L4 roles:** prioritize [Phase 1](Phase%201%20-%20Foundational%20Knowledge/Guide.md), [Phase 2](Phase%202%20-%20Embedded%20Systems/Guide.md), and selected Phase 4 tracks
- **L5 / L6 roles:** prioritize [Phase 1](Phase%201%20-%20Foundational%20Knowledge/Guide.md), [Phase 4 Track A](Phase%204%20-%20Track%20A%20-%20Xilinx%20FPGA/1.%20Xilinx%20FPGA%20Development/Guide.md), and [Phase 5F](Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20F%20-%20AI%20Chip%20Design/Guide.md)

Use [Start here](Start-Here.md) for the first robot and AI experiments, and the
[learning guide](Learning-Guide.md) to select prerequisites and assess your work.

---

## Layer Map — 8 Layers, 23 Sub-Layers

| Layer | Sub-Layer | Focus | Typical Roles |
|:-----:|:---------:|-------|---------------|
| **L1** | **L1a** Inference Optimization | Model optimization, profiling, quantization | ML Inference Engineer, AI Performance Engineer |
| | **L1b** Edge AI Deployment | On-device pipelines, Jetson/MCU, power-constrained | Edge AI Engineer, Embedded AI Engineer |
| | **L1c** AI Application | Vision, robotics, solutions engineering | CV Engineer, Robotics AI Engineer |
| | **L1d** Agentic AI & GenAI | LLM agents, RAG, tool use, GenAI applications | Agentic AI Engineer, GenAI Engineer, AI Engineer |
| | **L1e** ML Engineering & MLOps | Training pipelines, model lifecycle, deployment infra | ML Engineer, MLOps Engineer, AI/ML Engineer |
| **L2** | **L2a** Graph & IR Optimization | Fusion, memory planning, graph passes | DL Graph Optimization Engineer |
| | **L2b** Compiler Backend | LLVM, MLIR, code generation, custom targets | AI Compiler Engineer, GPU Compiler Engineer |
| | **L2c** Kernel Engineering | Triton, CUTLASS, Flash-Attention, hand-tuned kernels | Kernel Optimization Engineer, MTS Kernels |
| **L3** | **L3a** GPU/Accelerator Runtime | CUDA runtime, TensorRT execution, DLA scheduling | GPU Runtime Engineer, Inference Platform Engineer |
| | **L3b** Linux Kernel & Drivers | nvgpu, amdgpu, PCIe, DMA, IOMMU, device tree | Linux Kernel Engineer, Device Driver Engineer |
| | **L3c** HPC Infrastructure | NCCL, Slurm, K8s+GPU, GPUDirect, multi-node | Distributed Runtime Engineer, Resource Scheduler |
| **L4** | **L4a** Embedded Software | MCU, FreeRTOS, bare-metal, SPI/I2C/CAN | Embedded Software Engineer, RTOS Engineer |
| | **L4b** Embedded Linux & BSP | Yocto, L4T, kernel modules, OTA, rootfs | Embedded Linux Engineer, BSP Engineer |
| | **L4c** Automotive & IoT | ADAS firmware, ISO 26262, BLE/LoRa, cloud IoT | Automotive Embedded Engineer, IoT Engineer |
| **L5** | **L5a** Accelerator Architecture | Systolic arrays, dataflow, tensor core design | AI Accelerator Architect, GPU Architect |
| | **L5b** System & SoC Architecture | NoC, memory hierarchy, power domains, chiplet partitioning | SoC Architect, Memory Systems Architect |
| **L6** | **L6a** RTL Design | SystemVerilog datapaths, FSMs, IP implementation | RTL Design Engineer, ASIC Design Engineer |
| | **L6b** Design Verification | UVM, formal, constrained random, emulation | DV Engineer, Emulation Engineer |
| | **L6c** FPGA & HLS | Vivado, HLS, FPGA prototyping, timing closure | FPGA Engineer, HLS Engineer |
| **L7** | **L7a** Physical Design | P&R, floorplan, CTS, timing closure, signoff | Physical Design Engineer, STA Engineer |
| | **L7b** DFT & CAD | Scan insertion, ATPG, tool flow automation | DFT Engineer, CAD Engineer |
| **L8** | **L8a** Packaging & Process | CoWoS, chiplets, foundry interface, yield | Packaging Engineer, Process Engineer |
| | **L8b** Silicon Validation | Post-silicon bring-up, characterization, ATE | Validation Engineer, Test Engineer |

---

## L1a — Inference Optimization

**Focus:** Make ML models run faster — graph optimization, quantization, profiling, TensorRT/vLLM tuning.

| Title | What they do |
|-------|-------------|
| ML Inference Optimization Engineer | Graph-level optimization, TensorRT engine builds, INT8/FP8 quantization |
| AI Performance Engineer | Nsight profiling, roofline analysis, bottleneck identification |
| Applied ML Engineer (Inference) | Take research models to production with latency/throughput SLAs |

**Portfolio evidence:** Compare a model with a reference on held-out inputs; report quality, peak memory, and latency with the timing boundary and device named.

---

## L1b — Edge AI Deployment

**Focus:** Deploy inference on resource-constrained hardware — Jetson, Snapdragon, MCU, TinyML.

| Title | What they do |
|-------|-------------|
| Edge AI Deployment Engineer | Jetson/Snapdragon on-device inference, power/latency targets |
| Edge AI Engineer | Full pipeline: sensor → preprocess → inference → actuation |
| Embedded AI Engineer | TFLite Micro, TinyML on Cortex-M, ultra-low-power inference |

**Portfolio evidence:** Build a timestamped sensor-to-model pipeline. Detect stale observations, bound queues, and measure power and latency on the named device when available.

---

## L1c — AI Application (CV / Robotics / Solutions)

**Focus:** Domain-specific AI deployment — vision, robotics, customer-facing solutions.

| Title | What they do |
|-------|-------------|
| Computer Vision Engineer (Edge AI) | Detection, segmentation, tracking on constrained devices |
| Robotics AI Engineer | Perception + planning on robot hardware (ROS 2, Jetson) |
| AI Application Engineer | SDK integration, demo systems, customer deployment |
| AI Solutions Engineer | Pre/post-sales technical, benchmark customer workloads |

**Portfolio evidence:** Evaluate perception and a controller on held-out scenes or episodes; include a coordinate-frame error and a recovery case. Label simulated results.

---

## L1d — Agentic AI & GenAI

**Focus:** Build applications on top of large language models — agents, RAG pipelines, tool-use orchestration, GenAI products.

| Title | What they do |
|-------|-------------|
| Agentic AI Engineer | Design multi-step AI agents: tool use, planning, memory, tool routing, execution state, and evaluation |
| GenAI Engineer | Build GenAI-powered products: chatbots, code assistants, content generation, multimodal apps |
| AI Engineer | Full-stack AI application development: prompt engineering, fine-tuning, API integration, evaluation |

**Portfolio evidence:** Build a small retrieval or tool-use application with source citations, validated tool arguments, bounded retries, and an evaluation set containing failures.

---

## L1e — ML Engineering & MLOps

**Focus:** Build and operate ML training/inference pipelines — model lifecycle, data pipelines, serving infrastructure, monitoring.

| Title | What they do |
|-------|-------------|
| Machine Learning Engineer | Training pipelines, model architecture, feature engineering, evaluation |
| MLOps Engineer | CI/CD for models, experiment tracking (MLflow, W&B), model registry, serving infra |
| AI/ML Engineer | End-to-end: data → training → optimization → deployment → monitoring |
| ML Platform Engineer | Build internal ML platforms: compute scheduling, data versioning, model serving |
| Data/ML Infrastructure Engineer | GPU cluster management for training, distributed data pipelines |

**Portfolio evidence:** Version a dataset split and training configuration; reproduce a model, save/reload it, and demonstrate how a failed quality check blocks deployment.

---

## L2a — Graph & IR Optimization

**Focus:** Compiler front-end — graph passes, operator fusion, memory planning, layout transforms.

| Title | What they do |
|-------|-------------|
| DL Graph Optimization Engineer | Fusion passes, constant folding, memory planning, quantization insertion |
| AI Systems Compiler Engineer | Distributed graph partitioning, multi-device scheduling |
| Performance Compiler Engineer | Search-based tuning, cost models, roofline-guided pass selection |

**Portfolio evidence:** Implement one graph rewrite with its legality conditions. Compare against an unfused reference, including shapes for which the rewrite must be rejected.

---

## L2b — Compiler Backend (LLVM / MLIR / TVM)

**Focus:** Compiler infrastructure — IR design, lowering passes, instruction selection, code generation for GPU/NPU/TPU.

| Title | What they do |
|-------|-------------|
| AI Compiler Engineer | Full ML compiler: ONNX → IR → optimized target code |
| ML Compiler Backend Engineer | Target-specific codegen: NVPTX, AMDGPU, custom accelerator ISA |
| Compiler Engineer (LLVM/MLIR/TVM) | Framework-level: write passes, define dialects, implement lowering |
| GPU Compiler Engineer | GPU-specific: register allocation, occupancy, instruction scheduling |
| Code Generation Engineer | Custom NPU/TPU backend: ISA design, instruction selection, scheduling |

**Portfolio evidence:** Lower a small operator to a documented target. Include IR before/after, compiler tests, and a reference comparison for edge cases.

---

## L2c — Kernel Engineering

**Focus:** Hand-write or tune the actual GPU/accelerator kernels — Triton, CUTLASS, Flash-Attention, NCCL kernels.

| Title | What they do |
|-------|-------------|
| Kernel Optimization Engineer | Triton/CUTLASS kernel authoring, tiling, memory optimization |
| MTS Kernels (Member of Technical Staff) | Production attention/GEMM kernels for LLM training and inference |
| HPC Compiler Engineer | Vectorization, parallelization for scientific computing |

**Portfolio evidence:** Implement and validate a small kernel, explain its data movement, and measure it only on a supported target. A CPU model can explain arithmetic but cannot establish GPU speed.

---

## L3a — GPU/Accelerator Runtime

**Focus:** Execution layer — CUDA runtime, TensorRT engine execution, DLA scheduling, memory management.

| Title | What they do |
|-------|-------------|
| GPU Runtime Engineer | CUDA runtime internals: streams, events, memory pools, context |
| Accelerator Runtime Engineer | Custom NPU/TPU runtime: command queue, buffer management |
| Inference Platform Engineer | TensorRT engine execution, Triton server, dynamic batching |
| CUDA Runtime Engineer | Driver API, module loading, JIT compilation, multi-context |

**Portfolio evidence:** Show buffer ownership and event dependencies for an asynchronous execution queue; test cancellation, invalid shapes, and resource cleanup.

---

## L3b — Linux Kernel & Drivers

**Focus:** Kernel-space GPU/accelerator drivers, DMA, PCIe, IOMMU, interrupt handling.

| Title | What they do |
|-------|-------------|
| Linux Kernel Engineer (GPU/Drivers) | nvgpu, amdgpu, DRM, IOMMU/SMMU, memory-mapped I/O |
| Device Driver Engineer | PCIe/CXL endpoint driver, DMA engine, scatter-gather |
| Embedded Linux BSP Engineer | Yocto kernel customization, device tree, L4T, rootfs |

**Portfolio evidence:** Document a small device interface and test its bounds, lifetime, and error handling. Distinguish a userspace mock from a kernel driver tested with hardware.

---

## L3c — HPC Infrastructure

**Focus:** Multi-GPU/multi-node systems — NCCL, Slurm, Kubernetes+GPU, GPUDirect, InfiniBand.

| Title | What they do |
|-------|-------------|
| Distributed Runtime Engineer | NCCL tuning, AllReduce overlap, multi-node communication |
| Resource Scheduler Engineer | Slurm, K8s GPU scheduling, MIG/MPS, multi-tenant GPU sharing |
| Parallel Computing Engineer | MPI+CUDA, GPUDirect RDMA, storage I/O optimization |
| Systems Software Engineer (AI) | Full-stack performance: CPU-GPU coordination, profiling, debugging |

**Portfolio evidence:** Produce a topology diagram and a repeatable communication experiment. Identify ranks, message sizes, synchronization, and a diagnosed failure.

---

## L4a — Embedded Software (MCU / RTOS)

**Focus:** Bare-metal and RTOS firmware on microcontrollers — the hardware-closest software layer.

| Title | What they do |
|-------|-------------|
| Embedded Software Engineer | ARM Cortex-M/R, FreeRTOS/Zephyr, SPI/I2C/UART/CAN drivers |
| Firmware Engineer (AI/Edge SoC) | Command processor firmware, DMA scheduling, power management |
| Real-Time Systems Engineer | Scheduling, bounded timing analysis, RTOS or PREEMPT_RT integration |
| Bootloader / UEFI Engineer | U-Boot, UEFI, secure boot chain, A/B partition management |

**Portfolio evidence:** Implement a bounded producer/consumer path and a timing budget. Inject a missing sample or overloaded consumer, then show recovery without unbounded allocation.

---

## L4b — Embedded Linux & BSP

**Focus:** Linux kernel customization, board support packages, Yocto, OTA, production images.

| Title | What they do |
|-------|-------------|
| Embedded Linux Engineer | Yocto/Buildroot, kernel config, systemd, rootfs optimization |
| BSP Engineer | Board bring-up, device tree, driver integration, HAL |
| Jetson Platform Engineer | L4T customization, JetPack, carrier board bring-up, SPE firmware |

**Portfolio evidence:** Record an image build, boot configuration, peripheral contract, and rollback test. Explain which steps ran on a board and which used an emulated target.

---

## L4c — Automotive & IoT

**Focus:** Domain-specific firmware — ADAS safety standards, connected IoT, fleet management.

| Title | What they do |
|-------|-------------|
| Automotive Embedded Engineer (ADAS) | ISO 26262, AUTOSAR, ECU firmware, CAN/CAN-FD, functional safety |
| IoT Firmware Engineer | Low-power wireless (BLE, LoRa, Wi-Fi), OTA, cloud connectivity |
| Device Firmware Engineer | Storage controllers, NIC firmware, PCIe endpoint firmware |

**Portfolio evidence:** Write a device state machine with disconnect/reconnect and update-failure tests. Treat safety requirements and compliance evidence as separate work from a functioning demonstration.

---

## L5a — Accelerator Architecture

**Focus:** Define the compute engine — systolic arrays, dataflow, tensor core specs, PE design.

| Title | What they do |
|-------|-------------|
| AI Accelerator Architect | Systolic array dimensions, dataflow strategy, precision support |
| GPU Architect | SM/CU microarchitecture, warp scheduler, tensor core design |
| ML Systems Architect | Workload analysis → architecture decisions, hardware-software co-design |
| Performance Architect | Roofline modeling, bottleneck analysis, workload characterization |

**Portfolio evidence:** Use a real workload trace to compare two dataflows; account for operations, memory traffic, precision, and utilization. State the assumptions behind estimated area or power.

---

## L5b — System & SoC Architecture

**Focus:** Full-chip system design — NoC, memory hierarchy, I/O, power domains, chiplet partitioning.

| Title | What they do |
|-------|-------------|
| SoC Platform Engineer | Zynq/Versal PS-PL co-design, AXI interconnect, IP integration |
| Silicon Architect | Die floorplan, chiplet partitioning, power/thermal budgets |
| Memory Systems Architect | HBM controller, cache hierarchy, scratchpad design |
| Heterogeneous Computing Architect | CPU+GPU+NPU+DSP integration, coherency, shared memory |
| Edge AI Systems Architect | Accelerator design within a stated power and thermal budget |

**Portfolio evidence:** Build a memory and interconnect budget with explicit initiators, address ranges, coherency assumptions, and backpressure; check one bottleneck with a small model.

---

## L6a — RTL Design

**Focus:** Implement the architecture in synthesizable HDL — datapaths, controllers, interfaces.

| Title | What they do |
|-------|-------------|
| RTL Design Engineer | SystemVerilog implementation: PE arrays, FSMs, AXI interfaces |
| ASIC Design Engineer | Tape-out quality RTL, CDC handling, synthesis constraints |
| Logic Design Engineer | Combinational/sequential optimization, area/power trade-offs |
| SoC Integration Engineer | IP block integration, address maps, subsystem-level wiring |

**Portfolio evidence:** Implement a synthesizable block with reset, handshake, and overflow behavior; provide a reference testbench and synthesis constraints.

---

## L6b — Design Verification

**Focus:** Check RTL against its specification using simulation, scoped formal properties, coverage, and emulation.

| Title | What they do |
|-------|-------------|
| Design Verification Engineer | UVM constrained random, coverage-driven verification, assertions |
| Formal Verification Engineer | Property checking, equivalence checking for critical blocks |
| Emulation Engineer | Palladium/Zebu, run firmware on RTL at MHz for pre-silicon validation |

**Portfolio evidence:** Write a verification plan linking requirements to assertions and tests; insert a bug and show a counterexample. Explain what coverage does not prove.

---

## L6c — FPGA & HLS

**Focus:** FPGA prototyping, HLS-based accelerator design, Vivado/Quartus.

| Title | What they do |
|-------|-------------|
| FPGA Design Engineer | Vivado/Quartus, timing closure, IP integration, ILA debug |
| HLS Engineer | C/C++ to RTL (Vitis HLS), pragma optimization, dataflow |
| Hardware Design Engineer | General digital design on FPGA: interfaces, controllers |

**Portfolio evidence:** Implement a small accelerator and compare software/RTL outputs. Show constraints and achieved timing if the tool ran; report board execution separately.

---

## L7a — Physical Design

**Focus:** Synthesis, place & route, timing closure, power integrity, signoff.

| Title | What they do |
|-------|-------------|
| Physical Design Engineer | Floorplan, placement, CTS, routing, congestion management |
| STA Engineer | PrimeTime, setup/hold analysis, MCMM, OCV, timing ECOs |
| Power Integrity Engineer | IR drop, EM analysis, power grid design, DVFS planning |

**Portfolio evidence:** Run an accessible physical-design flow for a small block and explain timing, congestion, and constraint assumptions. An educational PDK result is not production signoff.

---

## L7b — DFT & CAD

**Focus:** Test insertion, ATPG, tool flow automation, methodology.

| Title | What they do |
|-------|-------------|
| DFT Engineer | Scan insertion, ATPG, BIST, fault coverage optimization |
| CAD/EDA Engineer | Tool flow scripts, methodology, custom EDA automation |
| Layout Engineer | Standard cell placement, DRC/LVS clean, metal optimization |

**Portfolio evidence:** Create a reproducible flow script or small scan/fault-model exercise; report tested faults, coverage denominator, and limitations.

---

## L8a — Packaging & Process

**Focus:** Advanced packaging, foundry interface, yield, process selection.

| Title | What they do |
|-------|-------------|
| Packaging Engineer | CoWoS, EMIB, Foveros, chiplet integration, substrate design |
| Process Integration Engineer | Foundry relationship, process node selection, yield optimization |
| Reliability Engineer | Burn-in, electromigration, thermal cycling, qualification |
| Supply Chain Engineer | Wafer allocation, lead time, foundry contracts |

**Portfolio evidence:** Compare two packaging options using documented electrical, thermal, and manufacturing constraints. Separate calculations from measured qualification data.

---

## L8b — Silicon Validation

**Focus:** Post-silicon bring-up, characterization, ATE programming, production test.

| Title | What they do |
|-------|-------------|
| Post-Silicon Validation Engineer | First silicon bring-up, debug, speed/power characterization |
| Test Engineer | ATE programming, production test development, yield analysis |

**Portfolio evidence:** Write a bring-up and characterization plan with instrument limits, calibration, test conditions, and failure triage. Use synthetic traces only when clearly labelled.

---

## Market Size & Industry Context

Use primary sources at the level they actually measure. These resources were
reviewed on **19 October 2026**:

| Primary source | What it supports | What it does not establish |
|---|---|---|
| [BLS: Computer Hardware Engineers](https://www.bls.gov/ooh/architecture-and-engineering/computer-hardware-engineers.htm) | Broad US occupation duties, preparation, wages, and employment projections | Pay or vacancy counts for a specific accelerator architect or compiler team |
| [BLS: Software Developers, Quality Assurance Analysts, and Testers](https://www.bls.gov/ooh/computer-and-information-technology/software-developers.htm) | Broad software occupations and their duties and outlook | A separate national count or growth rate for agentic AI, MLOps, or GPU kernel roles |
| [BLS: Electrical and Electronics Engineers](https://www.bls.gov/ooh/architecture-and-engineering/electrical-and-electronics-engineers.htm) | Broader electrical/electronic engineering work and preparation | An estimate specific to this course's embedded, FPGA, or packaging categories |
| [Apple: Hardware teams](https://www.apple.com/careers/us/work-at-apple/teams/hardware.html) | Employer descriptions of firmware, architecture, silicon verification, packaging, and related teams; links to available roles | A representative survey of employers or evidence that every listed team currently has a suitable opening |
| [NVIDIA: Careers](https://www.nvidia.com/en-us/about-nvidia/careers/) | Official routes to current openings, university recruiting, and hiring information | A count of unique niche-role vacancies or a promise of work arrangement |

BLS employment projections describe a broad occupation over a stated period.
Projected annual openings include replacement needs as well as growth; they are
not a count of live job advertisements. Keep the publication date, reference
period, geography, and occupation together when using a statistic. The BLS
profiles above provide those definitions and links to local wage data.

Employer team descriptions help interpret responsibilities. For example, Apple's
hardware page separates embedded firmware from silicon design/verification and
packaging. This is useful evidence that the tasks differ; it does not establish
which specialization is easier to enter or better paid. Follow its current-role
links to inspect the actual requirements.

Search-engine snippets can outlive a vacancy. Confirm the employer's live
description before treating an advertisement as open; a working URL alone does
not establish that applications are still being accepted.

## Job Posting Volume by Sub-Layer

A small personal search can identify relevant opportunities. It cannot establish
national posting volume, a shortage, or a growth ranking. The same requisition
may appear on several boards or under several locations. Count an employer's
requisition once in your sample, recording the locations separately. A general
team-interest page is not automatically an open requisition.

### Build a role-selection worksheet

Allow one study session. Choose one primary specialization and one adjacent
specialization from the map. Find **three current postings from at least two
employers** in locations and employment arrangements you could consider. If you
cannot find three suitable postings, record that outcome and revise your search;
do not invent missing examples.

For each posting, save this record in your own notes:

```text
employer and official URL:
requisition ID, title, location, date checked:
posting status: open | closed | unclear
required qualifications:
preferred qualifications:
work arrangement / travel / lab access: stated value or unknown
primary and adjacent layer, with one responsibility supporting each:
my existing evidence:
important gap and a small project that would address it:
```

Select **two required technical tasks** you can meaningfully practice. Link each
to a course lab, a testable acceptance condition, and an artifact someone else can
rerun. Record qualifications that a lab cannot replace, such as required work
experience or an employer's stated degree/equivalent-experience conditions.

### Worked selection example

Suppose a posting you read asks for embedded C/C++, sensor interfaces, and timing
debugging. You enjoy investigating dropped messages and already know basic C.
A reasonable primary choice is **L4a**, with **L1b** as the adjacent application.
Build a timestamped sensor queue, inject a delayed sample, and show that the
consumer rejects it. Report queue bounds and timing assumptions. A CPU test can
establish the queue logic; validating electrical signals and timing on a board
requires separate evidence.

This is an **illustrative decision**, not a claim about a particular vacancy.
Use [Embedded Systems](Phase%202%20-%20Embedded%20Systems/Guide.md) for the interface
work and [robotics foundations](Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20D%20-%20Robotics/Robotics%20Foundations/README.md)
for timestamps, frames, and control. For an AI-oriented alternative, use the
[CPU policy lab](Phase%203%20-%20Artificial%20Intelligence/0.%20AI%20and%20Robot%20Learning%20Foundations/Lab-01-CPU-Policy.md)
to demonstrate split/training/evaluation evidence before adding deployment.

### Check your interpretation

1. Five job-board results share one employer requisition ID. How many distinct
   requisitions have you demonstrated? **One**, unless the employer identifies
   separate requisitions; you have not demonstrated the number of hires.
2. A broad BLS occupation has a projected increase. Does that establish growth
   for L2c kernel engineering? **No.** The categories and measurement differ.
3. A posting does not mention remote work. Can you label it remote because its
   code runs in the cloud? **No.** Mark the arrangement unknown and verify it.
4. Your project passes all CPU tests. Are you qualified for a senior role?
   **That conclusion does not follow.** Explain the demonstrated task and limits;
   compare the remaining experience and responsibility requirements separately.

## Cross-Layer Summary

Choose an adjacent layer that explains a failure you cannot yet diagnose:

| Primary work | Useful adjacent work | Connected portfolio question |
|---|---|---|
| L1c robotics application | L4a embedded timing | Does the controller reject an old observation? |
| L1a inference optimization | L2c kernel engineering | Does a proposed speedup preserve the quality gate and numerical contract? |
| L2b compiler backend | L5a architecture | Does the generated code fit the target's instructions and memory limits? |
| L4b board support | L3b drivers | Can a peripheral failure be traced through configuration, driver, and application? |
| L6a RTL design | L6b verification | Which test or property detects a deliberately broken handshake? |
| L5b system architecture | L7a physical design | Which timing and power assumptions need implementation evidence? |

### Work Arrangement Summary

Read the individual posting for office location, remote eligibility, travel,
secure facilities, and lab access. If those are unclear, list them as questions
for the employer. A role's software/hardware label alone does not answer them.

## Hiring Priority for an AI Chip Startup

As a course exercise, derive required capabilities from a product milestone:
workload and quality definition, compiler/runtime support, architecture and RTL,
verification, physical feasibility, packaging, bring-up, and application testing.
These activities constrain one another. For example, memory traffic affects both
the architecture and compiler, while timing or package limits can invalidate an
early performance estimate.

Draw the dependencies and assign an owner and acceptance artifact to each.
Explain which capability is missing for the *next* milestone. This is a project
planning exercise, not a universal hiring order or an estimate of a startup's
headcount, payroll, or equity costs. The [Physical AI Design Studio](Physical-AI-Design-Studio.md)
provides concrete handoffs for this exercise.

## Where This Roadmap Takes You

| Learning route | Evidence to bring to a role discussion |
|---|---|
| AI and robot learning | Held-out evaluation, baseline, failure analysis, and reproducible model artifact |
| Embedded / Jetson | Interface and timing contracts, a recovery test, and named execution hardware |
| Compiler / inference | Reference-checked transformation or operator, documented constraints, and measured results where available |
| FPGA / RTL / verification | Testbench or formal properties, reset/handshake evidence, constraints, and tool reports |
| Architecture / chip design | Workload-derived tradeoffs, reference models, verification plan, and explicit implementation assumptions |

Complete the worksheet with one chosen role, two requirement-to-evidence links,
a runnable artifact, one limitation, and the next prerequisite you will study.
Use the [completion rubric](Learning-Guide.md#completion-rubric) to review the
artifact. Ask another learner to reproduce it and explain one failure case.
The result is evidence of specific skills; employment level and hiring decisions
remain tied to the actual role, experience, and employer assessment.
