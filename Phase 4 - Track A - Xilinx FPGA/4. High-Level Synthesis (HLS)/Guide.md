# 4. High-Level Synthesis (HLS)

> Use C/C++ to generate FPGA hardware, then verify whether the generated RTL actually meets throughput, area, memory, and interface requirements.

**Layer mapping:** L2, L5, and L6. HLS connects algorithm code, compiler scheduling, memory architecture, RTL generation, and FPGA implementation.

**Role targets:** HLS Engineer · FPGA Acceleration Engineer · ML Compiler/Hardware Co-design Engineer · AI Accelerator Prototyping Engineer

**Prerequisites:** [Xilinx FPGA Development](../1.%20Xilinx%20FPGA%20Development/Guide.md), [Advanced FPGA Design](../3.%20Advanced%20FPGA%20Design/Guide.md), C/C++, and basic performance profiling.

**What comes after:** [Runtime and Driver Development](../5.%20Runtime%20and%20Driver%20Development/Guide.md), [ML Compiler and Graph Optimization](../../Phase%204%20-%20Track%20C%20-%20ML%20Compiler%20and%20Graph%20Optimization/Guide.md), and [AI Chip Design](../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/6.%20AI%20Chip%20Design/Guide.md).

---

## Why This Module Exists

HLS is useful when it shortens the path from algorithm to hardware. It is dangerous when it hides the hardware cost of memory, loops, interfaces, and scheduling.

The HLS question is always:

```text
What hardware did this C/C++ imply, and is that hardware better than the CPU/GPU/RTL alternative?
```

This module teaches HLS as a compiler and hardware-design discipline, not as a shortcut around RTL understanding.

---

## Course Outcomes

By the end, you should be able to:

- write synthesizable C/C++ for HLS without accidental hardware explosions
- read HLS scheduling, latency, initiation interval, and resource reports
- use pipelining, unrolling, array partitioning, and dataflow directives intentionally
- design AXI-Lite, AXI memory-mapped, and AXI-Stream interfaces
- verify HLS blocks with C simulation, co-simulation, and RTL integration
- compare HLS-generated hardware against CPU, GPU, and hand-written RTL baselines
- document design-space tradeoffs with numbers

---

## Unit Map

| Unit | Focus | Artifact |
|------|-------|----------|
| 1 | HLS flow and reports | synthesized baseline kernel |
| 2 | Loops and pipelining | initiation-interval experiment |
| 3 | Memory architecture | array partitioning/banking report |
| 4 | Dataflow design | streaming pipeline with backpressure |
| 5 | Interfaces | AXI-connected HLS IP block |
| 6 | Verification | C sim, co-sim, and RTL validation package |
| 7 | Design-space exploration | Pareto table for latency/area/power |

---

## Unit 1: HLS Flow And Reports

### Learn

- C/C++ to RTL transformation
- synthesis, C simulation, co-simulation, export
- latency, initiation interval, tripcount, and resource estimates
- why estimates can differ from implemented results
- fixed-point versus floating-point implications

### Build It

Implement three baseline kernels:

1. vector add
2. FIR filter or convolution-like stencil
3. matrix-vector multiply

Synthesize each without aggressive directives first.

### Measure It

- latency
- initiation interval
- LUT/FF/BRAM/DSP estimate
- achieved clock target
- difference between HLS estimate and implementation if exported

### Ship It

A baseline HLS report explaining what hardware the code implied.

---

## Unit 2: Loops And Pipelining

### Learn

- loop pipelining
- loop unrolling
- loop-carried dependencies
- initiation interval limits
- resource sharing
- latency versus throughput

### Build It

Take the matrix-vector or FIR kernel and run variants:

- no pipeline
- pipelined loop
- unrolled loop
- pipelined + unrolled

### Measure It

- initiation interval
- latency
- throughput
- DSP usage
- BRAM pressure
- timing after implementation

### Ship It

A table showing which directive improved throughput and what it cost.

---

## Unit 3: Memory Architecture

### Learn

- array partitioning
- array reshaping
- memory banking
- BRAM versus URAM versus registers
- burst access
- data reuse
- memory bandwidth as the common HLS bottleneck

### Build It

Optimize a memory-bound kernel:

- matrix-vector multiply
- small convolution
- histogram
- feature extractor

Use partitioning or banking to feed parallel compute.

### Measure It

- read/write ports required
- BRAM/URAM usage
- achieved II
- burst efficiency
- throughput before/after memory changes

### Ship It

A memory architecture report with diagrams for data movement and storage.

---

## Unit 4: Dataflow Design

### Learn

- task-level parallelism
- `dataflow` regions
- FIFOs and streams
- producer/consumer balance
- backpressure
- deadlock risks
- pipeline fill and drain behavior

### Build It

Create a three-stage streaming pipeline:

```text
load -> transform -> store
```

Then split transform into two stages and add FIFO sizing experiments.

### Measure It

- stage latency
- end-to-end throughput
- FIFO depth sensitivity
- stall behavior
- resource cost

### Ship It

A streaming pipeline benchmark with a backpressure or deadlock debugging note.

---

## Unit 5: Interfaces

### Learn

- AXI4-Lite control interfaces
- AXI memory-mapped master interfaces
- AXI4-Stream interfaces
- register maps
- DMA integration
- host software control path
- interface timing and protocol constraints

### Build It

Package one HLS kernel as IP with:

- AXI-Lite control
- AXI memory or stream data path
- test application or driver
- block design integration

### Measure It

- host-to-kernel setup latency
- transfer throughput
- kernel throughput
- end-to-end acceleration including data movement

### Ship It

AXI-connected HLS IP with register map, integration diagram, and benchmark.

---

## Unit 6: Verification

### Learn

- golden reference model
- C simulation
- C/RTL co-simulation
- test vector generation
- numerical tolerance for fixed point
- waveform inspection
- integration testing with RTL or software

### Build It

Build a verification package for one HLS kernel:

- C reference
- randomized tests
- edge-case tests
- fixed-point comparison if relevant
- co-simulation run
- exported RTL integration smoke test

### Measure It

- test count
- coverage of edge cases
- numerical error
- co-simulation pass/fail logs

### Ship It

A verification report that would let another engineer trust the generated hardware.

---

## Unit 7: Design-Space Exploration

### Learn

- directive sweeps
- Pareto analysis
- latency/area/power tradeoffs
- automated report extraction
- when HLS is the wrong tool

### Build It

Run a parameter sweep across:

- unroll factors
- pipeline targets
- data widths
- array partitioning factors
- FIFO depths

### Measure It

- latency
- throughput
- LUT/FF/BRAM/DSP usage
- timing closure
- estimated power

### Ship It

A Pareto table and recommendation: which design point you would ship and why.

---

## Capstone

Build an HLS accelerator for a realistic kernel:

- image filter
- audio DSP block
- matrix-vector multiply
- quantized neural-network primitive
- packet-processing stage

Required evidence:

- CPU baseline
- HLS baseline
- optimized HLS version
- C sim and co-sim results
- implementation reports
- AXI integration
- end-to-end benchmark
- design-space table

The capstone is complete when the report proves whether HLS was a good implementation choice for the workload.

---

## Exit Criteria

You are ready to move on when you can:

- read HLS reports and predict implementation risk
- optimize loops and memory intentionally
- build streaming pipelines without deadlock
- integrate HLS IP into an AXI system
- verify generated RTL against a reference model
- compare speedup after including data movement and control overhead
- explain when hand-written RTL or a GPU kernel would be better
