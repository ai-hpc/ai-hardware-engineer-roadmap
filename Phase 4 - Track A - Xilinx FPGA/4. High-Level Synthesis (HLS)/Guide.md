# 4. High-Level Synthesis (HLS)

<div class="course-identity hls" markdown="1">
<div class="course-identity__icon">HLS</div>
<div markdown="1">
<p class="course-identity__eyebrow">Track A4 · High-Level Synthesis</p>
<p class="course-identity__title">Use C/C++ to generate FPGA hardware and prove whether the generated RTL is worth shipping.</p>
<p class="course-identity__meta">Artifact: HLS accelerator + Pareto report · Measure: II, latency, area, data movement</p>
</div>
</div>


> Use C/C++ to generate FPGA hardware, then verify whether the generated RTL actually meets throughput, area, memory, and interface requirements.

**Layer mapping:** L2, L5, and L6. HLS connects algorithm code, compiler scheduling, memory architecture, RTL generation, and FPGA implementation.

**Role targets:** HLS Engineer · FPGA Acceleration Engineer · ML Compiler/Hardware Co-design Engineer · AI Accelerator Prototyping Engineer

**Prerequisites:** [Xilinx FPGA Development](../1.%20Xilinx%20FPGA%20Development/Guide.md), [Advanced FPGA Design](../3.%20Advanced%20FPGA%20Design/Guide.md), C/C++, and basic performance profiling.

**What comes after:** [Runtime and Driver Development](../5.%20Runtime%20and%20Driver%20Development/Guide.md), [ML Compiler and Graph Optimization](../../Phase%204%20-%20Track%20C%20-%20DL%20Inference%20Optimization/Guide.md), and [AI Chip Design](../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20F%20-%20AI%20Chip%20Design/Guide.md).

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

Start with the [native arithmetic and banking lab](labs/README.md). It executes a bounded int8 matrix-vector contract on a CPU, checks all 512 supported shapes, and catches deliberately broken tail/saturation behavior. This gives you a checked algorithm to carry into vendor C simulation; it does not establish HLS synthesis or FPGA performance.

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

### Worked example: latency is different from initiation interval

Suppose an accelerator accepts its first sample at edge 0 and produces that sample's result at edge 6: latency is 6 clocks. If it can accept a new sample every 2 clocks, its initiation interval (II) is 2. Five samples enter at edges 0, 2, 4, 6, 8 and leave at 6, 8, 10, 12, 14. With no stalls, the last result appears after `latency + (N - 1) × II = 14` clocks.

An **assumed** implemented clock of 100 MHz and II=2 would permit 50 million samples/s in steady state; this is a calculation, not a measured result. Memory access conflicts, dependent accumulations, and limited operators can prevent II=1. A pipeline directive requests an outcome; the report tells you what the tool achieved. See [AMD's loop-pipelining explanation](https://docs.amd.com/r/en-US/ug1399-vitis-hls/Pipelining-Loops).

Start with the [RTL streaming lab](../1.%20Xilinx%20FPGA%20Development/labs/README.md) to see how sink backpressure changes accepted transfers. Then implement the same byte saturation contract in HLS and compare accepted inputs/results. C simulation checks algorithm behavior; RTL co-simulation checks the generated interface under its test conditions; implementation timing is a separate result.

**Check:** if latency becomes 10 clocks but II drops from 2 to 1, is the design always better? **Answer:** no. It improves ideal steady-state rate, but increases first-result latency and may use more area or miss the clock target. Compare the application's batch size, response deadline, stalls, and implemented reports.

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

### Worked example: unrolling needs matching ports

Four column lanes require four weight reads and four vector reads for each group. If each array resides in a separate two-read-port memory, the ideal memory service bound is at least two cycles per group. Four suitable banks can reduce that bound, but the indexing must distribute the accesses: addresses `0,1,2,3` map to four cyclic banks, while `0,4,8,12` all map to one. Array dimension and stride matter. [Run the bank fixtures](labs/README.md#4-count-memory-demand-before-requesting-unroll), then compare the model with an actual HLS memory/scheduling report.

Partitioning creates separately accessible storage; reshaping widens storage access. Neither is a promise of external DDR bandwidth. Loading a tile into a suitable local layout may be necessary before the desired compute parallelism is possible.

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

### Worked example: finite buffering absorbs bursts, not permanent overload

For a FIFO, occupancy after an edge is previous occupancy plus accepted writes minus accepted reads. Count transfers only when their protocol accepts them. For accepted-write counts `[1,1,1,1,0,0]` and accepted-read counts `[0,0,1,0,1,1]` over six edges, occupancy from empty is `[1,2,2,3,2,1]`; this trace needs capacity at least three under the assumed simultaneous-transfer behavior. A different downstream stall can require more.

If a producer permanently generates two items per cycle and a consumer can take one, no finite FIFO sustains the mismatch without backpressure or dropping. In a cyclic dataflow graph, also examine initial tokens and blocking order: capacity alone does not repair every deadlock. C simulation may not model the same finite channel capacity and concurrent scheduling as RTL. Test fill, drain, reset, and downstream stalls in the generated interface.

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

**Study method:** use the [shared lesson cycle and completion rubric](../../Learning-Guide.md) with this course's existing labs. Predict first, test one variation, retain a failure case, and revisit the self-check after two days and one week.


## Integration lab: Which resource prevents the desired initiation interval?

Synthesize a matrix-vector kernel at unroll factors 1, 2, and 4. For each variant count required reads per cycle and available bank ports. Distinguish loop II from whole-invocation throughput, including DMA startup.

**Acceptance evidence:** Report C/RTL parity and achieved implementation timing. Explain one non-improving variant from the memory/dependency report, then pass the chosen interface and buffer alignment to the runtime course.

**Continue:** take this artifact to [5. Runtime and Driver Development](../5.%20Runtime%20and%20Driver%20Development/Guide.md). Explain which assumption from this course the next engineer must preserve.
