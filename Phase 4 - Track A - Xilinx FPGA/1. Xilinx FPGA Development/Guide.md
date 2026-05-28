# 1. Xilinx FPGA Development

> Learn the Xilinx FPGA flow by building, simulating, timing, debugging, and documenting real RTL projects.

**Layer mapping:** L5-L6. This module connects RTL design, FPGA implementation, timing closure, on-chip debug, and hardware validation.

**Role targets:** FPGA Engineer · RTL Design Engineer · Hardware Acceleration Engineer · AI Accelerator Prototyping Engineer

**Prerequisites:** [Digital Design and HDL](../../Phase%201%20-%20Foundational%20Knowledge/1.%20Digital%20Design%20and%20Hardware%20Description%20Languages/Guide.md), [Computer Architecture](../../Phase%201%20-%20Foundational%20Knowledge/2.%20Computer%20Architecture%20and%20Hardware/Guide.md), and basic command-line Git.

**What comes after:** [Zynq UltraScale+ MPSoC](../2.%20Zynq%20UltraScale%2B%20MPSoC/Guide.md), [Advanced FPGA Design](../3.%20Advanced%20FPGA%20Design/Guide.md), and [High-Level Synthesis](../4.%20High-Level%20Synthesis%20%28HLS%29/Guide.md).

---

## Why This Module Exists

Vivado is not the skill. The skill is turning a hardware idea into a verified bitstream that works on a board and meets timing.

This module teaches the full FPGA loop:

```text
RTL -> simulation -> synthesis -> implementation -> timing -> bitstream -> board debug -> report
```

Do not treat the tool as a button-clicking IDE. Treat it as an engineering flow that produces artifacts another hardware engineer can review.

---

## Course Outcomes

By the end, you should be able to:

- create a clean Vivado project and keep it under version control
- write synthesizable Verilog/SystemVerilog or VHDL for small modules
- build self-checking testbenches
- read synthesis, utilization, timing, and power reports
- constrain clocks and basic I/O correctly
- debug a design in simulation and on hardware
- package a reusable IP block with documentation
- explain what changed between RTL simulation and implemented hardware

---

## Unit Map

| Unit | Focus | Artifact |
|------|-------|----------|
| 1 | Vivado project flow | reproducible project skeleton |
| 2 | RTL and simulation | self-checking testbench and waveform capture |
| 3 | Synthesis and implementation | utilization and timing report |
| 4 | Constraints and timing | XDC file and timing-closure note |
| 5 | IP Integrator and AXI basics | block design with address map |
| 6 | On-chip debug | ILA/VIO capture and debug write-up |
| 7 | Reusable IP packaging | packaged IP core with README |

---

## Unit 1: Vivado Project Flow

### Learn

- project mode versus non-project mode
- source hierarchy and constraints organization
- generated files versus source files
- reproducible builds
- board files and part selection
- Tcl automation for builds

### Build It

Create a minimal repository:

```text
rtl/
tb/
constraints/
scripts/
docs/
reports/
```

Add a Tcl script that can create the project, add sources, run synthesis, and export reports.

### Measure It

- Can the project be rebuilt from a clean clone?
- Are generated files excluded from version control?
- Are reports written to a predictable path?

### Ship It

A clean Vivado project skeleton with `make` or script-driven rebuild instructions.

---

## Unit 2: RTL And Simulation

### Learn

- combinational versus sequential logic
- resets, clock enables, and register-transfer structure
- blocking versus non-blocking assignments
- module interfaces and parameterization
- testbench structure
- assertions and self-checking tests

### Build It

Implement three modules:

1. counter with enable and synchronous reset
2. UART-like byte transmitter or SPI-style shifter
3. small streaming datapath with valid/ready handshake

For each module, write a self-checking testbench.

### Measure It

- number of directed tests
- assertion failures caught intentionally
- waveform capture that explains one bug

### Ship It

RTL, testbenches, simulation commands, and one short debug note.

---

## Unit 3: Synthesis And Implementation

### Learn

- synthesis versus implementation
- LUTs, flip-flops, BRAM, DSP slices, and routing
- inferred versus instantiated hardware
- resource sharing and retiming
- warning triage
- bitstream generation

### Build It

Synthesize and implement the streaming datapath from Unit 2.

Generate:

- utilization report
- timing summary
- power estimate
- schematic or netlist screenshot if useful

### Measure It

- LUT/FF/BRAM/DSP usage
- critical path
- worst negative slack
- achieved clock frequency

### Ship It

An implementation report that explains what hardware the RTL became.

---

## Unit 4: Constraints And Timing

### Learn

- clock constraints
- input and output delays
- generated clocks
- false paths and multicycle paths
- setup, hold, slack, and critical path interpretation
- when timing constraints hide bugs instead of fixing them

### Build It

Add constraints for:

- primary clock
- reset path policy
- basic I/O timing
- one intentionally over-aggressive clock target

Then close timing by changing the design, not only the constraints.

### Measure It

- before/after worst negative slack
- critical path before/after optimization
- resource cost of the fix

### Ship It

An XDC file plus a timing-closure note explaining the bottleneck and the actual hardware fix.

---

## Unit 5: IP Integrator And AXI Basics

### Learn

- IP catalog
- block design structure
- AXI4-Lite versus AXI4-Stream versus AXI memory-mapped interfaces
- address maps
- reset and clocking blocks
- packaging custom RTL for block design use

### Build It

Create a block design with:

- clock/reset block
- AXI interconnect
- one custom AXI-Lite register block or streaming peripheral
- one simple vendor IP block

### Measure It

- address map correctness
- register read/write test
- timing and utilization after integration

### Ship It

Block design diagram, address map, and software or testbench proof that the custom block responds correctly.

---

## Unit 6: On-Chip Debug

### Learn

- simulation debug versus hardware debug
- Integrated Logic Analyzer (ILA)
- Virtual I/O (VIO)
- trigger conditions
- debug cores and timing/resource cost
- how to avoid "debugging by hoping"

### Build It

Insert ILA probes into the streaming datapath or AXI block.

Capture:

- reset release
- first transaction
- one error or corner case
- one throughput measurement if applicable

### Measure It

- debug core resource overhead
- captured cycle timing
- difference between expected and observed hardware behavior

### Ship It

ILA screenshots or exported captures plus a debug write-up.

---

## Unit 7: Reusable IP Packaging

### Learn

- parameterized RTL
- interface documentation
- IP packager
- versioning and metadata
- example designs
- verification collateral

### Build It

Package one block from this module as reusable IP.

Include:

- parameters
- clock/reset assumptions
- interface timing
- testbench
- example instantiation
- synthesis/timing reports

### Measure It

- integration time in a new project
- warnings generated during packaging
- resource and timing numbers on the target board

### Ship It

A reusable IP folder that another engineer can instantiate without reading the whole source tree.

---

## Capstone

Build a small board-validated FPGA subsystem:

- custom RTL datapath
- simulation testbench
- Vivado project script
- XDC constraints
- implementation reports
- ILA debug capture
- board demo
- README with rebuild and validation steps

Good capstone examples:

- AXI-Lite controlled PWM or GPIO peripheral
- SPI sensor reader with FIFO
- streaming image filter
- UART packet parser
- fixed-point matrix-vector block

The capstone is complete when someone else can rebuild the bitstream, understand the timing report, and reproduce the board-level behavior.

---

## Exit Criteria

You are ready for the next FPGA modules when you can:

- build a Vivado project from source
- write and simulate small RTL blocks
- interpret timing and utilization reports
- debug both simulation and hardware behavior
- constrain a design without hiding real timing problems
- package a small reusable IP block
- explain the engineering evidence behind a working bitstream
