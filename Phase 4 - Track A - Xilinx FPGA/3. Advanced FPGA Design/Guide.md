# 3. Advanced FPGA Design

> Move beyond working RTL into timing-closed, clock-safe, power-aware FPGA systems that survive real board constraints.

**Layer mapping:** L5-L6. This module focuses on timing closure, clock-domain crossings, high-speed interfaces, floorplanning, power analysis, partial reconfiguration, and hardware robustness.

**Role targets:** Senior FPGA Engineer · FPGA Timing Closure Engineer · Hardware Acceleration Engineer · FPGA Systems Architect

**Prerequisites:** [Xilinx FPGA Development](../1.%20Xilinx%20FPGA%20Development/Guide.md), comfort reading timing reports, and at least one board-validated FPGA project.

**What comes after:** [High-Level Synthesis](../4.%20High-Level%20Synthesis%20%28HLS%29/Guide.md), [Runtime and Driver Development](../5.%20Runtime%20and%20Driver%20Development/Guide.md), and [AI Chip Design](../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/6.%20AI%20Chip%20Design/Guide.md).

---

## Why This Module Exists

Many FPGA designs work in simulation and fail in hardware because timing, clocks, resets, interfaces, power, or physical layout were treated as afterthoughts.

Advanced FPGA work is about making this statement defensible:

```text
The design is functionally correct, timing-clean, physically realistic, debuggable, and measured on hardware.
```

---

## Course Outcomes

By the end, you should be able to:

- identify and fix critical timing paths
- design safe clock-domain crossings
- use floorplanning and placement constraints when they are justified
- reason about high-speed I/O and board-level signal integrity constraints
- estimate and measure FPGA power
- use formal or assertion-based checks for high-risk logic
- explain the tradeoffs of partial reconfiguration
- produce a professional timing and hardware validation report

---

## Unit Map

| Unit | Focus | Artifact |
|------|-------|----------|
| 1 | Timing closure | before/after timing report |
| 2 | Clock/reset domains | CDC-safe design and verification note |
| 3 | Floorplanning | constrained implementation comparison |
| 4 | High-speed interfaces | interface constraint and SI checklist |
| 5 | Power optimization | power estimate and reduction report |
| 6 | Formal and assertions | property checks or assertion suite |
| 7 | Partial reconfiguration | design feasibility note or small demo |

---

## Unit 1: Timing Closure

### Learn

- setup and hold timing
- critical path analysis
- fanout, logic depth, routing delay, and congestion
- pipelining, retiming, replication, and register balancing
- false paths and multicycle paths
- timing closure methodology

### Build It

Take a design that fails timing under an aggressive clock target.

Apply real fixes:

- pipeline a datapath
- reduce fanout
- split combinational logic
- register interfaces
- adjust resource mapping

### Measure It

- worst negative slack before/after
- total negative slack before/after
- critical path logic versus route delay
- resource cost of the fix

### Ship It

A timing-closure report with the failed path, the fix, and evidence that the design still simulates correctly.

---

## Unit 2: Clock And Reset Domains

### Learn

- metastability
- two-flop synchronizers
- pulse synchronization
- asynchronous FIFOs
- valid/ready handshakes across domains
- reset synchronization
- CDC tool reports and waiver discipline

### Build It

Implement:

- a single-bit CDC synchronizer
- a pulse crossing
- an asynchronous FIFO or handshake bridge

Add tests that intentionally stress boundary cases.

### Measure It

- CDC report results
- simulation coverage for crossing behavior
- latency across the crossing
- resource use

### Ship It

A CDC-safe module library with a short "when to use which crossing" guide.

---

## Unit 3: Floorplanning And Physical Constraints

### Learn

- Pblocks
- placement constraints
- routing congestion
- clock region boundaries
- physical synthesis
- when floorplanning helps and when it makes the design brittle

### Build It

Run the same design with and without a small floorplan.

Use floorplanning to solve a real issue:

- routing congestion
- long critical path
- interface locality
- debug core placement

### Measure It

- timing difference
- routing congestion
- implementation runtime
- resource placement

### Ship It

A comparison report that explains whether the floorplan was worth keeping.

---

## Unit 4: High-Speed Interfaces

### Learn

- source-synchronous interfaces
- DDR timing considerations
- SerDes concepts
- PCIe and Ethernet at a system level
- differential signaling
- impedance, length matching, and termination
- board constraints that affect FPGA logic

### Build It

Pick one interface path:

- DDR memory interface review
- Ethernet or PCIe example design bring-up
- camera/video stream interface
- source-synchronous test design

### Measure It

- link bring-up status
- eye or margin data if available
- throughput
- error counters
- timing constraints

### Ship It

An interface bring-up checklist with constraints, test commands, and failure symptoms.

---

## Unit 5: Power Optimization

### Learn

- static versus dynamic power
- clock gating and clock enables
- data toggle rate
- BRAM/DSP power
- voltage and frequency tradeoffs
- thermal constraints
- Xilinx Power Estimator and implemented-design power reports

### Build It

Take a working design and reduce power by:

- reducing unnecessary toggling
- adding clock enables
- lowering clock frequency where acceptable
- changing buffering or data width

### Measure It

- estimated power before/after
- throughput before/after
- timing before/after
- thermal reading if available

### Ship It

A power report that explains what changed and what performance cost it introduced.

---

## Unit 6: Formal Checks And Assertions

### Learn

- assertion-based verification
- simple safety and liveness properties
- bounded model checking
- equivalence checks at a conceptual level
- where formal helps more than simulation

### Build It

Add assertions to one high-risk block:

- FIFO
- handshake bridge
- arbiter
- packet parser
- control FSM

Prove or test properties such as:

- no overflow
- no underflow
- request eventually acknowledged
- illegal state never reached

### Measure It

- properties checked
- counterexamples found
- simulation bugs caught by assertions

### Ship It

Assertion file or formal test harness with a short verification note.

---

## Unit 7: Partial Reconfiguration

### Learn

- static region versus reconfigurable partition
- dynamic function exchange
- partial bitstreams
- interface stability
- timing closure with reconfigurable modules
- security and update risks

### Build It

For most learners, a feasibility study is enough. Build a small demo only if your board and tool license make it practical.

Demo idea:

- static shell with AXI interface
- two reconfigurable modules implementing different transforms
- runtime switch and validation test

### Measure It

- partial bitstream size
- reconfiguration time
- timing impact
- interface constraints

### Ship It

Either a small partial-reconfiguration demo or a design note explaining why the technique does or does not fit your target product.

---

## Capstone

Take one earlier FPGA design and make it production-grade:

- timing-clean at target frequency
- CDC reviewed
- reset strategy documented
- power estimated
- debug strategy defined
- hardware validation captured
- constraints reviewed

The capstone is complete when the report explains not only that the design works, but why it should keep working under realistic constraints.

---

## Exit Criteria

You are ready to move on when you can:

- close timing with design changes, not only constraints
- recognize unsafe clock/reset crossings
- explain the physical cause of at least one timing problem
- estimate and reduce power without breaking throughput requirements
- add assertions to risky control logic
- produce a validation package that looks credible to another FPGA engineer
