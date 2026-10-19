# 3. Advanced FPGA Design

<div class="course-identity advanced-fpga" markdown="1">
<div class="course-identity__icon">CDC</div>
<div markdown="1">
<p class="course-identity__eyebrow">Track A3 · Advanced FPGA Design</p>
<p class="course-identity__title">Make FPGA systems timing-clean, clock-safe, power-aware, and hardware-debuggable.</p>
<p class="course-identity__meta">Artifact: timing-closed validation package · Measure: slack, CDC, power, robustness</p>
</div>
</div>


> Move beyond working RTL into timing-closed, clock-safe, power-aware FPGA systems that survive real board constraints.

**Layer mapping:** L5-L6. This module focuses on timing closure, clock-domain crossings, high-speed interfaces, floorplanning, power analysis, partial reconfiguration, and hardware robustness.

**Role targets:** Senior FPGA Engineer · FPGA Timing Closure Engineer · Hardware Acceleration Engineer · FPGA Systems Architect

**Prerequisites:** [Xilinx FPGA Development](../1.%20Xilinx%20FPGA%20Development/Guide.md), comfort reading timing reports, and at least one board-validated FPGA project.

**What comes after:** [High-Level Synthesis](../4.%20High-Level%20Synthesis%20%28HLS%29/Guide.md), [Runtime and Driver Development](../5.%20Runtime%20and%20Driver%20Development/Guide.md), and [AI Chip Design](../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20F%20-%20AI%20Chip%20Design/Guide.md).

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

### Work one path before changing a constraint

For a same-clock register path with zero skew, use the simplified setup inequality `t_cq,max + t_logic,max + t_setup + uncertainty ≤ period`. Assume 0.3 ns clock-to-Q, 5.9 ns logic/routing, 0.2 ns setup, and 0.1 ns uncertainty. At a 6 ns period the required time is 6.5 ns, so setup slack is **−0.5 ns**. These are invented teaching values, not a device report.

After a valid pipeline change, suppose each stage's worst logic/routing delay is at most 3.2 ns. The simplified stage requirement becomes 3.8 ns, which fits 6 ns, but latency and valid/data alignment changed and need regression tests. Real timing includes launch/capture clock paths, skew, corners, and exceptions.

Hold uses a different constraint. With zero skew, `t_cq,min + t_logic,min ≥ t_hold + uncertainty`. If the first two terms are 0.10 and 0.05 ns while the right side is 0.20 ns, hold slack is **−0.05 ns**. Increasing the clock period does not repair this same-edge hold failure.

A false-path exception means the path does not require the stated functional timing check; it is not a way to declare an inconvenient path correct. A multicycle exception must follow the actual enable/capture protocol and corresponding setup/hold relationships. Review unconstrained paths as well as WNS/TNS. “No failing reported paths” is weak evidence when essential clocks or I/O delays were never constrained.

See [AMD timing-summary report details](https://docs.amd.com/r/en-US/ug906-vivado-design-analysis/Details-of-the-Timing-Summary-Report) for the full report and constraint checks.

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

### Start with the crossing's data contract

The [streaming lab](../1.%20Xilinx%20FPGA%20Development/labs/README.md) assumes one clock. Connecting its ready/valid/data wires directly between unrelated clocks breaks that assumption. Synchronizing every data bit independently also does not preserve a multi-bit word: different bits can reach the destination on different edges.

Use a synchronizer for a suitable single-bit level, a specified pulse/handshake mechanism for an event, and a handshake bridge or asynchronous FIFO for a coherent data item. Choose the reset/recovery behavior on both sides as part of that interface. AMD's [XPM CDC library](https://docs.amd.com/r/en-US/ug1704-spartan-ultrascaleplus-libraries/XPM_CDC_HANDSHAKE) provides a concrete full-handshake crossing to study; select the library version for your target device.

**Worked question:** a camera word contains `{frame_id, pixel}`. Is it enough to synchronize both fields bit by bit? **Answer:** no; the receiver must observe the fields from the same accepted item. Transfer the complete word with a protocol that preserves that relationship. Check occupancy, order, reset flushing, and progress under different clock ratios. Digital simulation can reveal protocol errors but does not model analog metastability probability or replace a CDC analysis.

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

### Connect power to completed work

If a measured system consumes 12 W while completing 600 valid results/s, its energy per result is `12/600 = 0.020 J`. A second design at 10 W and 400 results/s consumes 0.025 J/result despite using less instantaneous power. Include idle/static power and rejected/dropped work in the declared boundary. These values are hypothetical.

Clock enables can suppress register updates but do not remove all clock-tree power. Use device-supported clock-control resources when gating is justified; do not AND a clock with an arbitrary logic signal. A power estimator's switching assumptions must match the workload before comparing it with a board measurement.

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

### Separate safety from progress

“No output transfer occurs without a prior accepted input” is a safety property. “Every accepted input eventually produces output” is a progress property and needs assumptions: if the sink holds ready low forever, a finite buffered producer cannot force completion. State reset behavior, environmental fairness or bounded stalls, and the induction/bounded-search scope. A passing simulation assertion checks the exercised trace; a bounded formal run checks its stated bound, not automatically every future cycle.

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

Before reconfiguring, stop admission, drain or explicitly abort outstanding transactions, isolate the changing region's outputs, and establish reset/restart state. A generation ID can reject a stale completion but cannot stop an old DMA request from writing memory. Device support, partition interfaces, tool/IP releases, licensing, and static-design compatibility belong in the feasibility artifact. Follow [AMD Dynamic Function eXchange documentation](https://docs.amd.com/r/en-US/ug909-vivado-partial-reconfiguration/Introduction).

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

**Study method:** use the [shared lesson cycle and completion rubric](../../Learning-Guide.md) with this course's existing labs. Predict first, test one variation, retain a failure case, and revisit the self-check after two days and one week.


## Integration lab: Does a faster datapath still work across clock boundaries?

Take the previous streaming block across two clocks using an asynchronous FIFO. Randomize the phase, downstream stalls, and reset order in simulation; inspect the CDC report and timing constraints.

**Acceptance evidence:** Explain each remaining CDC finding and show no loss, duplication, or reordering for the checked traces. A two-flop synchronizer on a multi-bit payload is not an acceptable bus-transfer design.

**Continue:** take this artifact to [4. High-Level Synthesis (HLS)](../4.%20High-Level%20Synthesis%20%28HLS%29/Guide.md). Explain which assumption from this course the next engineer must preserve.
