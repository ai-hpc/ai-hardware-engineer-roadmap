# Lecture 02 — From specification to implementation evidence

[Course guide](../Guide.md) · [Previous](Lecture-01.md) · [Next: Lecture 03](Lecture-03.md)

The first design task is to write a contract precise enough that a wrong implementation can be rejected. Use the [one-entry streaming incrementer](../../../../Phase%204%20-%20Track%20A%20-%20Xilinx%20FPGA/1.%20Xilinx%20FPGA%20Development/labs/README.md) as the running example. It accepts an unsigned byte, adds one with saturation at 255, and emits the result through a valid/ready interface. It has one storage slot. This small module exposes the same ordering and flow-control problems that appear at accelerator command and DMA boundaries.

## Separate the stages

| Stage | What it establishes | What it does not establish |
|---|---|---|
| Parse/elaborate | The source and design hierarchy are accepted | Correct output values or temporal behavior |
| Functional simulation | Behavior matches the checker on exercised traces | Exhaustive behavior, synthesis, or timing closure |
| Formal verification | Stated properties hold under stated assumptions | Properties omitted from the specification |
| Synthesis | RTL maps to a selected logic representation | Routed timing, actual board behavior, measured power |
| Physical implementation | Placement/routing under specified constraints | Correctness outside the constraints or silicon measurements |

A result should name its stage. “Passed simulation with Icarus 12” is reviewable. “Chip verified” does not say what evidence exists. The provided lab has been simulated; this course review did not perform synthesis, place-and-route, or silicon validation.

## Write the transfer rule before the datapath

At each active clock edge, an input is accepted only when `s_valid && s_ready`. An output is delivered only when `m_valid && m_ready`. If valid is high and ready is low, the producer must retain its value until a transfer or the specified reset behavior. The example masks handshakes during reset and clears its valid state on a reset clock edge; read the lab's precise reset convention.

The one-slot buffer can accept a replacement in the same cycle its old output is consumed. A useful readiness equation is `empty || downstream_ready`, subject to the reset rule. If full and downstream is stalled, accepting another input would overwrite an undelivered value.

**Worked trace:** accept 254 into an empty buffer; its stored result is 255. Stall the output for three cycles. Every stalled cycle must expose the same 255 with valid asserted. Then consume it while accepting 255; the replacement is also 255 because arithmetic saturates. On the next ready cycle, deliver that replacement. Two accepted items must produce two deliveries unless reset explicitly flushes one.

## Execute the baseline

From the repository root, with Icarus Verilog installed:

```bash
python3 'Phase 4 - Track A - Xilinx FPGA/1. Xilinx FPGA Development/labs/run_stream_lab.py' --seed 1 --output /tmp/rtl-baseline.json
```

The report includes acceptance, delivery, flush, stall, replacement, and saturation counters. Read them alongside the testbench; a counter is meaningful only if its event definition matches the contract. Keep the seed and tool version with the report.

**Exercise:** an implementation emits the right values on a continuously ready interface, but changes output data while stalled. Which stage and test catch it? **Answer:** the directed stall test in functional simulation can reject it before synthesis. A no-backpressure test cannot establish this requirement.

Your deliverable is a one-page contract with ports, clock/reset behavior, numerical overflow rule, transfer rule, and a short expected trace. Then continue to [Lecture 03](Lecture-03.md) and give that contract to a generator.
