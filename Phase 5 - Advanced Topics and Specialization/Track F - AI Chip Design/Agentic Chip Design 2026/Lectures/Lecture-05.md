# Lecture 05 — Test the checker with deliberate RTL faults

[Course guide](../Guide.md) · [Previous](Lecture-04.md) · [Next: Lecture 06](Lecture-06.md)

Before trusting a checker, make it reject errors you understand. The course's [RTL protocol lab](../../../../Phase%204%20-%20Track%20A%20-%20Xilinx%20FPGA/1.%20Xilinx%20FPGA%20Development/labs/README.md) includes three deterministic fault injections. They modify a temporary copy of the design, run the actual simulator, and require the expected assertion category. They do not edit the checked-in reference or accept a failed run as a successful design.

## Run three negative cases

From the repository root:

```bash
python3 'Phase 4 - Track A - Xilinx FPGA/1. Xilinx FPGA Development/labs/run_stream_lab.py' --inject-fault overwrite-stall
python3 'Phase 4 - Track A - Xilinx FPGA/1. Xilinx FPGA Development/labs/run_stream_lab.py' --inject-fault wrap-overflow
python3 'Phase 4 - Track A - Xilinx FPGA/1. Xilinx FPGA Development/labs/run_stream_lab.py' --inject-fault retain-on-reset
```

Each command should report `fault_detected` with the associated assertion. The wrapper's successful exit means the negative-test experiment caught the expected bug. The candidate simulation itself exits nonzero. This distinction prevents a reporting mistake: a successful fault-detection test is not a passing RTL candidate.

| Fault | Violated requirement | Expected category |
|---|---|---|
| Overwrite while stalled | Undelivered data must remain stable | `STALL` |
| Wrap 255+1 to zero | Arithmetic saturates at 255 | `DATA` |
| Retain valid state through reset | Reset flushes buffered validity | `VALID` |

These mutations were executed with Icarus 12.0 during the course review. The lab also runs multiple deterministic pseudorandom backpressure seeds; see its validation record for the actual runs.

## Read a counterexample as a timeline

For overwrite-stall, reconstruct the first failing sequence: which value was accepted, whether the output was valid, whether the consumer was ready, and which next input changed the stored value. The root cause is an incorrect state-update condition, not merely “the scoreboard expected another byte.”

A useful debug note contains the contract clause, first divergence cycle, relevant state and handshakes, proposed cause, and a testable repair. Avoid changing several independent behaviors before rerunning; otherwise you cannot tell which modification fixed the error or introduced another one.

## Strengthen the test deliberately

Add a directed reset while full, repeated saturating inputs, prolonged downstream stall, or simultaneous consume/replace. For each added case, predict the reference queue and expected counts before running. If reset flushes an item, the conservation rule is `accepted = delivered + flushed + remaining`, using a consistent measurement interval.

Formal properties can complement these traces. State assumptions about the input producer, reset, and output consumer. A liveness property needs an environment progress assumption: if ready is allowed to stay low forever, eventual delivery is impossible. A safety property such as output stability during a stall does not require eventual readiness.

**Exercise:** the overwrite mutation passes when downstream ready is always high. Is the fault injection broken? **Answer:** no; that stimulus never exercises the faulty transition. Add the relevant stall rather than increasing random runtime without checking coverage.

Your artifact is a fault-to-property table and one annotated failing trace. Continue to [Lecture 06](Lecture-06.md) to use these results as structured feedback in a bounded repair loop.
