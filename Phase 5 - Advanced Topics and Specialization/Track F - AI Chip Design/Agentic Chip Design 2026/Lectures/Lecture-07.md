# Lecture 07 — Optimize area and timing without losing function

[Course guide](../Guide.md) · [Previous](Lecture-06.md) · [Next: Lecture 08](Lecture-08.md)

Physical design metrics are meaningful only after defining the technology and constraints. An RTL edit that reduces source lines does not necessarily reduce cells; a shorter combinational expression does not prove a faster routed path. First preserve the functional contract with the [independent protocol checker](../../../../Phase%204%20-%20Track%20A%20-%20Xilinx%20FPGA/1.%20Xilinx%20FPGA%20Development/labs/README.md).

## Name the metric and stage

Synthesis can report mapped cells and logic structure for a selected library. Static timing analysis uses clocks, input/output delays, uncertainty, timing corners, and interconnect estimates. Placement and routing add physical effects. Power estimation additionally needs activity assumptions. A simulator wall time or a gate count is not a measurement of silicon power or clock frequency.

Use the [Yosys synthesis tutorial](https://yosyshq.readthedocs.io/projects/yosys/en/latest/getting_started/example_synth.html) to understand synthesis stages and the [OpenROAD flow documentation](https://openroad.readthedocs.io/en/latest/main/README.html) for physical implementation. Follow the selected platform's complete setup. These synthesis/physical flows were not executed in this course review; do not fill the report with fabricated PPA results.

## Make comparisons comparable

Hold technology/library, timing corners, constraints, tool versions, effort settings, and implementation seed constant when comparing two candidate designs. Report any deliberate change. A design meeting a looser clock constraint is not directly comparable to one optimized for a tighter clock.

**Worked tradeoff:** suppose an experiment under identical conditions reports area 100 units and delay 2.0 ns for A, area 120 and delay 1.6 ns for B, and area 110 and delay 2.2 ns for C. B trades more area for less delay. C is dominated by A because it is both larger and slower. If the required delay is 1.8 ns, A is infeasible and B is the only feasible option among these three. These numbers illustrate decision logic; they are not course measurements.

An area-delay product can rank some designs, but it must not hide hard requirements. A tiny design missing a control deadline is not rescued by a favorable product. Likewise power must meet a thermal budget rather than simply becoming another weighted score.

## Preserve latency and throughput semantics

Inserting a pipeline register may improve a combinational timing path while adding a cycle of response latency. For a valid/ready interface, update the implementation and checker only if the public contract permits that latency change. Maintain output ordering and behavior under backpressure. A fixed-cycle interface may prohibit the change entirely.

The one-entry incrementer is intentionally small. Optimizing it is a lesson in evidence discipline, not a route to an impressive TOPS claim. A more meaningful next experiment is a wider datapath or a queue with an explicitly specified depth and simultaneous push/pop behavior.

**Exercise:** after retiming, all values are correct but an output arrives one cycle later. Is the candidate equivalent? **Answer:** it depends on the interface contract. A protocol permitting variable latency may accept it; a cycle-accurate contract may reject it. Decide before scoring.

Deliver a comparison template with functional status, constraints, stage, metrics, and feasibility. Leave unmeasured PPA fields marked unmeasured. Continue to [Lecture 08](Lecture-08.md) to budget the design loop itself.
