# Lecture 07 — Connect inference to a robot's deadline and energy budget

[Previous: speculative decoding](Lecture-06.md) · [Course index](README.md)

On a robot, an inference output is useful only when it has the right meaning and
arrives in time to affect an appropriate action. Tokens per second, accelerator
peak operations and model accuracy measure different things. The capstone joins
these measurements through one application contract.

**Prerequisites:** the earlier lessons' memory, quality and timing boundaries.
Bring a working CPU model or operator and its reference checks. A GPU or robot
is an optional next stage, not a prerequisite for reasoning clearly about the
pipeline. Allow two or three sessions for a small capstone.

## Start from an observation-to-action timeline

Record sensor capture, transport completion, inference start/end, command
acceptance and actual application. A queue can make an apparently fast model
act on old data. The model's execution duration is only one part of observation
age at actuation. A command accepted exactly at its expiry may never be applied
for a complete control interval.

Use the [timed robot-loop lab](../../../Phase%203%20-%20Artificial%20Intelligence/0.%20AI%20and%20Robot%20Learning%20Foundations/Lab-02-Timed-Robot-Loop.md)
as the CPU capstone baseline. It holds a genuinely trained small policy fixed
while varying simulated queues and inference delays. Held-out prediction error
stays fixed, but closed-loop outcomes change. The simulator's delays are chosen
inputs, not measured CPU or Jetson latency.

State whether the action is position, velocity, acceleration or torque, which
frame it uses, and how long it remains valid. A faster model cannot repair an
incorrect unit conversion or an expired observation. A fallback also needs its
own sensing assumptions; a velocity-damping fallback requires a usable velocity
estimate when the main observation stream fails.

## Read hardware specifications precisely

The [Jetson Thor specification](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/)
lists the T5000's 128 GB memory, 273 GB/s peak bandwidth, 40–130 W power modes
and 2,070 **FP4 sparse TFLOPS**. Preserve the precision and sparsity qualifier.
These figures are not robot control frequency, dense FP16 throughput, measured
memory bandwidth or whole-system power. Other Thor modules have different limits.

For an illustrative dense 7B model, an idealized four-bit weight payload is
3.5 GB before scales and unchanged tensors. Reading it once per batch-one token
at 273 GB/s would take at least 12.82 ms, an optimistic ceiling of 78 tokens/s
for that traffic assumption. Real execution adds cache reads, computation and
other costs; usable bandwidth can be lower. Multiplying this bound by an
acceptance length while ignoring draft/verification cost is invalid.

Quantization is one option. A smaller model, different task formulation,
batching strategy or operator implementation may be preferable. Small control
policies can run well in higher precision. No universal rule requires every
robot model to use four-bit weights.

## Energy, thermals and software support

Measure energy over the same task interval used for outcomes. Include retries,
failed attempts and idle time when they belong to the operating cycle. A short
peak-throughput run does not establish sustained performance after the board
heats up. Record ambient conditions, power mode, cooling, sensor activity and
concurrent CPU/GPU workloads.

Select a software stack that supports the board, model and operators. The
[Jetson compatibility guide](../../../Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/Compatibility.md)
separates supported combinations. TensorRT, TensorRT-LLM and TensorRT Edge-LLM
are distinct components with release-specific coverage. A datacenter recipe
cannot be assumed to work on a Jetson merely because both use CUDA.

## The capstone: one change with a defensible result

Choose one of two routes. For a CPU-only route, use the trained-policy/timed-loop
lab and change one queue or expiry behavior while keeping weights and initial
conditions fixed. For an available device, deploy a supported small model,
retain its CPU reference and measure a single optimization such as supported
quantization or graph capture.

Keep the baseline and candidate on the same test episodes or requests. Define
quality, numerical tolerance, deadline and resource limits before measuring.
Record individual outcomes, not only an aggregate average. A model change needs
a new quality evaluation; replacing the architecture is not a drop-in kernel
optimization on the same model.

| Evidence | CPU route | Optional deployed route |
|---|---|---|
| Correctness | Reference outputs, units and state checks | Same checks plus actual backend outputs |
| Timing | Explicit simulated delay/queue model | Capture-to-actuation timestamps and distributions |
| Task quality | Paired synthetic rollouts with stated limits | Held-out task outcomes under a declared protocol |
| Resources | Derived byte and compute counts | Measured allocation, sustained power and energy |
| Failure | Stale or rejected result with explained recovery | Bounded fault test appropriate to the actual system |

**Worked interpretation:** a candidate doubles operator throughput but reduces
completed tasks from 18/20 to 12/20. It fails a predeclared “no quality reduction”
gate on that fixed suite. The small sample does not prove a population-level
failure rate or equivalence margin; report paired outcomes and uncertainty before
making broader claims. An unchanged or slower candidate is still a useful
experiment if it explains the limiting stage.

**Exit check:** another learner can reproduce your baseline, identify the one
change, and explain both a success and a failure. Your final claim names its
execution level: CPU fixture, simulator, board measurement or physical task.
Reviewed October 18, 2026; no Thor performance or physical-robot outcome is
claimed by this lesson.
