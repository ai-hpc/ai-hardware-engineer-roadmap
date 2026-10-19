# Embedded and edge AI engineering: follow the data

Practice explaining where bytes live, when observations become stale, and how a
device recovers from faults. These original scenarios use illustrative numbers;
they are not measured board performance or a specific employer's interview list.

**Prerequisites:** [embedded systems](../../Phase%202%20-%20Embedded%20Systems/Guide.md),
basic C/Python, and the [observation freshness lab](../../Phase%202%20-%20Embedded%20Systems/Labs/Observation-Freshness.md).
**Artifact:** a data-ownership diagram, a timing budget, and one fault-injection
record. Allow two 60–90 minute practice sessions.

Try each problem before reading the guidance. A useful answer connects a number
or mechanism to a measurement that could disprove your assumption.

## 1. How much traffic does an RGB camera create?

An uncompressed camera produces packed 1920 × 1080 RGB8 frames at 30 frames/s.
Estimate the payload rate and the extra memory traffic from two full-frame copies.

<details markdown="1"><summary>Answer guidance</summary>

One frame contains `1920 × 1080 × 3 = 6,220,800 bytes`. The payload is
`186,624,000 bytes/s`, or approximately 186.6 decimal MB/s. One copy reads and
writes that payload, so two copies add about `4 × 186.6 = 746.5 MB/s` of traffic.
This counts only the copies. Sensor writes, preprocessing, model reads, row
padding, cache behavior, and other services affect the full memory-system load.

Distinguish RGB8 from a raw Bayer or compressed stream before applying the formula.
Measure the actual buffer stride and format, then inspect copies and synchronization
in the pipeline.

**Follow-up:** a producer and consumer share DRAM. That does not prove they share
the same compatible buffer or can access it concurrently without synchronization.

</details>

## 2. The camera produces a frame every 10 ms; inference takes 20 ms

What happens to an unbounded FIFO queue? Would a queue that keeps only the newest
waiting frame improve the situation?

<details markdown="1"><summary>Answer guidance</summary>

The producer delivers 100 frames/s and the consumer can finish at most 50/s under
these assumptions. The backlog grows by roughly 50 frames each second, and an
increasingly old frame reaches inference. More queue capacity postpones overflow;
it does not remove the rate mismatch.

Keeping the newest waiting frame bounds waiting backlog by discarding superseded
frames. It cannot make the current nonpreemptive inference finish sooner and may
be inappropriate if every sample must be preserved. Separate a lossless recording
path from a freshness-oriented control path when the task needs both.

**Follow-up:** measure accepted-action age and dropped-frame count, not only
inference throughput. The [freshness lab](../../Phase%202%20-%20Embedded%20Systems/Labs/Observation-Freshness.md)
lets you reproduce the queue tradeoff without a board.

</details>

## 3. Can asynchronous execution turn 9 ms into 6 ms?

A workload has 2 ms input transfer, 6 ms compute, and 1 ms output transfer per
frame. Compare serial execution with ideal overlap across independent frames.

<details markdown="1"><summary>Answer guidance</summary>

Serial service takes 9 ms per frame, giving an ideal ceiling of about 111 frames/s.
If buffering and hardware permit transfers to overlap computation across frames,
the steady-state initiation interval can approach 6 ms, or about 167 frames/s.
An individual frame still depends on its own input transfer, compute, and output
transfer. Pipeline throughput does not automatically reduce its 9 ms latency.

This is an analytical ceiling. Confirm copy-engine capabilities, pinned-memory
requirements, dependencies, memory-bandwidth contention, and buffer ownership
with a trace. Startup/drain and queueing also affect observed results.

**Follow-up:** the next frame depends on the previous action's result. Explain
which independence assumption no longer holds before proposing more buffering.

</details>

## 4. INT8 inference gives different outputs after a port

The weights and activations are both signed INT8. What should you check before
blaming the compiler or retraining the model?

<details markdown="1"><summary>Answer guidance</summary>

Check tensor layout, channel order, normalization, quantization scale and
zero-point, per-channel axis, rounding, saturation, accumulator width, and output
requantization. Save a small input and intermediate reference tensors to locate
the first divergence.

For a 256-term dot product with raw signed INT8 operands, a conservative absolute
sum bound is `256 × 128 × 128 = 4,194,304`, which fits signed INT32. Accumulating
in INT8 does not. Zero-point subtraction or longer reductions require their own
bound; do not reuse this calculation unchanged.

**Follow-up:** absolute error is small but the predicted action changes sign near
zero. Evaluate the task-relevant consequence as well as a global error average.

</details>

## 5. An update boots, then loses the sensor driver

What would a useful update acceptance test and rollback record contain?

<details markdown="1"><summary>Answer guidance</summary>

Record the board/module identity, boot firmware, kernel, device tree, modules,
root filesystem, application version, and model/runtime pair. Test the actual
sensor path, timestamps, recovery, and application behavior after the update.
A successful boot or service process is insufficient evidence of working I/O.

Specify when an update is considered healthy and which component can choose a
known-good image if that condition is not met. Test interrupted update and failed
health-check cases on an appropriate test device using the platform's documented
recovery procedure. Preserve logs that distinguish boot failure from application
failure.

**Follow-up:** the new application changes its state format. Rolling back binaries
without a compatible data migration can leave the older application unable to
read its state.

</details>

## 6. Average latency is good; long runs fail deadlines

Propose a measurement plan that can distinguish thermal throttling, queue growth,
and a rare blocking operation.

<details markdown="1"><summary>Answer guidance</summary>

Warm up the actual workload and record per-request stage times, queue lengths,
observation age, temperature, clock/power state, and errors for a declared duration.
Report percentiles and the number/fraction of deadline misses, preserving the raw
timeline. Correlate the first degraded interval with these signals.

Repeat one controlled intervention: for example, hold input rate below measured
service capacity to test queue growth, or inspect an execution trace for the
blocking call. Change one condition at a time. A short idle-board benchmark does
not establish sustained behavior in the deployed enclosure.

**Follow-up:** reducing clocks lowers instantaneous power but increases runtime.
Energy is the integral of power over time; measure both before concluding that
the change saves energy per completed task.

</details>

## Final artifact and exit criteria

Draw camera → driver → buffer → preprocessing → model → command validation →
controller. Annotate each arrow with ownership, format, timestamps, and completion
notification. Attach one normal trace and one fault trace. State what remains
untested if you only ran a numerical model or recorded-data exercise.

You are ready to move on when you can compute the memory and timing budget,
identify an invalid ownership or freshness assumption, and design a small test
that reveals it. Use the shared [learning rubric](../../Learning-Guide.md) to review
correctness, explanation, measurement, reproduction, and transfer.

[Interview overview](../README.md) · [Jetson platform](../../Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/1.%20Nvidia%20Jetson%20Platform/Guide.md)
