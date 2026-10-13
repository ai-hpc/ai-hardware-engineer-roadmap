# Lecture 22 — Asynchronous Control, ROS 2, and Real-Time Action Chunking

[Previous](Lecture-21.md) · [Course overview](README.md) · [Next](Lecture-23.md)

**Prerequisites:** Lectures 10, 20, and 21; basic concurrency and timestamp arithmetic.

**Effort:** 90 min teaching, 30 min CPU setup, 2–3 h scheduling lab; ROS integration adds 3–5 h.

**Evidence cutoff:** October 13, 2026. Numerical examples below are illustrative unless explicitly identified as executed lab results.

## One robot, several clocks

The camera exposes frames, a GPU predicts chunks, a controller consumes targets, and actuators track commands. These operations have different periods and may execute concurrently. Model inference is only one part of this system. A policy can be numerically accurate on its input frame and operationally wrong because the world changed before the command arrived.

**Outcome:** design a scheduler that handles delayed results, reset epochs, action expiry, and buffer underruns with explicit behavior.

The [ROS 2 clock design](https://design.ros2.org/articles/clock_and_time.html) distinguishes system, steady, and ROS time, including simulation-time jumps. Use a monotonic clock for local elapsed-duration measurements. Use simulation/ROS timestamps to associate sensor data with simulated events, and explicitly handle reset or backward jumps. Subtracting timestamps from unrelated clock domains produces a number, but not a valid age. Distributed machines require synchronization or an estimated mapping with an uncertainty budget.

## Derive an action schedule

Let observation time be \(t_o\), inference completion \(t_c\), and action period \(\Delta\). A chunk predicts commands for nominal times \(t_o+i\Delta\), where \(i=0,\ldots,H-1\). If execution begins at \(t_c\), the commands scheduled before \(t_c\) have expired under this convention. The first non-expired index is

\[
i_{\mathrm{first}} = \left\lceil\frac{t_c-t_o}{\Delta}\right\rceil.
\]

This is a scheduling model, not a universal definition of a model's action horizon. Some training datasets label their first action at the next control tick or relative to inference completion. Establish that alignment before implementing the equation.

For \(H=8\), \(\Delta=20\) ms, and a 70 ms inference delay, the first future command is index 4 at 80 ms; indices 0–3 are already past. Only four commands remain, ending at 140 ms. The interval from 70 to 80 ms still needs a specified controller behavior. “Drop stale actions” is incomplete unless the current command source covers the gap.

The CPU helper `usable_chunk_indices` implements this simple future-start rule and a maximum observation-age gate at each intended action start; delayed dispatch requires rechecking age at execution. It does not implement interpolation, trajectory continuity, a controller, or the research algorithm below. Its explicit limitation makes it useful for understanding the timing arithmetic.

## Why naive asynchronous replacement can fail

Suppose a previous chunk guides a gripper toward an object while a new inference request runs. The robot executes several old actions before the new chunk becomes available. The new chunk was generated from an earlier observation and may assume a different executed prefix. Replacing all commands can create a sudden velocity or direction change.

The paper [Real-Time Execution of Action Chunking Flow Policies](https://arxiv.org/abs/2506.07339) develops asynchronous execution using conditioning on portions of the already committed trajectory. Read its assumptions about delay and chunk overlap before adapting it to a new action representation. “Asynchronous” alone is not the method; maintaining consistency between committed actions and newly generated continuations is the key design problem.

A practical scheduler needs an explicit commit boundary. Commands within the boundary are already issued and cannot be retroactively changed. Future commands may be replaced if they satisfy epoch, timestamp, continuity, and validity checks. The commit horizon trades reaction speed against stability and compute latency. Longer commitment can hide inference jitter but delays response to disturbances.

## Define the message and state machine

Use a versioned message contract containing request ID, reset epoch, observation timestamp, clock domain, intended action times, action schema ID, model revision, and finite physical commands. Never infer these from arrival order. Two concurrent requests can complete in the opposite order.

**Exercise state machine — pseudocode:**

```text
RESET -> clear history and action buffer; increment epoch; await fresh sensors
READY -> submit bounded inference request; continue valid committed commands
RESULT -> reject wrong epoch/schema, stale observation, or invalid commands
ACCEPT -> merge only permitted future portion; record superseded request
UNDERRUN -> invoke task-specific fallback and record reason
FAULT -> controller-owned response; require an explicit recovery transition
```

Bound the queue. A FIFO of unprocessed camera frames can keep the GPU busy while delivering progressively older observations. A latest-observation policy may be appropriate for a reactive controller; a history-dependent model needs a correctly sampled history rather than arbitrary dropping. Preserve synchronized camera/proprioception associations when discarding frames.

A fallback depends on the robot and task. Holding a target, reducing speed, retreating, and stopping have different mechanical consequences. Specify it through the controller's operating contract and test it in simulation before any physical trial. A VLA service timeout should not leave undefined actuator behavior.

## ROS integration and observability

Separate transport from the control authority. A ROS publisher should validate requests and expose status; the controller should own command acceptance and enforce its limits independently of model generation. Choose QoS, queue depth, and executor scheduling using measured message rates and acceptable age. Reliable delivery does not guarantee timely delivery, and low queue depth does not guarantee synchronized sensors.

Log a trace ID from camera frame through preprocessing, inference, scheduler acceptance, and controller application. Capture request cancellation, buffer length, discarded-prefix count, command age, missed deadlines, and reset transitions. Report latency distributions under concurrent image transport and recording, since recording can alter the timing being measured.

## Guided lab

Run the [lab guide](Lab-Guide.md). Reproduce the 70 ms example, then vary delays through 0, 20, 70, 140, and 180 ms. Plot usable action count versus delay, with a separate curve for the maximum-age policy. At exactly one action boundary, state the chosen equality convention and test it.

Implement an offline event simulator with two requests completing out of order. Add a reset between submission and completion. The required negative cases are an old epoch, a future-dated observation, a nonfinite command, and a fully expired chunk. Keep rejected messages in the event log; silently discarding them makes diagnosis harder.

An advanced lab integrates the state machine with the Isaac adapter from Lecture 20. Inject a fixed 80 ms delay and occasional 200 ms stalls. Compare synchronous pause-during-inference, naive async replacement, and your continuity-aware scheduler. Use the same task starts and label each scheduling policy precisely. No result from a CPU timing fixture demonstrates physical real-time guarantees.

## Self-check and answer guidance

1. Can an eight-action chunk always hide eight periods of latency? No. Delay consumes the chunk's useful future horizon, and commands may become invalid as the world changes.
2. Why use a reset epoch as well as a timestamp? Simulation time can restart, so an old result can have a numerically plausible timestamp in a new episode.
3. Does dropping old frames preserve a history-conditioned policy? Only if the resulting history matches its sampling and temporal contract.
4. Why can faster average inference worsen control? Increased jitter, queueing, or larger batches can increase tail observation age despite a better average compute time.

---

[Previous](Lecture-21.md) · [Lab guide](Lab-Guide.md) · [Next](Lecture-23.md)
