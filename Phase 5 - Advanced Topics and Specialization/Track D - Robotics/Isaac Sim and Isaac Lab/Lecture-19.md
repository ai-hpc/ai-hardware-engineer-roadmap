# Lecture 19 — ROS 2 timing and exported policy contracts

**Prerequisites:** Lectures 08, 13, and 15–18. **Time:** 3–4 hours. **Outcome:** demonstrate that an exported policy receives the intended observations and acts on fresh data.

## Export more than weights

Save observation names/order, units, frame convention, normalizer statistics, history length, recurrent state handling, action scale/clipping, joint order, policy period, and model/runtime versions. A numerically correct network with a wrong normalizer is a different controller.

First compare reference and exported actions on identical saved observations. Then compare closed-loop behavior: small action differences can accumulate into large trajectory differences.

## Two clocks, one explicit boundary

Simulation time may pause, reset, or run faster than wall time. Use ROS simulation time consistently for simulated message semantics, and a monotonic wall clock for local execution-duration measurement. Cross-process wall latency requires synchronized clocks or a round-trip design.

Log sensor capture, observation assembly, policy start/end, command publication, and command application. Name the clock used by every field.

For a hypothetical 50 Hz controller, the nominal period is 20 ms. A 4 ms policy does not guarantee the deadline when capture, queues, preprocessing, transport, and actuation also consume time.

## Execute the transport boundary first

The [atomic ROS exercise](Simulation-CPU-Lab.md#3-atomic-observations-and-simulator-owned-actions) actually transports requests and replies locally, including duplicate, stale, prior-episode and mixed-sample failures. The simulator owns the request clock and echoes are compared on that same clock. A policy process must not advance its integrated target merely because it sent a reply: the reply may never be accepted. Carry the last **applied** action in the next observation.

## Lab

1. Replay saved observations through reference and exported policy.
2. Check action tolerances and joint-name mapping.
3. Enable the ROS bridge in a single-robot simulated scene.
4. Record the observe-to-act timeline and p50/p95/p99 latency with sample count.
5. Pause/reset simulation time and verify handling.
6. Delay or duplicate a command in the test harness; verify stale commands are rejected or replaced by the specified hold behavior.
7. Re-evaluate success under realistic delay/jitter.

The [ROS 2 clock design](https://design.ros2.org/articles/clock_and_time.html) explains time sources and jumps. For QoS and bridge configuration, use your pinned ROS/Isaac versions rather than mixing distro instructions.

## Freshness fixture

The supplied CPU exercise rejects future timestamps, expired commands, and mismatched episode IDs. Episode IDs matter because a command from the previous simulation episode can have an otherwise plausible timestamp after reset.

This small validator is a teaching component, not a certified robot safety controller. In this lab it runs against simulated or recorded events.

**Independent task:** change queue depth and compare latency tails and task outcomes. A reliable delivery mode that accumulates obsolete commands may be worse for a control loop than a small freshness-checked queue.

## Self-check

- Why is network inference timing insufficient? End-to-end observation age and command application determine control behavior.
- Why test clock jumps? Reset can invalidate chronological assumptions.
- Why evaluate closed-loop? Matching one action vector cannot establish stable trajectories.

**Exit artifact:** export manifest, action comparison, timing trace, stale-message test, and measured simulated task outcomes.

[Previous](Lecture-18.md) · [Lab sequence](Lab-Guide.md) · [Next: transfer](Lecture-20.md)
