# First embedded lab: keep an observation fresh

Replay a sensor stream through a slower processing worker and decide which results are still useful. The same reasoning applies to an MCU queue, a Linux inference service, or a ROS 2 perception callback.

**Layers:** L3 runtime, L4 embedded, L1 application. **Roles:** embedded, edge AI, and robotics engineers. **Prerequisites:** Python lists and functions; [numbers, shapes, and bytes](../../Phase%201%20-%20Foundational%20Knowledge/Labs/Numbers-Shapes-and-Bytes.md) is a useful first step. **Next:** [embedded software](../2.%20Embedded%20Software/Guide.md), [embedded Linux](../3.%20Embedded%20Linux/Guide.md), and [robotics](../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20D%20-%20Robotics/Guide.md).

**Access:** any computer with Python; no board or simulator installation. **Time:** 5 minutes setup, 30–45 minutes experiments, 20 minutes explanation. **Artifact:** a JSON event trace, fault-check results, and a queue/deadline decision. This is a deterministic simulation of queue behavior. It cannot validate board timing, interrupt latency, sensor calibration, middleware behavior, or a robot controller.

## Why a correct answer can arrive too late

A camera captures a person at one position. Processing returns the correct position from that image, but the person has moved. Numerical correctness and freshness answer different questions.

```text
capture → wait in queue → process → check age → publish or reject
             15 ms        25 ms
age at completion = queue wait + processing time = 40 ms
```

This lab uses one simulated clock, zero transport time, observations every **10 ms**, processing time **25 ms**, and maximum permitted age **40 ms**. The sensor produces `100 observations/s`; one worker can finish at most `40 observations/s`. A queue cannot repair this sustained rate mismatch. It can absorb a short burst, drop observations, or accumulate delay.

The contract accepts age **≤ 40 ms**. A result at exactly 40 ms passes; 41 ms fails. Choosing the boundary explicitly prevents two components from disagreeing.

## Predict before running

For a FIFO queue, observation 0 is captured at 0 ms and completes at 25 ms. Observation 1, captured at 10 ms, then completes at 50 ms: age 40 ms. Observation 2, captured at 20 ms, completes at 75 ms: age 55 ms. All three computations are correct, but the third misses the freshness contract.

A **latest observation** queue keeps only the newest waiting observation while the worker is busy. It does not cancel work already running. At 25 ms it begins observation 2, which finishes at 50 ms with age 30 ms. The improvement comes from dropping obsolete waiting work.

Use this policy only when intermediate observations may be skipped. It is inappropriate for a byte stream, transaction log, or an integration algorithm that requires every sample. An IMU estimator may need a different policy from a camera-based planner.

## Run two queue policies

**Tested here:** Python 3.12.3 on Linux, standard library only. Run from the repository root. Each command finishes in under a second; no real waiting occurs.

```bash
python3 'Phase 2 - Embedded Systems/Labs/observation_freshness.py' --self-test --output /tmp/observation-freshness.json
cat /tmp/observation-freshness.json
```

Expect seven passing tests. The report compares FIFO with room for three waiting observations against replacement of a single waiting observation. A running observation is outside queue capacity. Equal-time arrivals are processed before completions; this explicit tie rule makes the fixture reproducible.

Read `simulate` as a timeline: take the next event, add an arrival or finish a computation, and start work if the worker is free. Read `FreshnessGate.check` as the contract at the output boundary. `nearest_rank` defines exactly how percentiles are computed.

The default **synthetic fixture** produces these exact counts:

| Policy | Completed | Published | Stale | Dropped | Maximum completed age |
|---|---:|---:|---:|---:|---:|
| FIFO, three waiting slots | 7 | 2 | 5 | 3 | 95 ms |
| Latest, one waiting slot | 5 | 5 | 0 | 5 | 35 ms |

These are event-model results, not measurements from a board or middleware implementation.

For both policies verify:

- `generated = completed + dropped` after the simulation drains.
- `queue_high_water ≤ waiting_capacity`.
- Every published result has age ≤ 40 ms.
- The FIFO trace contains stale completions. The latest policy publishes more fresh results in this fixture while explicitly dropping other observations.

Save the raw trace. A latency statistic over published results alone hides rejected work, so the report includes all completed ages, drop counts, status counts, and published maximum age. With only ten arrivals, p95 is a description of this fixture, not a reliable estimate of a deployment tail.

## Validate more than a timestamp

Each observation has a reset epoch, increasing sequence number, capture time, three values, units, and coordinate frame. The gate rejects:

| Fault | Why it matters | Expected status |
|---|---|---|
| Completion from before reset | An old asynchronous job must not enter the new run | `wrong_epoch` |
| Millimetres or a camera frame where metres in the base frame are required | Plausible numbers can represent different physical quantities | `wrong_contract` |
| Wrong value count or NaN | A backend can otherwise propagate invalid data | `invalid_values` |
| Capture time after completion | Clock domain, timestamp, or replay logic is inconsistent | `future_timestamp` |
| Duplicate or decreasing sequence | A late result must not replace a newer accepted result | `duplicate_or_reordered` |
| Completion after the age limit | Computation used an obsolete observation | `stale` |

The sequence rule assumes one ordered stream; multiworker pipelines need an explicit policy for out-of-order completions. The reset test constructs a new gate with a new epoch and demonstrates that an old completion is rejected while a new observation is accepted. This does not implement cancellation or resource cleanup; those belong in the service integration.

## Guided variation: change processing time

```bash
python3 'Phase 2 - Embedded Systems/Labs/observation_freshness.py' --service-ms 5 --output /tmp/observation-fast.json
python3 'Phase 2 - Embedded Systems/Labs/observation_freshness.py' --service-ms 45 --output /tmp/observation-slow.json
```

Predict both before running. At 5 ms all ten observations should be published without drops. At 45 ms even an observation with no queue wait is too old, so neither queue policy can publish a result. An empty set of published ages is represented as `null`, not zero latency.

**Independent transfer:** add a finite burst of slow service, such as one 60 ms job, and retain the original baseline tests. Compare FIFO and latest policy through the burst and recovery. State the rule for stale outputs and missing outputs before running. Add one test that would fail if the freshness check were removed. A real actuation boundary also needs a command-expiry/watchdog policy during silence; this observation gate does not implement one.

## Move from a fixture to a board

Use a monotonic clock for elapsed intervals within a clock domain, retain the original capture timestamp, and record the clock source. Python provides [`monotonic_ns`](https://docs.python.org/3/library/time.html#time.monotonic_ns); it is not a synchronized timestamp that can be subtracted blindly across computers. Sensor clocks, host clocks, and simulation clocks need an explicit mapping and uncertainty budget. Do not reset capture time on receipt: that hides transport delay.

Carry the same IDs and deadlines through DMA completion, inference, and publication. Measure capture-to-publication age as well as compute time. Middleware queue depth, reliability, deadline, and lifespan settings influence delivery, but application semantics still need checks; see the [ROS 2 QoS concepts](https://docs.ros.org/en/jazzy/Concepts/Intermediate/About-Quality-of-Service-Settings.html).

Before claiming hardware completion, record board/OS/runtime revisions, capture provenance, queue occupancy, all completion/drop counts, and recovery behavior under bounded load. A Linux process that meets observed deadlines is not thereby proven hard real-time.

## Self-check and answers

1. **Will doubling the queue solve a permanently overloaded worker?** No. It postpones dropping while increasing waiting time. Increase sustainable service capacity, reduce input load, or define which work can be discarded.
2. **Is inference time the same as observation age?** No. Age also includes capture/transport/queue delays and any postprocessing before the chosen boundary.
3. **Does latest-only always meet a deadline?** No. Processing alone can exceed the deadline. It can also discard information required by an algorithm.
4. **Why keep the epoch if sequence numbers exist?** A reset can restart sequence numbering while old work remains in flight. The epoch separates those runs.
5. **Can a timestamp from another computer be subtracted from `monotonic_ns()`?** Not without a defined clock mapping. Even equal-looking units do not imply equal origins or rates.
6. **Why reject the wrong frame rather than just rename it?** Transforming a vector requires the actual rotation/translation and their conventions. Changing a label does not transform the values.

**Exit criteria:** explain one stale trace by hand; reproduce the baseline and variations; catch each fault; choose a queue policy for a named sensor; and state which claims still require hardware. Carry the JSON schema and age contract into the embedded Linux observation service.
