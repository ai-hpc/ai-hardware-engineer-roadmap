# Lecture 5 - Build one complete Espressif learning project

**Course:** [Education guide](../Guide.md) | **Previous:** [Specialization budgets](Lecture-04.md)

Finish with a sensor companion that another learner can build and diagnose. Use either a real board or the CPU parser plus OpenThread simulation path; label the boundary clearly. The course's CPU tests are available now, while board behavior must come from your own observations.

## Define the contract

Use a teaching requirement of one observation every 100 ms and an unavailable indication after 500 ms without a valid observation. Choose units, signedness, allowable range, session identifier, sequence policy, and clock interpretation. These times are exercise settings, not safety limits for a physical robot.

The receiver accepts a record only after checking format, identity, sequence/session, and freshness. It distinguishes a valid zero from missing data. It reports the last valid sample time separately from receive time. If sender/receiver clocks are independent, do not calculate one-way age by subtracting them without an offset/synchronization assumption.

## Build in stages you can inspect

1. Run the [C framing tests](../../../labs/Framed-Telemetry.md) and explain one corruption case.
2. Record a board heartbeat, or run the [OpenThread CPU lab](../../../IoT/OpenThread/labs/CPU-Simulation.md) if no board is available.
3. Add one sensor or explicitly labeled synthetic data source. Preserve status and original sample time.
4. Introduce a slow consumer and a connection outage. Verify bounded memory and the selected discard/backpressure behavior.
5. Add one feature from Lecture 4: an inference indicator, measured low-power schedule, or protocol gateway. Keep the earlier failure tests.

Do not quietly replace a real sensor requirement with generated data. Synthetic data is useful for deterministic protocol tests; it does not establish sensor accuracy or physical calibration.

## A test table with expected observations

| Test | Expected behavior | Evidence |
|---|---|---|
| Truncated record | Reject; accept a later valid record after resynchronization | Parser result and recovery trace |
| Queue saturation | Apply the documented bounded policy | Queue peak and discard/backpressure count |
| Duplicate sequence | No duplicate application event within the same session | Accepted/rejected event log |
| Restart | New session handled without accepting old records | Boot marker and session transition |
| Network outage | Data becomes unavailable within the exercise policy | Last-valid time and status transition |
| Reconnect | Only policy-eligible data becomes current | New sequence and original sample time |
| Invalid AI input / low confidence | Remain in the documented non-actuating state | Input case and deterministic decision |

On a physical device, add power/reset, peripheral timeout, and long-duration operation checks. Observe temperature and memory behavior over the relevant workload. A ten-second happy-path demonstration does not establish long-term stability.

## Package the evidence

Deliver source/configuration, exact target and dependency versions, a build/run procedure, input fixtures, test results, and the measurements you actually made. Include a short list of pending hardware checks. Keep public test keys separate from deployment credentials.

For AI work, add dataset split/preprocessing, quantized model identity, error counts, latency distribution, and peak memory. For power work, add measurement location, voltage, per-state duration/current, retries, and the resulting average. For a gateway, add the field mapping and source-clock assumptions.

## Final answer check

Can a teammate explain why a received packet was rejected, reproduce a reset/reconnect, and tell whether a displayed value is old? If yes, you have built an understandable embedded system with a useful contract. If the only evidence is “the example compiled,” return to the missing stage before adding another feature.

Continue into [IoT](../../../IoT/Guide.md), or carry the same observations into the later robotics and AI deployment courses. The interfaces and failure evidence remain useful even when the model, radio, or processor changes.
