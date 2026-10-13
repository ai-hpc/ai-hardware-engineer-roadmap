# Zigbee - Step-by-Step Lectures

Work through these in order. Each file is one session: read it, do the lab at the end, then move on.

**Hub:** [Zigbee course guide](../Guide.md)

| Lecture | Topic |
|---------|-------|
| [Lecture-01](Lecture-01.md) | What Zigbee is and where it fits |
| [Lecture-02](Lecture-02.md) | Roles, topology, and network formation |
| [Lecture-03](Lecture-03.md) | The Zigbee stack: ZDO, APS, endpoints, and clusters |
| [Lecture-04](Lecture-04.md) | Security, commissioning, sleepy devices, and OTA |
| [Lecture-05](Lecture-05.md) | ESP32-C6 practical path: devices, NCP, gateway, and Jetson context |

## Practice and assessment

**Entry diagnostic (prerequisite check):** Explain coordinator/router/end-device roles and endpoint/cluster addressing. If this is unfamiliar, use the prerequisite material above or the [roadmap diagnostics](../../../../../Learning-Guide.md#diagnose-before-reading) before starting the lab.

1. **Baseline lab:** Implement or inspect a sensor attribute report and its coordinator-side receipt.
2. **Guided variation:** Rejoin after a controlled disconnect and test duplicate or stale reports.
3. **Independent transfer:** repeat the method on a new input or configuration and explain where your original assumptions stop applying.

**Evidence to submit:** Report delivery, latency, retry count, rejoin time, and endpoint/cluster mapping. Keep raw results and reproduction steps; distinguish measured results from estimates or work not run.

**Self-check:** Which observation would disprove your explanation of the result? Identify a competing explanation and the smallest experiment that separates them. A passing answer names a concrete predicted difference, not only another tool to try.

**Exit criteria:** reproduce the baseline, demonstrate the negative/variation check, and meet the [shared completion rubric](../../../../../Learning-Guide.md#completion-rubric). Use the [lesson cycle](../../../../../Learning-Guide.md#a-repeatable-lesson-cycle) for retrieval practice and spaced review.
