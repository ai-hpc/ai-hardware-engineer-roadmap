# Arduino-ESP32 - Step-by-Step Lectures

Work through these in order. Each file is one session: read it, do the lab at the end, then move on.

**Hub:** [Espressif track guide](../Guide.md)

| Lecture | Topic |
|---------|-------|
| [Lecture-01](Lecture-01.md) | What `arduino-esp32` is and where it fits |
| [Lecture-02](Lecture-02.md) | Supported chips, install path, and first board bring-up |
| [Lecture-03](Lecture-03.md) | What is really under the Arduino layer: ESP-IDF, FreeRTOS, and core architecture |
| [Lecture-04](Lecture-04.md) | Peripherals, libraries, connectivity, and system design with Espressif boards |
| [Lecture-05](Lecture-05.md) | Arduino as an ESP-IDF component and the path from prototype to product |

## Practice and assessment

**Entry diagnostic (prerequisite check):** Build and flash a minimal example and identify board/SDK versions. If this is unfamiliar, use the prerequisite material above or the [roadmap diagnostics](../../../../Learning-Guide.md#diagnose-before-reading) before starting the lab.

1. **Baseline lab:** Extend a sensor or GPIO example with bounded task communication and connection-state logging.
2. **Guided variation:** Force reconnects and queue pressure while the sampling task continues.
3. **Independent transfer:** repeat the method on a new input or configuration and explain where your original assumptions stop applying.

**Evidence to submit:** Build manifest, heap use, sample jitter, connection recovery time, and error counters. Keep raw results and reproduction steps; distinguish measured results from estimates or work not run.

**Self-check:** Which observation would disprove your explanation of the result? Identify a competing explanation and the smallest experiment that separates them. A passing answer names a concrete predicted difference, not only another tool to try.

**Exit criteria:** reproduce the baseline, demonstrate the negative/variation check, and meet the [shared completion rubric](../../../../Learning-Guide.md#completion-rubric). Use the [lesson cycle](../../../../Learning-Guide.md#a-repeatable-lesson-cycle) for retrieval practice and spaced review.
