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

## Turn the five lectures into a measured firmware increment

After L1, draw the stack and locate one called function in the pinned source. After L2, save a boot log and exact board/core settings. After L3, record which task runs application work and which callback context delivers a peripheral event. After L4, add a sequence-numbered periodic sample and measure missed samples during a reconnect. After L5, reproduce that behavior in the selected ESP-IDF/component workflow.

A written comparison alone is preparation. The course artifact is a rebuildable project, serial log, queue/drop counters, and a before/after migration comparison. If hardware is unavailable, label source tracing as analysis and leave runtime measurements unclaimed.

**Carry forward:** Five staged evidence items culminating in a rebuildable firmware project. Continue with [the next course](../Education/Guide.md) and reuse that evidence.
