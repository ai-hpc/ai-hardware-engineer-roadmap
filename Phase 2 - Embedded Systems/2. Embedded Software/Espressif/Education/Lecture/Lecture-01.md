# Lecture 1 - Turn Espressif Education into a learning route

**Course:** [Education guide](../Guide.md) | **Next:** [Prerequisites you can demonstrate](Lecture-02.md)

Use [Espressif Education](https://www.espressif.com/en/ecosystem/education) to locate official hardware, examples, and solution families. Then choose one measurable learning route. A vendor catalogue is useful for finding resources; your course progress should be demonstrated by artifacts you can explain and reproduce.

Start with a **sensor companion for a robot**. It reads one observation, validates it, and sends a timestamped report to a host. The initial output is a log or indicator. This keeps the first experiment small while teaching the same data and failure boundaries needed by larger AI systems.

## A route with useful stopping points

| Stage | Task | Evidence before continuing |
|---|---|---|
| Foundations | Parse a bounded binary record on your CPU | Corrupt, truncated, oversized and wrapped-counter inputs handled |
| Board entry | Build, flash, and print a heartbeat | Chip/board/framework identity plus repeated boot observations |
| Sensor | Read identity, status and one measurement | Correct units, failure code, last-valid time |
| Transport | Send a structured record to a host | Session/sequence/freshness checks and disconnect behavior |
| Specialization | Add one AI, low-power or gateway feature | A budget and a test that could disprove success |

The [framed telemetry lab](../../../labs/Framed-Telemetry.md) completes the first row without hardware. The [Arduino course](../../Guide.md) covers entry and peripheral integration. The [OpenThread CPU lab](../../../IoT/OpenThread/labs/CPU-Simulation.md) exercises a real network stack without purchasing radios. CPU experiments have explicit limits; they do not measure your MCU's power or interrupt latency.

## Select a project by constraints

Write three facts before selecting a board: required physical interfaces, data rate and freshness requirement, and power source. A camera stream, a voice trigger, and a battery temperature sensor have different memory/radio constraints. Buying the most capable board does not resolve an unspecified workload.

Example teaching requirements:

- One sensor record every 100 ms, with units and source session.
- Display “unavailable” after 500 ms without a valid record.
- Reconnect without treating buffered old data as current.

These numbers are exercise choices, not validated robot safety limits. They let you write concrete tests now and replace the limits when a real application provides its requirements.

## Exercise and worked answer

Choose one feature from the education catalogue and write its dependency chain. For a voice indicator: audio capture → sample format → feature extraction/model input → inference → threshold/debounce → visible state. A wake-word model is not the audio driver, and its score is not permission to move a robot.

Mark each dependency as already demonstrated, available as an official example, or still to be implemented. Stop adding features when a prerequisite is unexplained. Your deliverable is one route, one input/output contract, and one first runnable test—not a list of everything the vendor supports.
