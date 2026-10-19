# Lecture 3 - ESP-IDF, FreeRTOS, and ownership under the Arduino layer

**Course:** [Espressif guide](../Guide.md) | **Previous:** [Bring-up](Lecture-02.md) | **Next:** [Peripherals and connectivity](Lecture-04.md)

Your `loop()` runs inside a system that also schedules framework, driver, and network work. A call that blocks your application task does not necessarily stop all tasks, but it can still make your observation late or exhaust a queue. Begin by defining who owns each buffer and how long each step can take.

## Follow a sample, then follow its memory

Use three conceptual stages: acquire a sample, validate/convert it, send a record. They need not be three tasks. One task with a small state machine is often easier to reason about until measurements justify concurrency.

If stages become tasks, a FreeRTOS queue copies its configured item bytes. A queue of `Sample` values copies records; a queue of `Sample*` copies only addresses. Sending a pointer to a local array that goes out of scope is invalid. Sending a pointer to a reusable global array can also be wrong if acquisition overwrites it before transmission finishes.

Two valid starting designs are:

1. A queue of small value records, with bounded capacity and an explicit full-queue policy.
2. A fixed buffer pool with a single owner for each buffer and a documented transfer/return protocol.

Do not call the second design zero-copy until you have identified every hidden driver copy and proved lifetime through completion. The [native C parser](../../labs/Framed-Telemetry.md) deliberately uses bounded storage and one parser owner to keep this contract visible.

## Work a queue budget

Suppose acquisition produces 100 records/s and formatting can sustain only 40 records/s. The backlog grows by 60 records/s; a 12-record queue fills in roughly 0.2 seconds from empty under this simple steady-rate model. Increasing the queue does not fix sustained overload.

If every observation is needed for logging, reduce production, increase service capacity, or backpressure the producer where the physical interface permits it. If a robot only needs the freshest observation, replacing an old pending record may be appropriate. Count discarded records and make that policy visible. A control command queue needs different semantics from replaceable telemetry.

## Scheduling details that change copied examples

ESP-IDF modifies FreeRTOS behavior for supported multicore targets. Task stack sizes in its API are **bytes**, while base FreeRTOS commonly specifies `StackType_t` elements. Do not pin a task to core 1 without checking that the target and configuration have that core. Core affinity also does not isolate a task from shared memory, peripheral, or radio contention. See [ESP-IDF FreeRTOS documentation](https://docs.espressif.com/projects/esp-idf/en/stable/esp32/api-reference/system/freertos_idf.html).

Use blocking waits when there is no work. A busy high-priority loop can starve lower-priority work. Keep ISR operations within the selected port's supported `FromISR` APIs and priority rules. Network callbacks may run in another task context; record their documented context before sharing state or making blocking calls. [`WiFi.onEvent` documentation](https://docs.espressif.com/projects/arduino-esp32/en/latest/api/wifi.html) explicitly discusses callback threading.

`volatile` is appropriate for some hardware accesses. It does not make a shared counter increment atomic, protect a multi-field record, or create a task-to-task synchronization protocol. Use the platform's synchronization facilities and choose the smallest critical section that protects a real invariant.

## Measure before splitting tasks

Log sample time, enqueue time, dequeue time, processing completion, queue depth, and drop count. Report observation age at the consumer, not only task CPU time. Avoid flooding the console while measuring: logging consumes CPU, buffers, and wire bandwidth.

A 40-byte record sent as UART 8-N-1 at 115200 baud occupies approximately `40 × 10 / 115200 = 3.47 ms` of wire time. At 100 records/s it uses about 35% of nominal line capacity before other traffic. `Serial.write` latency, actual wire completion, and end-to-end record age are different measurements.

## Practice with answers

- **A queued pointer points to a stack-local sample. Is queue send enough?** No. The payload lifetime must outlast its final consumer; copy the value or use owned persistent storage.
- **A 40 Hz consumer receives 100 Hz input. Does a larger queue make samples fresher?** It usually allows more backlog. Choose a service-rate or data-retention policy.
- **Does assigning a task to a core prove a deadline?** No. Establish execution, blocking, interference, and release-time assumptions, then measure/analyze them.

Run [Keep an observation fresh](../../../Labs/Observation-Freshness.md), then draw the ownership and queue policy for your board. Its deterministic CPU results explain the contract; they do not establish ESP32 interrupt or scheduling latency.
