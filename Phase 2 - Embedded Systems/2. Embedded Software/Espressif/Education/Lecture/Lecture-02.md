# Lecture 2 - Embedded development basics you can demonstrate

**Course:** [Education guide](../Guide.md) | **Previous:** [Learning route](Lecture-01.md) | **Next:** [First ESP-IDF build](Lecture-03.md)

Before combining radios and AI, make the numbers and bytes concrete. The exercises below are small enough to solve by hand, then check in the [native C telemetry lab](../../../labs/Framed-Telemetry.md).

## 1. Decode before scaling

Suppose a teaching sensor returns bytes `FF 9C`, big-endian, as a signed 16-bit integer in hundredths of a degree Celsius. Combine the bytes into unsigned `0xFF9C = 65436`. Because its sign bit is set, interpret it as `65436 - 65536 = -100`. Apply the scale last: **−1.00 °C**.

If you reverse the byte order you get a completely different value; a valid I2C ACK cannot detect that mistake. If you subtract an offset or scale before establishing the numeric representation, plausible-looking output can hide the error. Real sensors may use packed fields or a different encoding: read the selected datasheet.

## 2. Count the bytes you actually store

An array of 100 three-axis samples represented as three `int16_t` values contains `100 × 3 × 2 = 600` payload bytes. A C struct can add padding; `sizeof` tells you the host representation, not a portable wire format. Serialize fields explicitly with a documented byte order and length.

A queue of ten pointers on a 32-bit MCU has 40 bytes of pointer payload, plus RTOS metadata. It does not contain ten independent sample buffers. Those buffers need separate storage and ownership. Return to the [ownership lecture](../../Lecture/Lecture-03.md) if this distinction is uncertain.

## 3. Budget a serial line

At 115200 baud with 8-N-1 framing, each byte takes ten wire bits. A 40-byte packet therefore needs about 3.47 ms. Sending 100 packets/s uses roughly 34.7% of nominal line capacity. USB serial buffering, application scheduling, and retransmission can add delay; this arithmetic establishes a capacity estimate, not end-to-end latency.

If the consumer pauses for 200 ms while production continues at 100 records/s, twenty new records arrive. Choose a bounded buffer or drop/backpressure policy before implementing it. For live robot state, replaying all old records after reconnect may be worse than keeping the latest valid one.

## 4. Separate compile, link, flash, and run

Compilation checks source against headers and produces object files. Linking resolves symbols and places code/data according to the target layout. Flashing writes an image to a device. Running tests checks observed behavior. Success at one stage does not imply success at the next.

Keep warnings enabled for your host logic. The course's C lab uses strict warnings plus AddressSanitizer and UndefinedBehaviorSanitizer, and includes deliberately broken candidates that the tests reject. The host toolchain does not reproduce MCU alignment, peripheral registers, or DMA automatically.

## Answer checks

- **`00 64` with the same format?** `100 × 0.01 = 1.00 °C`.
- **A struct has a `uint8_t` followed by `uint32_t`; is it five wire bytes?** Not necessarily. Host padding and byte order are implementation details; serialize five specified bytes explicitly if that is the protocol.
- **A CRC passes; can the record still be unusable?** Yes: it may be stale, duplicated, from another session, outside a physical range, or unauthorized.

Deliver the four calculations and a passing parser run. Explain one rejected test input before continuing.
