# Lab: turn a byte stream into a checked telemetry record

A UART read may contain half a packet, several packets, or the end of one followed by the start of another. DMA completion and a task wakeup are not application message boundaries. This lab builds a small native C parser that handles those cases with bounded storage, rejects damaged frames, recovers at the next delimiter, and distinguishes wrapping counters from ordinary increasing integers.

Allow 45–75 minutes. You need Python 3.10+ and a C11 compiler. No board or additional Python package is required. The code was compiled and executed with GCC 13.3 on Ubuntu 24.04; address and undefined-behavior sanitizers were also run. This establishes CPU behavior for the supplied tests, not MCU timing, DMA coherence, radio security, or RTOS scheduling.

## 1. Start with the wire contract

The decoded packet has this exact byte order:

| Field | Bytes | Interpretation |
|---|---:|---|
| Version | 1 | Exactly 1 for this exercise |
| Sequence | 2 | Unsigned 16-bit, least significant byte first |
| Payload length | 1 | 0 through 32 |
| Payload | length | Application-defined bytes |
| CRC | 2 | Least significant byte first, over version through payload |

The CRC is CRC-16/IBM-3740, also called CCITT-FALSE: polynomial 0x1021, initial value0xffff, no bit reflection, and no final XOR. The standard check string `123456789` gives 0x29b1. Name all parameters; “CRC16” alone is ambiguous.

The raw packet is encoded with **COBS**, then terminated by one zero byte. Encoded data contains no zero bytes, so zero provides a reliable boundary even when the payload itself contains zeros. For this lab's raw maximum of 38 bytes, COBS adds one code byte and the delimiter adds one more: maximum wire size is 40 bytes. This small encoder deliberately handles the course's bounded packet size; it is not presented as a general encoder for arbitrarily long payloads. [Original COBS paper](https://stuartcheshire.org/papers/COBSforToN.pdf).

The parser performs framing, format, and CRC checks before replacing the output record. A failed packet leaves the caller's previous output unchanged. That is useful, but the application must still mark old data stale instead of silently treating it as a new report.

**CRC is error detection, not authentication.** An intentional sender can calculate a valid CRC for false data. Network admission, authenticated messages, replay protection, and authorization are separate requirements for a real system.

## 2. Compile and run the implementation

From the repository root:

```bash
cc -std=c11 -O2 -Wall -Wextra -Wconversion -Werror \
  'Phase 2 - Embedded Systems/2. Embedded Software/labs/telemetry_frame.c' \
  'Phase 2 - Embedded Systems/2. Embedded Software/labs/telemetry_frame_test.c' \
  -o /tmp/course-telemetry-frame
/tmp/course-telemetry-frame
python3 -m unittest discover \
  -s 'Phase 2 - Embedded Systems/2. Embedded Software/labs' -v
```

The C suite checks 99 payload fixtures: every length 0–32 with zero, 255, and mixed-byte patterns. It splits each encoded packet at every byte boundary, then checks consecutive packets without resetting the parser. It also checks the known CRC vector, corrupted CRC, malformed COBS, truncation, overlong input, recovery, output preservation on failure, sequence wrap, and tick wrap.

The Python wrapper compiles the actual C sources, runs sanitizers, and builds two deliberately wrong variants. Bypassing the CRC check must fail the corrupted-frame test; accepting a one-tick future timestamp must fail the age test. Missing a compiler produces an explicit skip; a present compiler rejecting the source is a failure.

Read [telemetry_frame.h](telemetry_frame.h), [telemetry_frame.c](telemetry_frame.c), and the [C tests](telemetry_frame_test.c). `telemetry_feed` consumes one byte and returns a status. You can call it repeatedly over any received chunk; the state machine does not care how the driver split the stream.

## 3. Explain recovery, not just the happy path

The parser has two modes: collecting an encoded packet, and discarding an overlong packet. It stores at most 39 encoded bytes. When another nonzero byte arrives, it returns `TELEMETRY_TOO_LONG` once, clears its buffered length, and discards until a zero delimiter. The delimiter returns it to collection mode.

This policy bounds storage even if a disconnected or faulty sender supplies an indefinitely long nonzero stream. It does not bound how long the application waits for useful data. An application-level receive timeout must handle silence and endless malformed traffic.

A truncated packet followed by zero is rejected. A later well-formed packet is accepted. If the delimiter itself is lost, two packets may merge and be rejected together; resynchronization occurs at a later delimiter. Framing recovery does not guarantee delivery of every packet.

**Exercise:** why not return the partially decoded payload when a CRC fails? **Answer:** the receiver does not know which fields were damaged. The sequence, length, or payload might be wrong. Publish only a fully checked record and separately report failure/freshness state.

## 4. Handle sequence and timer rollover deliberately

An unsigned 16-bit sequence wraps from 65535 to 0. The helper computes `difference = (uint16_t)(candidate - previous)` and accepts a newer sequence only if the difference is nonzero and below 32768. Equal means duplicate; the opposite half-range means old or ambiguous.

This comparison assumes a shared session and a bounded separation of less than half the sequence space. It cannot determine order after an arbitrarily long disconnection. A device reboot also needs an explicit session identifier or handshake; a small sequence number must not silently establish a fresh session. The lab helper is not an authenticated replay-protection system.

For a wrapping 32-bit tick counter, elapsed time is unsigned `now - sampled`. The lab accepts only elapsed values below half the counter range and no greater than the configured maximum age. It rejects a maximum-age window reaching half the range.

**Worked rollover:** sampled tick is 4,294,967,293 and now is 3. Unsigned subtraction gives 6 ticks, so a six-tick limit accepts it and a five-tick limit rejects it. A sampled value 100 with now 99 instead appears almost a full counter range old and is rejected as outside the permitted window. These calculations assume timestamps use the same clock and unit. Subtracting unrelated MCU and Linux clocks is invalid without a synchronization/offset model.

**Exercise:** may you use the receive timestamp as the sensor sampling timestamp? **Answer:** only if the interface explicitly defines age since reception. It does not measure how old the underlying observation was before transmission. Carry a sample timestamp or state the weaker guarantee.

## 5. Connect it to an MCU without changing the contract

Keep parser state owned by one task or execution context. An ISR can notify a task or place bytes/descriptors into a supported bounded transport. The task feeds complete available byte ranges into the parser, handles status counters, and publishes accepted records with receive time and freshness state.

If DMA fills a circular buffer, define who owns each region, how wrap is detected, what happens on overrun, and when cache maintenance is required. Do not parse bytes the DMA may still overwrite. A `volatile` qualifier neither creates buffer ownership nor makes a cache coherent.

For FreeRTOS, use the appropriate `FromISR` APIs only from permitted interrupt priorities. Stream/message buffers assume a single writer and reader unless access is explicitly serialized. A queue or notification transport needs an overflow policy and a wakeup/completion rule; the parser's bounded packet size does not prevent the upstream queue from overflowing. [FreeRTOS stream buffers](https://github.com/FreeRTOS/FreeRTOS-Kernel/blob/main/include/stream_buffer.h), [Cortex-M interrupt rules](https://www.freertos.org/Documentation/02-Kernel/03-Supported-devices/04-Demos/ARM-Cortex/RTOS-Cortex-M3-M4).

For a robot sensor, publish a record such as `{session, sequence, sample_time, receive_time, payload, validity}`. The wire fixture here includes only sequence and payload; adding session/timestamp fields requires a versioned format and updated size/CRC tests. A radio address or valid CRC alone does not establish freshness.

## 6. Extend one requirement and keep the negative tests

Choose one extension: an application timeout, a session identifier, a typed temperature payload, or an additional protocol version. Write the new contract first. Include an unsupported-version or invalid-unit test and preserve the existing corrupt/truncated/overlong cases.

Your deliverable is a wire-layout table, the passing native test report, one rejected fault trace, and an ownership diagram from UART/DMA to parser to application. Then continue to [observation freshness](../../Labs/Observation-Freshness.md), [OpenThread](../IoT/OpenThread/Guide.md), or [Zigbee](../IoT/Zigbee/Guide.md). The transport changes; the application still needs unambiguous data, ownership, ordering, and age.
