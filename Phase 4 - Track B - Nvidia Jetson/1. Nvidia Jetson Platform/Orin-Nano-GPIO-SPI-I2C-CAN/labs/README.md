# CPU lab: decode a sensor message across C and Python

**Time:** 45–60 minutes. **Prerequisites:** Python 3; Linux and a C compiler for the native ABI check. No Jetson, CAN adapter, root privileges, or third-party Python packages are required.

You will read a small protocol, predict its bytes, run an actual C/Python interoperability check, and diagnose missing or repeated messages. The experiment uses real local datagram sockets. It does **not** create a SocketCAN interface, access a controller, or measure physical CAN timing.

## 1. Read the application contract

Our fictional distance-and-wheel-speed sensor sends standard Classic CAN data frames with ID `0x321`. Its eight-byte payload is:

| Byte(s) | Meaning | Encoding |
| --- | --- | --- |
| 0 | Schema version | Unsigned 8-bit, exactly 1 |
| 1 | Sequence within a known sender session | Unsigned 8-bit, wraps at 255 |
| 2–3 | Wheel speed, mm/s | Signed 16-bit little endian |
| 4–5 | Distance, mm | Unsigned 16-bit little endian |
| 6–7 | Age declared by sender, ms | Unsigned 16-bit little endian |

All these units and fields are application decisions; CAN does not define them for us. The maximum representable distance is 65.535 m. Larger values must be rejected or represented by a revised schema, not silently wrapped. The example supplies no special “invalid distance” sentinel; malformed/out-of-range input is rejected.

**Predict:** sequence 127, speed −250 mm/s, distance 1,200 mm, sender age 15 ms should produce `01 7F 06 FF B0 04 0F 00`. Derive the speed bytes yourself: −250 modulo 65,536 is 65,286 = `0xFF06`; little endian places `06` first.

## 2. Distinguish payload bytes from the Linux ABI

The application payload is eight bytes. Linux's `struct can_frame` record is 16 bytes: identifier/flags, payload length, reserved/padding fields, and an aligned eight-byte data area. Its integer fields use the host ABI, while our payload explicitly uses little endian.

`can_abi.c` includes the actual Linux header and uses compile-time assertions for record size and offsets. It emits an independently written golden payload and checks an input record. `sensor_protocol.py` packs/unpacks the corresponding record with Python `struct`.

This record is not the serialized electrical CAN frame. Arbitration, control, CRC, acknowledgement, bit stuffing, and bus timing are not included by counting its 16 bytes. It is also not a portable file format across hosts with different native endianness.

## 3. Run the checks

From the repository root:

```bash
python3 -m unittest discover -s 'Phase 4 - Track B - Nvidia Jetson/1. Nvidia Jetson Platform/Orin-Nano-GPIO-SPI-I2C-CAN/labs' -v
python3 'Phase 4 - Track B - Nvidia Jetson/1. Nvidia Jetson Platform/Orin-Nano-GPIO-SPI-I2C-CAN/labs/sensor_protocol.py'
```

**Actual validation, 2026-10-18:** 11 tests passed on Linux x86-64, Python 3.12.3, GCC 13.3.0. The native test compiled with C11, `-Wall -Wextra -Werror -O2`, exchanged records in both directions with Python, and rejected a truncated record and a changed payload. On a machine without Linux/`cc`, that check is explicitly skipped; the Python checks alone are not native ABI validation.

The remaining checks cover the golden vector, extreme signed/unsigned values, invalid types/ranges, every short record length, concatenated records, flag/ID confusion, invalid payload versions/lengths, sequence wraparound, duplicate/lost/reordered messages, and actual datagram message boundaries. Sources and hashes are recorded in [validation.json](validation.json); the demonstration output is in [example-result.json](example-result.json).

## 4. Diagnose the sequence log

The demonstration receives sequences `254, 255, 0, 0, 3, 2`:

| Sequence | Interpretation |
| --- | --- |
| 254 | First observation in this tracker session |
| 255 | Next sample |
| 0 | Valid wraparound |
| 0 | Duplicate; do not advance |
| 3 | Forward gap; two samples absent |
| 2 | Old or ambiguous; do not advance |

Modulo counters alone cannot distinguish arbitrary long gaps, restarts, and reordered traffic. This tracker accepts forward deltas 1–127 and treats deltas 128–255 as old/ambiguous. It assumes at most 126 missed frames between accepted observations. At 100 Hz, a silence long enough to violate that assumption requires an explicit resynchronization/session rule. It does not authenticate a sender or automatically identify a reboot.

**Exercise:** add 128 after an initial zero. Explain why rejecting it is more honest than calling it definitely “128 missing samples.” Then add an explicit session identity to a new protocol version and describe how its larger message changes the transport design.

## 5. Reject the wrong kind of frame

An extended frame can have the same low identifier bits as a standard frame. A remote frame requests data rather than carrying our application payload. An error record reports a different class of event. The decoder checks the complete identifier/flags and refuses all of these for schema 1.

The lab also shows a masked identifier comparison equivalent to a standard-data-frame filter. Kernel error-frame subscription uses a separate SocketCAN option, so this helper is not a complete socket configuration. Keep decoder validation even when the transport is filtered.

**Exercise:** change the version byte from 1 to 2. The decoder must reject it. A version mismatch should not be “fixed” by interpreting the same bytes under a schema that happened to fit their length.

## 6. Explain what the age field cannot prove

A received age value of 15 ms is the sender's declaration at its chosen point in processing. The message may then spend an unknown time in queues and transit. Starting a monotonic timer when it arrives lets the receiver measure additional time since receipt; it does not reconstruct the missing transport delay.

A capture-age guarantee needs a justified clock mapping, a bounded transport/queue model, or another protocol design that establishes the necessary information. This lab intentionally does not use its age field to authorize an actuator. A CRC, sequence, and recent receive timestamp cannot substitute for that missing evidence.

## 7. Choose the next experiment

A Linux `vcan` exercise can test real SocketCAN sockets, filters, and delivery without a physical controller. It needs kernel support and appropriate namespace/network administration permissions. This review host could not create a `vcan` device, so no `vcan` result is claimed here.

A physical extension uses two bench nodes, verified transceivers and wiring, supported bit timing, and an explicit receiver log. Compare emitted payloads with received payloads, then introduce a disconnected sender and check the application's timeout policy. Controller loopback, virtual CAN, and physical two-node tests establish different things.

**Completion:** submit the golden-byte derivation, the sequence diagnosis, passing test output with any skips identified, and a short explanation of why this lab does not yet establish fresh robot observations. Continue with the [peripheral integration course](../Guide.md).
