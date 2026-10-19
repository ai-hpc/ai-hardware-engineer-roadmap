# Lecture 1 - What `arduino-esp32` is and where it fits

**Course:** [Espressif guide](../Guide.md) | **Next:** [Choose a board and bring it up](Lecture-02.md)

`arduino-esp32` is Espressif's implementation of the Arduino programming model for supported ESP32-family chips. It gives you `setup()`, `loop()`, familiar peripheral APIs, and a board-package workflow. Underneath, it uses ESP-IDF drivers, FreeRTOS, and chip-specific startup support. This course uses that approachable entry point to build a sensor node whose failures you can explain.

You need basic C/C++, the ability to build a program, and the distinction between a voltage supply and a signal pin. Start with the [native C telemetry lab](../../labs/Framed-Telemetry.md) if you have no board. Its parser and tests run on a normal computer; a board adds peripheral and timing behavior that must be measured separately.

## Trace one byte through the system

Imagine a robot's sensor module reporting a temperature to a Linux computer:

```text
physical sensor → I2C peripheral → driver → application record
               → bounded transmit queue → UART driver → wire → host parser
```

Arduino's `Wire` API helps perform the bus transaction. Your code still decides what register to read, how to convert its bytes, whether the sample is fresh, and what to do when the bus fails. `Serial.write()` hands bytes to a driver; returning from that call does not necessarily mean the last stop bit is already on the wire.

This distinction makes debugging manageable. An address NACK points you toward wiring, address, power, or device state. A valid read with an implausible value points toward configuration, byte order, signedness, scaling, or calibration. A correct value arriving too late points toward scheduling and queues. “The Arduino library is broken” is too broad to test.

## Chip, module, board, framework

| Term | What it supplies | What you must still identify |
|---|---|---|
| SoC | CPU, memory interfaces, peripherals and selected radios | Exact chip and revision |
| Module | SoC plus supporting components, often flash and RF integration | Flash/PSRAM options, antenna and integration rules |
| Development board | Module or SoC plus power, USB/debug and headers | Board revision, schematic, pin routing, USB path |
| Arduino core | APIs and build integration for supported targets | Core version, selected board and options |
| Application | Your sensing, communication, and control behavior | Data formats, deadlines, error handling and tests |

Two boards using the same SoC can route different pins to LEDs, USB, flash, or external connectors. A library name does not establish that its assumed pins are safe for your board.

## Choose the smallest complete application

For this course, begin with a once-per-second serial heartbeat, then a bounded sensor read, then a framed observation. Add one network transport after those work. Keep motor actuation outside the first experiment: prove observation, age, and disconnect behavior before connecting a control path.

A useful module boundary is `read_sample()` returning either a valid record or an explicit error. Keep conversion and validity logic independent of `Wire`, `Serial`, or Wi-Fi classes. You can then test malformed bytes, extreme values, and counter wrap on a CPU while hardware-specific integration remains small.

Arduino can remain the application layer in a well-engineered product. ESP-IDF can provide more direct configuration and component control when required. The choice follows the application's constraints; changing framework names does not automatically improve architecture.

## Exercise with an example answer

Draw the sensor-to-host path and mark where each fault is detected: absent sensor, corrupt UART packet, duplicate record, stale record, and restarted sender.

A reasonable answer is: driver detects failed bus transaction; parser checks packet integrity and format; application checks sequence within a session; consumer checks observation age; a boot/session identifier distinguishes restart from an old duplicate. No single checksum solves all five problems.

Your output is one diagram and a one-page interface contract. In the next lecture you will record enough board and software identity to reproduce its first heartbeat.

Primary references: [official Arduino ESP32 project](https://github.com/espressif/arduino-esp32), [Arduino ESP32 documentation](https://docs.espressif.com/projects/arduino-esp32/en/latest/), and [ESP-IDF FreeRTOS behavior](https://docs.espressif.com/projects/esp-idf/en/stable/esp32/api-reference/system/freertos_idf.html).
