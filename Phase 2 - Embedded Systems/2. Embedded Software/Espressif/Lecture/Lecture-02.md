# Lecture 2 - Supported chips, install path, and first board bring-up

**Course:** [Espressif guide](../Guide.md) | **Previous:** [Platform layers](Lecture-01.md) | **Next:** [Tasks and ownership](Lecture-03.md)

The first milestone is a reproducible boot log and serial heartbeat. You do not need a sensor, network credentials, or a custom PCB yet. The board instructions below are documented exercises; this course revision did not flash a physical ESP32. Use the [CPU telemetry lab](../../labs/Framed-Telemetry.md) if hardware is unavailable.

## Record the platform before installing examples

Keep the SoC, board name/revision, module marking, flash/PSRAM size, USB connector used, board profile, core version, and host OS in your lab notes. Check the board schematic rather than assuming that `LED_BUILTIN` is a plain GPIO LED; some boards use an addressable RGB device or no LED.

Selected radio differences useful for this course:

| Chip | Relevant built-in connectivity | Consequence |
|---|---|---|
| ESP32-C3 / ESP32-S3 | Wi-Fi and Bluetooth LE | Do not assume a native 802.15.4 radio |
| ESP32-C6 | Wi-Fi, Bluetooth LE, IEEE 802.15.4 | Can support Thread/Zigbee with an appropriate software stack |
| ESP32-H2 | Bluetooth LE and IEEE 802.15.4 | Do not select it for a standalone Wi-Fi exercise |
| ESP32-S2 | Wi-Fi | Do not copy a BLE example to it |
| ESP32-P4 | No integrated Wi-Fi/Bluetooth radio | A wireless board may add a separate companion chip |

This is a task-oriented subset, not a complete support matrix. Verify the [official boards table](https://docs.espressif.com/projects/arduino-esp32/en/latest/boards/boards.html), including the component/static-library support route for C2 and C61. Radio capability alone does not establish that any two protocol roles can run together.

## Pin a compatible version

Reviewed 2026-10-18: stable Arduino ESP32 **3.3.12** is based on **ESP-IDF 5.5.5**; the releases page also lists a **4.0.0-RC1 prerelease**. Standalone ESP-IDF's stable documentation is **6.1**. These are different release lines. Installing the newest standalone IDF does not establish compatibility with the stable Arduino component. See the [3.3.12 release](https://github.com/espressif/arduino-esp32/releases/tag/3.3.12), [component compatibility](https://docs.espressif.com/projects/arduino-esp32/en/latest/esp-idf_component.html), and [IDF versions](https://docs.espressif.com/projects/esp-idf/en/stable/esp32c6/versions.html).

For Arduino IDE, use the official stable board-package URL in the [installation guide](https://docs.espressif.com/projects/arduino-esp32/en/latest/installing.html), install the selected core version, and select your exact board/options. Save the verbose compile output. For command-line builds, use [Arduino CLI](https://arduino.github.io/arduino-cli/latest/installation/) and record the fully qualified board name rather than relying on a previously selected IDE board.

## A serial-only first sketch

Create a sketch with this content. Select the board's documented serial/USB console settings and open its port at 115200 baud where the transport uses a UART.

```cpp
#include <Arduino.h>

uint32_t last_report_ms = 0;
uint32_t sequence = 0;

void setup() {
    Serial.begin(115200);
    // Do not wait forever for a USB terminal: the device must boot unattended.
    delay(200);
    Serial.print("chip=");
    Serial.println(ESP.getChipModel());
    Serial.print("sdk=");
    Serial.println(ESP.getSdkVersion());
}

void loop() {
    const uint32_t now = millis();
    if (static_cast<uint32_t>(now - last_report_ms) >= 1000u) {
        last_report_ms = now;
        Serial.printf("seq=%lu uptime_ms=%lu heap=%u\n",
                      static_cast<unsigned long>(sequence++),
                      static_cast<unsigned long>(now),
                      static_cast<unsigned>(ESP.getFreeHeap()));
    }
    delay(1);  // Yield between checks; this is not a hard real-time scheduler.
}
```

Upload, observe ten reports, reset the board, and observe the sequence restarting. Record compile success and hardware execution as separate results. The unsigned subtraction handles `millis()` wrapping when checks occur sufficiently often; this reporting loop is not a precise periodic sampling clock. A later sensor record needs an explicit session identifier because sequence zero alone cannot distinguish reboot from wrap or replay.

## Diagnose the layer that failed

| Symptom | Check |
|---|---|
| Build fails before uploading | Selected target, core/library versions, first compiler error |
| No serial port | Data-capable USB cable, board connector, host permissions and driver |
| Upload cannot enter download mode | Board-specific boot/reset procedure and strapping pins |
| Upload succeeds, no text | Correct port, native USB CDC setting, console mapping, baud rate |
| Repeated boot banners | Reset reason, power supply/cable, watchdog or exception log |
| Board stops when terminal closes | Remove an unbounded `while (!Serial)` dependency |

**Answer check:** Why start with serial instead of blink? It avoids assuming an LED pin and reveals chip/SDK identity and repeated resets. It still depends on correctly selecting the board's console path.

Deliver the sketch, board settings, build log, and ten-line boot/heartbeat record. Mark hardware execution pending if you only compiled.
