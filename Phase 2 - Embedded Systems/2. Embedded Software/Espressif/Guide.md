# Espressif and Arduino-ESP32

A structured mini-course for engineers who want to understand **Espressif's Arduino platform as real embedded software**, not just as hobby-board sketching.

This course is placed under **Phase 2 - Embedded Software** because `arduino-esp32` sits at an important boundary:

- beginner-friendly application code
- production-capable MCU SoCs
- FreeRTOS and ESP-IDF under the hood
- practical Wi-Fi, BLE, and peripheral bring-up

It is the natural companion to the main **ARM MCU / FreeRTOS / buses** material and to the **IoT** subtrack.

---

## Why this course exists

Many people learn `arduino-esp32` in the wrong order:

- first by copying sketches from random blogs
- then by adding Wi-Fi or BLE examples
- only later by realizing the platform is built on top of **ESP-IDF**, **FreeRTOS**, board variants, and vendor libraries

That makes it hard to answer practical engineering questions like:

- when is Arduino enough?
- when should I move to ESP-IDF?
- how much of the Espressif stack is hidden under the Arduino API?
- which ESP32 chips are really supported?
- how do I go from a demo sketch to a product codebase?

This course fixes that by treating `arduino-esp32` as a real embedded platform.

---

## What you will learn

- What the `espressif/arduino-esp32` project actually is.
- Which ESP32-family chips are supported and how support differs.
- How Arduino on Espressif sits on top of **ESP-IDF** and **FreeRTOS**.
- How board variants, libraries, peripherals, and connectivity fit together.
- When to use the standard Arduino workflow and when to use **Arduino as an ESP-IDF component**.
- How to think about migration from prototype code to production firmware.

---

## Step-by-step lectures

Each lecture is a separate file under **[Lecture/](Lecture/README.md)**. Work in order.

| # | Topic | Lecture |
|---|-------|---------|
| 1 | What `arduino-esp32` is and where it fits | [Lecture-01.md](Lecture/Lecture-01.md) |
| 2 | Supported chips, install path, and first board bring-up | [Lecture-02.md](Lecture/Lecture-02.md) |
| 3 | What is really under the Arduino layer: ESP-IDF, FreeRTOS, and core architecture | [Lecture-03.md](Lecture/Lecture-03.md) |
| 4 | Peripherals, libraries, connectivity, and system design with Espressif boards | [Lecture-04.md](Lecture/Lecture-04.md) |
| 5 | Arduino as an ESP-IDF component and the path from prototype to product | [Lecture-05.md](Lecture/Lecture-05.md) |

---

## Recommended study pattern

For each lecture:

1. understand the platform boundary first
2. connect Arduino concepts to real embedded-system layers
3. keep asking what is handled by Arduino and what is handled by ESP-IDF
4. treat the sketch as the surface, not the whole system

Do not stop at "it compiled." Understand the stack below it.

---

## Official references used throughout

- [Arduino core for the ESP32 family of SoCs - README](https://github.com/espressif/arduino-esp32)
- [Arduino-ESP32 online documentation](https://docs.espressif.com/projects/arduino-esp32/en/latest/)
- [Getting Started](https://docs.espressif.com/projects/arduino-esp32/en/latest/getting_started.html)
- [Installing](https://docs.espressif.com/projects/arduino-esp32/en/latest/installing.html)
- [Libraries](https://docs.espressif.com/projects/arduino-esp32/en/latest/libraries.html)
- [Arduino as an ESP-IDF component](https://docs.espressif.com/projects/arduino-esp32/en/latest/esp-idf_component.html)
- [Migration guide 2.x to 3.0](https://docs.espressif.com/projects/arduino-esp32/en/latest/migration_guides/2.x_to_3.0.html)

---

**Next:** [Lecture 01 - What `arduino-esp32` is and where it fits](Lecture/Lecture-01.md)
