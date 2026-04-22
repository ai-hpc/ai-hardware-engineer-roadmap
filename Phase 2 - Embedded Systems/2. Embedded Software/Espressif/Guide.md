# Espressif

A structured embedded-software track for engineers who want to learn **Espressif platforms seriously**, not just as isolated board demos.

This track sits under **Phase 2 - Embedded Software** because Espressif development naturally combines:

- MCU firmware
- peripherals and board bring-up
- connectivity
- FreeRTOS
- ESP-IDF
- rapid prototyping through Arduino

It is the natural companion to the main **ARM MCU / FreeRTOS / buses** material and to the **IoT** subtrack.

---

## Why this track exists

Many people learn Espressif platforms in the wrong order:

- first by copying Arduino sketches
- then by adding Wi-Fi or BLE examples
- only later by realizing the platform includes:
  - different ESP32-family SoCs
  - ESP-IDF
  - FreeRTOS
  - board variants
  - connectivity stacks
  - solution frameworks like RainMaker, ESP-SR, and OpenThread

That makes it hard to answer practical engineering questions like:

- where should I start in the Espressif ecosystem?
- when is Arduino enough?
- when should I move to ESP-IDF?
- how do the official education materials fit together?
- how do I go from starter examples to a product direction?

This track fixes that by splitting Espressif learning into two connected paths.

---

## What you will learn

- How `arduino-esp32` fits into the Espressif stack.
- How Espressif’s official education path organizes learning.
- How to choose between Arduino, ESP-IDF, and more advanced solution paths.
- How to connect boards, frameworks, peripherals, and connectivity into one learning plan.
- How to move from first board bring-up to product-shaped firmware work.

---

## Courses in this track

### 1. Arduino-ESP32 foundation

Use this first if your main entry point is `espressif/arduino-esp32`.

- [Course guide](Lecture/README.md)
- focus: supported chips, board bring-up, Arduino layering, ESP-IDF component path

### 2. Official education path

Use this to follow Espressif’s broader official education direction.

- [Course guide](Education/Guide.md)
- focus: study plan, development environments, starter examples, solution tracks, and practical course directions

---

## Recommended study pattern

Recommended order:

1. start with the **Arduino-ESP32 foundation** if you need a fast and concrete entry
2. work through the **official education path** to widen your view beyond sketches
3. branch toward:
   - [IoT](../IoT/Guide.md) for OpenThread and Zigbee
   - later Phase 4 tracks for Jetson, FPGA, or deeper AI deployment

Do not stop at "it compiled." Understand the system around it.

---

## Official references used throughout

- [Espressif Education](https://www.espressif.com/en/ecosystem/education)
- [Arduino core for the ESP32 family of SoCs - README](https://github.com/espressif/arduino-esp32)
- [Arduino-ESP32 online documentation](https://docs.espressif.com/projects/arduino-esp32/en/latest/)
- [Getting Started](https://docs.espressif.com/projects/arduino-esp32/en/latest/getting_started.html)
- [Installing](https://docs.espressif.com/projects/arduino-esp32/en/latest/installing.html)
- [Libraries](https://docs.espressif.com/projects/arduino-esp32/en/latest/libraries.html)
- [Arduino as an ESP-IDF component](https://docs.espressif.com/projects/arduino-esp32/en/latest/esp-idf_component.html)
- [Migration guide 2.x to 3.0](https://docs.espressif.com/projects/arduino-esp32/en/latest/migration_guides/2.x_to_3.0.html)

---

**Next:** [Arduino-ESP32 lectures](Lecture/README.md) · [Official education path](Education/Guide.md)
