# Lecture 2 - Supported chips, install path, and first board bring-up

**Course:** [Espressif guide](../Guide.md) | **Phase 2 - Embedded Software**

**Previous:** [Lecture 01](Lecture-01.md) | **Next:** [Lecture 03 - What is really under the Arduino layer: ESP-IDF, FreeRTOS, and core architecture](Lecture-03.md)

---

## Why this lecture matters

Before writing code, you need to know **what platform you are actually standing on**.

For `arduino-esp32`, that means three things:

1. which ESP32 chips are supported
2. which workflow you are using
3. whether your first hardware bring-up is supposed to be simple or advanced

If you skip those questions, you end up copying examples for the wrong chip or the wrong build model.

---

## Supported chips

According to Espressif's official `arduino-esp32` README, the stable core supports these ESP32-family SoCs:

- `ESP32`
- `ESP32-C3`
- `ESP32-C5`
- `ESP32-C6`
- `ESP32-H2`
- `ESP32-P4`
- `ESP32-S2`
- `ESP32-S3`

Important note from Espressif:

- `ESP32-C2` and `ESP32-C61` are supported differently
- they require either:
  - **Arduino as an ESP-IDF component**
  - or rebuilding the static libraries

That distinction matters because **not every chip supports the simplest Arduino flow equally**.

Official references: [README supported chips](https://github.com/espressif/arduino-esp32), [Getting Started](https://docs.espressif.com/projects/arduino-esp32/en/latest/getting_started.html), [Libraries](https://docs.espressif.com/projects/arduino-esp32/en/latest/libraries.html)

---

## The two main workflow styles

### 1. Classic Arduino-style workflow

This is the path most people imagine first:

- install the board package
- select the board
- write a sketch
- upload

This is the best path for:

- first board bring-up
- GPIO, UART, Wi-Fi, BLE, and sensor experiments
- fast iteration

### 2. Arduino as an ESP-IDF component

Espressif documents this as the more advanced path.

This is best when you need:

- tighter control over project structure
- direct ESP-IDF integration
- advanced peripherals or middleware
- more production-like firmware organization

This distinction is one of the **most important practical lessons** in the Espressif ecosystem.

---

## What “first board bring-up” should look like

Your first success should be **deliberately boring**.

Do not start with:

- Matter
- Thread
- custom partitions
- complex multitasking
- cloud stacks

Start with:

1. board package installed correctly
2. correct board selected
3. serial upload works
4. serial monitor works
5. a minimal LED or serial example runs

That tells you:

- USB path is correct
- flashing works
- your chosen board profile is sane
- the development environment is not the blocker

---

## What to verify during bring-up

After the first upload, verify these basics:

- boot messages appear on serial
- your code reaches `setup()`
- your loop runs repeatedly
- a GPIO output or serial print behaves as expected
- board-specific features like LED pin or USB CDC match the selected board profile

This sounds trivial, but it prevents **many hours of confusion** later.

---

## Why chip identity matters more on Espressif than beginners expect

A lot of Arduino tutorials blur boards together.

That is dangerous on Espressif because:

- different chips have different radio combinations
- peripheral availability differs
- memory and flash assumptions differ
- low-power behavior differs
- some newer chips have different support maturity

So you should think in this order:

- which **SoC**?
- which **board variant**?
- which **workflow model**?

not just:

- which example sketch?

---

## Best current starting point for this roadmap

For this roadmap, a very strong learning path is:

- `ESP32-C3` or `ESP32-S3` for broad community examples
- `ESP32-C6` when you want Wi-Fi + BLE + 802.15.4 relevance

That makes `arduino-esp32` especially useful before:

- OpenThread
- Zigbee
- connected sensor products
- voice peripherals and local gateways

---

## Lab

Make a short table with these columns:

- `Chip`
- `Why I might choose it`
- `Would I start with Arduino workflow or ESP-IDF component workflow?`

Fill in at least:

- `ESP32-C3`
- `ESP32-C6`
- `ESP32-S3`

Then write one sentence about why `ESP32-C2` / `ESP32-C61` need extra care.

---

**Previous:** [Lecture 01](Lecture-01.md) | **Next:** [Lecture 03 - What is really under the Arduino layer: ESP-IDF, FreeRTOS, and core architecture](Lecture-03.md)
