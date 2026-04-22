# Lecture 2 - Embedded development basics for the ESP32 path

**Course:** [Official education path guide](../Guide.md) | **Phase 2 - Embedded Software**

**Previous:** [Lecture 01](Lecture-01.md) | **Next:** [Lecture 03 - Getting started with ESP32: chips, boards, flashing, and first examples](Lecture-03.md)

---

## Why this lecture matters

On Espressif’s official education page, the study plan starts with **Embedded Development Basics**.

That is the correct choice.

Before talking about:

- Wi-Fi
- BLE
- Matter
- Zigbee
- voice

you still need the embedded foundation underneath.

---

## What the official study plan emphasizes

Espressif’s study-plan section includes basics like:

- voltage, current, resistance, capacitance
- common components
- C language
- functions, pointers, memory
- project compilation and linking
- GPIO, timer, UART, SPI, I2C
- TCP and UDP
- Git
- FreeRTOS
- Linux instructions

That list is a strong reminder:

> ESP32 development is still embedded systems, not just “wireless app code.”

---

## How to map those basics into this roadmap

You do not need to relearn everything here from scratch if you already worked through Phase 1 and the main Embedded Software module.

Instead, use the official Espressif list as a checklist:

### Already covered by our roadmap

- C and system programming foundations
- MCU and peripheral thinking
- FreeRTOS
- buses like UART, SPI, I2C
- basic networking concepts

### What Espressif adds

- a concrete platform where all those pieces meet
- real wireless SoCs
- practical cloud and protocol solution paths
- product-oriented kits and examples

---

## The important mindset

When you see an ESP32 example, ask:

- which bus is really being used?
- what memory assumptions are hidden here?
- is this blocking or event-driven?
- what part is board-specific?
- where does FreeRTOS show up implicitly?

That is how you turn examples into engineering understanding.

---

## What not to do

Do not use Espressif’s higher-level examples to hide weak fundamentals.

If you do not understand:

- UART
- I2C
- GPIO
- interrupts
- memory

then advanced ESP32 examples will still feel like magic.

And magic is fragile.

---

## The real lesson

The official education path starts with basics because connected products fail on basic mistakes all the time:

- wrong pins
- wrong voltage assumptions
- poor timing
- incorrect tasking
- weak communication handling

That is why this lecture belongs early.

---

## Lab

Write a checklist with these groups:

- C / memory
- buses / peripherals
- networking basics
- RTOS / project tools

For each group, write:

- what I already know
- what I still need to strengthen before complex ESP32 projects

---

**Previous:** [Lecture 01](Lecture-01.md) | **Next:** [Lecture 03 - Getting started with ESP32: chips, boards, flashing, and first examples](Lecture-03.md)
