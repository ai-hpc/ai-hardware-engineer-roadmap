# Lecture 3 - Getting started with ESP32: chips, boards, flashing, and first examples

**Course:** [Official education path guide](../Guide.md) | **Phase 2 - Embedded Software**

**Previous:** [Lecture 02](Lecture-02.md) | **Next:** [Lecture 04 - Solution tracks: AI, connectivity, peripherals, low power, and gateways](Lecture-04.md)

---

## Why this lecture matters

After basics, Espressif’s official education path moves into **Getting Started with ESP32**.

That is where learners need to answer:

- what is the difference between a chip, module, and development board?
- which development environment should I use?
- what are the first examples I should really run?

---

## Chip vs module vs development board

This distinction is important and often skipped.

### Chip

The chip is the actual SoC.

Example:

- `ESP32-C6`
- `ESP32-S3`

### Module

The module packages the chip with things like:

- flash
- RF design
- certification support

This is what many real products use.

### Development board

The development board gives you:

- USB
- easy headers
- power support
- a fast start

If you blur these three together, you make poor hardware decisions later.

Espressif’s education page explicitly points learners to understanding these differences first.

---

## The first hardware concepts that matter

Espressif’s study plan also calls out:

- strapping pins
- run mode
- download mode

That matters because "board does not flash" is often not a software problem.

It is often:

- wrong boot mode
- wrong USB/serial assumptions
- wrong board setup

So even first bring-up is still embedded engineering.

---

## Choosing the development environment

Espressif’s official education page presents three useful levels:

- **UIFlow**
  - very low-code
  - not the main route for this roadmap
- **Arduino**
  - good for entry-level and fast practical projects
- **ESP-IDF**
  - good for more complex projects and flexible development

That is a healthy official message because it avoids pretending one environment is right for every learner and every product.

For this roadmap:

- use **Arduino** for fast hardware and connected-device starts
- use **ESP-IDF** when project structure and control start to matter more

---

## The first examples that actually matter

Espressif’s education page points learners toward the kinds of examples that make sense:

- `hello_world`
- `blink`
- Wi-Fi station
- peripheral examples

That progression is correct because it moves from:

- compile and flash
- to GPIO
- to connectivity
- to peripherals

This is much better than starting with a giant showcase demo.

---

## Why the build system matters early

The education page also points learners to:

- partition table
- `CMakeLists`
- `sdkconfig`
- `Kconfig`
- component manager

That is a major hint from Espressif:

> serious ESP32 work quickly becomes build-system-aware work

So even when you begin with Arduino, you should know that the deeper project model exists.

---

## Lab

Write a short plan with:

1. one board you would start with
2. one framework you would start with
3. three examples you would run in order

Then explain why that order is better than jumping straight into a complex demo.

---

**Previous:** [Lecture 02](Lecture-02.md) | **Next:** [Lecture 04 - Solution tracks: AI, connectivity, peripherals, low power, and gateways](Lecture-04.md)
