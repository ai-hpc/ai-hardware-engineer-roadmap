# Lecture 4 - Peripherals, libraries, connectivity, and system design with Espressif boards

**Course:** [Espressif guide](../Guide.md) | **Phase 2 - Embedded Software**

**Previous:** [Lecture 03](Lecture-03.md) | **Next:** [Lecture 05 - Arduino as an ESP-IDF component and the path from prototype to product](Lecture-05.md)

---

## Why this lecture matters

Most people meet `arduino-esp32` through examples like:

- blink
- Wi-Fi scan
- BLE example
- I2C sensor read

Those examples are useful, but the real skill is learning to see them as parts of a complete embedded system.

This lecture is about that shift.

---

## Peripherals are not just APIs

When you use:

- `Wire`
- `SPI`
- `Serial`
- GPIO APIs

you are not just calling friendly functions. You are making decisions about:

- pin use
- timing
- interrupt behavior
- buffering
- bus ownership
- power
- library coupling

That is why Phase 2 treats these as embedded software topics, not just "maker features."

---

## Libraries are helpful, but they define architecture

Espressif's Arduino documentation includes a large set of supported APIs and libraries, but the important engineering question is not only:

> "Does a library exist?"

It is:

> "What architecture does this library push me toward?"

For example:

- a simple sensor library may be perfectly fine
- a networking library may quietly assume its own event model
- a wireless stack example may shape your tasking and memory use

So when adding libraries, ask:

- is this helping the product?
- or is it forcing accidental architecture?

---

## Connectivity is where Espressif becomes especially valuable

One reason `arduino-esp32` matters more than many classic Arduino cores is that Espressif chips are connectivity-heavy.

That means the platform is especially useful for:

- Wi-Fi products
- BLE devices
- gateway prototypes
- low-cost control nodes
- connected sensor systems

And for newer chips like `ESP32-C6`, it becomes relevant to broader wireless-system learning too.

This is why the Espressif course belongs next to the IoT course family in the roadmap.

---

## System design question: what kind of product am I building?

When working with Espressif boards, try to classify the product early.

### Type 1: quick proof-of-concept

Example:

- sensor data over Wi-Fi
- BLE remote
- simple web dashboard

For this kind of project, Arduino-first is often the right choice.

### Type 2: serious connected device

Example:

- product with OTA
- multiple peripherals
- strict power or timing constraints
- long-lived field deployment

For this kind of project, you need to start thinking beyond "sketches" even if Arduino remains the top-level layer.

### Type 3: advanced vendor-platform firmware

Example:

- heavy wireless integration
- advanced control over system internals
- product with feature layering and maintainability constraints

This is where Arduino as an ESP-IDF component or a partial migration becomes much more attractive.

---

## Good embedded habits inside Arduino projects

Even in Arduino-style projects, try to keep these habits:

- separate hardware access from application logic
- avoid giant `loop()` functions that do everything
- document pin assignments and board assumptions
- isolate connectivity code from sensor code
- treat libraries as dependencies, not magic
- think about failure cases, not only happy-path demos

These habits make migration much easier later.

---

## The system-level lesson

The real value of `arduino-esp32` is not just that it makes hardware easier.

It lets you prototype complete connected systems quickly, while still teaching you where the deeper embedded boundaries are:

- board support
- peripheral use
- wireless stacks
- runtime behavior
- framework layering

That is why the platform is worth studying seriously.

---

## Lab

Take one example project idea, such as:

- Wi-Fi sensor node
- BLE remote
- ESP32-C6 connected switch

Then divide the firmware into four blocks:

- board/peripheral layer
- connectivity layer
- application logic
- update/debug/support layer

Write one sentence for what each block should own.

If you can separate those cleanly, you are already thinking beyond copy-paste sketches.

---

**Previous:** [Lecture 03](Lecture-03.md) | **Next:** [Lecture 05 - Arduino as an ESP-IDF component and the path from prototype to product](Lecture-05.md)
