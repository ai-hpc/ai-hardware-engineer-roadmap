# IoT Networking and Device Connectivity

<div class="course-identity auto-course" style="--course-accent: #0284c7; --course-accent-rgb: 2, 132, 199;" markdown="1">
<div class="course-identity__icon">INAD</div>
<div markdown="1">
<p class="course-identity__eyebrow">Deep Dive · Embedded Systems</p>
<p class="course-identity__title">Specialized course identity for IoT Networking and Device Connectivity.</p>
<p class="course-identity__meta">Artifact: bring-up or firmware demo · Measure: boot, latency, power, reliability</p>
</div>
</div>


*Builds on [**Embedded Software**](../Guide.md) once you are comfortable with MCU buses, interrupts, and RTOS basics. This sub-layer shifts from board-local communication like SPI/UART/I2C/CAN to **networked embedded systems** that must join, secure, and maintain real deployments.*

---

## Why This Sub-Layer Exists

Point-to-point buses teach you how one MCU talks to one peripheral. IoT protocols teach you how many devices form a network, recover from node loss, conserve power, and still present a clean IP-facing software model to gateways and cloud services.

This matters because modern embedded products rarely stop at "sensor attached to MCU." A production device usually needs secure onboarding, field updates, a mesh or star network, and a way to bridge low-power radios to Linux hosts, mobile apps, or cloud APIs.

---

## What You Study Here

### OpenThread

OpenThread is the best first protocol in this sub-layer because it forces you to connect **real embedded constraints** with **real networking ideas**. You need to understand IEEE 802.15.4 radios, low-power scheduling, IPv6, packet compression, host/MCU splits, and border routers all at once.

It also creates a direct bridge to later roadmap work:

* **MCU / RTOS path:** run the Thread stack directly on an SoC and reason about tasks, timers, radio drivers, and power states.
* **Linux / host path:** use a radio co-processor (RCP) over UART or SPI and let a Linux host run the higher Thread stack and border-router logic.

Start here:

* [**OpenThread**](OpenThread/Guide.md)

### Zigbee

Zigbee is the next useful protocol after OpenThread because it teaches a different embedded networking philosophy on top of the same low-power radio family. Instead of being IP-first, Zigbee is much more **device-model-first**: endpoints, clusters, bindings, trust-center behavior, and role selection shape the firmware architecture.

It also creates a direct bridge to later roadmap work:

* **MCU / RTOS path:** build coordinator, router, or end-device firmware directly on an SoC such as ESP32-C6.
* **Linux / host path:** use a Zigbee Network Co-Processor (NCP) or gateway-style design and let a stronger host manage control logic.

Then study:

* [**Zigbee**](Zigbee/Guide.md)

---

## Why IoT Protocols Matter for AI Hardware Engineers

AI systems at the edge are rarely isolated. They live inside products with commissioning flows, low-power sensor networks, battery constraints, gateways, and secure remote management.

If you can reason about Thread, RCP/NCP splits, border routers, and low-power IPv6 networking, you are much better prepared to build systems where MCUs, Linux hosts, accelerators, and cloud services all cooperate instead of existing as disconnected demos.
