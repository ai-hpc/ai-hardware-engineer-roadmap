# Lecture 5 - ESP32-C6 practical path: devices, NCP, gateway, and Jetson context

**Course:** [Zigbee guide](../Guide.md) | **Phase 2 - Embedded Software, IoT**

**Previous:** [Lecture 04 - Security, commissioning, sleepy devices, and OTA](Lecture-04.md)

---

## Why ESP32-C6 matters here

Espressif's official Zigbee material makes ESP32-C6 relevant because it can be used to build:

- Zigbee devices
- Zigbee coordinators and routers
- gateway-style host/co-processor systems

Official reference: [ESP Zigbee SDK introduction](https://docs.espressif.com/projects/esp-zigbee-sdk/en/latest/esp32c6/introduction.html)

That makes ESP32-C6 a useful bridge between:

- pure MCU Zigbee devices
- Linux-hosted gateway experiments

---

## Three practical implementation styles

### 1. Standalone Zigbee device

In this model, the ESP32-C6 runs the **whole Zigbee application locally**.

Use this for:

- sensors
- switches
- simple controllers

This is the **cleanest way** to learn endpoint, cluster, and role configuration.

### 2. Network Co-Processor (NCP)

In the Zigbee NCP model:

- the Zigbee stack lives on the coprocessor
- a host processor controls it over a host interface
- the protocol is **ESP ZNSP over SLIP**
- the transport can be **UART** or **SPI**

Official references: [ESP Zigbee NCP guide](https://docs.espressif.com/projects/esp-zigbee-sdk/en/latest/esp32c6/user-guide/ncp.html), [ESP Zigbee NCP API](https://docs.espressif.com/projects/esp-zigbee-sdk/en/latest/esp32c6/api-reference/esp_zigbee_ncp.html)

This is the **most relevant model** for a Linux host such as Jetson.

### 3. Gateway / multi-chip design

Espressif also documents a `zigbee_gateway` example that uses:

- a host SoC
- an 802.15.4 radio side

That is useful because it shows real Zigbee gateway designs are often **multi-processor systems**, not single-chip toys.

Official reference: [ESP Zigbee gateway example](https://github.com/espressif/esp-zigbee-sdk/tree/main/examples/zigbee_gateway)

---

## What changes on Jetson

For Thread on Jetson, the model was:

- host stack on Linux
- `otbr-agent` or `ot-daemon`
- `wpan0`

For Zigbee on Jetson, the model is different:

- no `wpan0`
- no OTBR-style IPv6 border-router path
- the host speaks Zigbee control messages to the coprocessor

That means success is measured by:

- forming or joining a Zigbee network
- reading state from the coprocessor
- sending cluster or management commands

not by seeing a Linux IP interface appear.

---

## Why this matters for your current roadmap path

This roadmap now contains all three related ideas:

- [OpenThread](../../OpenThread/Guide.md) for low-power IP mesh networking
- [ESP32-C6 OpenThread RCP on Jetson Orin Nano](../../../../../Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/5.%20Application%20Development/2.%20Network%20and%20Connectivity/ESP32-C6-OpenThread-RCP-Jetson-Orin-Nano.md) for Linux-hosted Thread with a radio coprocessor
- [ESP32-C6 Zigbee NCP on Jetson Orin Nano](../../../../../Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/5.%20Application%20Development/2.%20Network%20and%20Connectivity/ESP32-C6-Zigbee-NCP-Jetson-Orin-Nano.md) for a Zigbee coprocessor path

That comparison is valuable because it shows the **same radio family can support very different software architectures**.

---

## The right takeaway from this whole course

Zigbee should now look like:

- a low-power embedded network
- with strong device-behavior modeling
- where node role, security, and application structure are tightly coupled
- and where host/NCP designs are a real option, not just single-chip examples

If you can compare Zigbee against Thread and explain why one is IP-native while the other is cluster-and-endpoint-centric, then this course has done its job.

---

## Lab

Design two architectures for the same smart-building product:

### Architecture A

- ESP32-C6 as a standalone Zigbee device

### Architecture B

- Jetson as host
- ESP32-C6 as Zigbee NCP

For each, answer:

- where does application logic live?
- where do network credentials live?
- which side handles user-visible device behavior?
- what becomes easier?
- what becomes harder?

---

**Previous:** [Lecture 04 - Security, commissioning, sleepy devices, and OTA](Lecture-04.md) | **Next:** [Course hub](../Guide.md)
