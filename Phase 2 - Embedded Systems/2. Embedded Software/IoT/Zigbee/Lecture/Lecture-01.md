# Lecture 1 - What Zigbee is and where it fits

**Course:** [Zigbee guide](../Guide.md) | **Phase 2 - Embedded Software, IoT**

**Next:** [Lecture 02 - Roles, topology, and network formation](Lecture-02.md)

---

## The shortest correct definition

Zigbee is a **low-power wireless networking stack for embedded devices** built on top of **IEEE 802.15.4**.

That sentence matters because it prevents two common mistakes:

- Zigbee is **not** just "802.15.4"
- Zigbee is **not** an IP network like Thread

IEEE 802.15.4 gives Zigbee the radio and MAC foundation. Zigbee adds its own:

- network behavior
- routing model
- device roles
- security model
- application-layer object model

Official references: [CSA Zigbee specification](https://csa-iot.org/wp-content/uploads/2023/04/05-3474-23-csg-zigbee-specification-compressed.pdf), [Silicon Labs Zigbee overview](https://docs.silabs.com/zigbee/8.2.1/zigbee-fundamentals/01-overview)

---

## Why Zigbee exists

Zigbee was designed for systems that need:

- low power
- modest data rates
- many devices
- good enough reliability in noisy real-world radio environments
- device-to-device control and sensing

This makes it a good fit for:

- home automation
- building automation
- sensors
- lighting
- metering
- industrial monitoring

Zigbee is a bad fit if your first requirement is:

- high bandwidth
- video
- large file transfer
- direct general-purpose IP networking

That is why it sits closer to **control and telemetry** than to Wi-Fi-style networking.

---

## Zigbee vs Thread

You already have OpenThread in this IoT path, so the most useful comparison is direct.

### What they share

- both commonly use **IEEE 802.15.4**
- both target low-power embedded networks
- both support mesh-like behavior
- both are common in smart-home and gateway products

### What they do differently

- **Thread** is an IPv6-first constrained IP network
- **Zigbee** is a non-IP networking stack with its own application model

That difference changes the **whole engineering style**.

With Thread, you think in terms of:

- IPv6
- 6LoWPAN
- UDP
- border routers

With Zigbee, you think in terms of:

- endpoints
- clusters
- attributes
- bindings
- trust center and network keys

So even though the radios may look similar on a schematic, the **software architecture is not the same**.

---

## Why Zigbee belongs in Embedded Software

Zigbee is not just a networking topic. It is deeply **embedded-software-shaped**.

You need to reason about:

- device roles and power states
- persistent network credentials
- event-driven application logic
- endpoint and cluster configuration
- security material in flash or secure storage
- application behavior on tiny MCUs

This is exactly the kind of system where firmware, networking, and product behavior are **tightly coupled**.

---

## The mental model to keep

Think of Zigbee as:

- a **low-power mesh-capable network**
- with a **strong application model**
- built for **device control and sensing**
- on top of **802.15.4**

That is the correct starting point for everything else in this course.

---

## Lab

Write a short comparison note with two columns:

- "What Zigbee inherits from 802.15.4"
- "What Zigbee adds above 802.15.4"

Then add one more column:

- "What Thread does differently"

If you can explain that clearly, you are ready for the next lecture.

---

**Previous:** [Course hub](../Guide.md) | **Next:** [Lecture 02 - Roles, topology, and network formation](Lecture-02.md)
