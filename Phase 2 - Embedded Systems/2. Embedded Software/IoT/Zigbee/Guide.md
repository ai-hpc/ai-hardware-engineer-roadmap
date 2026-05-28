# Zigbee - Low-Power Mesh Networking for Embedded Systems

<div class="course-identity auto-course" style="--course-accent: #4f46e5; --course-accent-rgb: 79, 70, 229;" markdown="1">
<div class="course-identity__icon">ZLPM</div>
<div markdown="1">
<p class="course-identity__eyebrow">Deep Dive · Embedded Systems</p>
<p class="course-identity__title">Specialized course identity for Zigbee - Low-Power Mesh Networking for Embedded Systems.</p>
<p class="course-identity__meta">Artifact: bring-up or firmware demo · Measure: boot, latency, power, reliability</p>
</div>
</div>


A structured mini-course for engineers who want to understand **Zigbee as an embedded networking system**, not just as a consumer smart-home buzzword.

This course is placed under **Phase 2 - Embedded Software -> IoT** because Zigbee sits at the boundary between:

- MCU firmware
- low-power wireless networking
- application-layer data models
- gateway and host integration

It is the natural companion to [OpenThread](../OpenThread/Guide.md). Thread teaches low-power IPv6 mesh networking. Zigbee teaches low-power mesh networking with a stronger application-profile and cluster-model tradition.

---

## Why this course exists

Many embedded engineers learn Zigbee backwards:

- first by pairing a commercial bulb or switch
- then by reading vendor SDK examples
- only later by understanding roles, routing, clusters, bindings, and security

That order makes real systems harder to reason about.

This course fixes that by starting from the actual architecture:

- what Zigbee is built on
- how devices form and maintain a network
- how application behavior is modeled with endpoints and clusters
- how security and low-power behavior really work
- how ESP32-C6 fits into a practical Zigbee path

---

## What you will learn

- How Zigbee uses **IEEE 802.15.4** without becoming an IP network like Thread.
- What **Coordinator**, **Router**, and **End Device** roles actually mean.
- How Zigbee networking differs from simple star topologies.
- Why **ZDO**, **APS**, **ZCL**, endpoints, clusters, groups, and bindings matter.
- How Zigbee security, trust-center behavior, and join procedures fit together.
- How sleepy devices save power and what that costs in design complexity.
- How ESP32-C6 can be used for Zigbee devices, gateways, and NCP-style designs.

---

## Step-by-step lectures

Each lecture is a separate file under **[Lecture/](Lecture/README.md)**. Work in order.

| # | Topic | Lecture |
|---|-------|---------|
| 1 | What Zigbee is and where it fits | [Lecture-01.md](Lecture/Lecture-01.md) |
| 2 | Roles, topology, and network formation | [Lecture-02.md](Lecture/Lecture-02.md) |
| 3 | The Zigbee stack: ZDO, APS, endpoints, and clusters | [Lecture-03.md](Lecture/Lecture-03.md) |
| 4 | Security, commissioning, sleepy devices, and OTA | [Lecture-04.md](Lecture/Lecture-04.md) |
| 5 | ESP32-C6 practical path: devices, NCP, gateway, and Jetson context | [Lecture-05.md](Lecture/Lecture-05.md) |

---

## Recommended study pattern

For each lecture:

1. understand the network concept first
2. connect it to the embedded implementation problem
3. compare Zigbee against Thread where helpful
4. read the vendor SDK examples only after the architecture is clear

Do not memorize commands first. Build the mental model first.

---

## Official references used throughout

- [CSA Zigbee specification](https://csa-iot.org/wp-content/uploads/2023/04/05-3474-23-csg-zigbee-specification-compressed.pdf)
- [Silicon Labs Zigbee Fundamentals - Overview](https://docs.silabs.com/zigbee/8.2.1/zigbee-fundamentals/01-overview)
- [Silicon Labs Zigbee Fundamentals - Mesh Networking](https://docs.silabs.com/zigbee/8.2.1/zigbee-fundamentals/02-zigbee-mesh-networking)
- [Silicon Labs Zigbee Fundamentals - Network Node Types](https://docs.silabs.com/zigbee/8.2.0/zigbee-fundamentals/03-network-node-types)
- [Silicon Labs Zigbee Fundamentals - The Zigbee Stack](https://docs.silabs.com/zigbee/8.2.1/zigbee-fundamentals/05-the-zigbee-stack)
- [Silicon Labs Zigbee Security - Concepts](https://docs.silabs.com/zigbee/9.0.0/zigbee-security/02-concepts)
- [ESP Zigbee SDK introduction](https://docs.espressif.com/projects/esp-zigbee-sdk/en/latest/esp32c6/introduction.html)
- [ESP Zigbee NCP guide](https://docs.espressif.com/projects/esp-zigbee-sdk/en/latest/esp32c6/user-guide/ncp.html)

---

**Next:** [Lecture 01 - What Zigbee is and where it fits](Lecture/Lecture-01.md)
