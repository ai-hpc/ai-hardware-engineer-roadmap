# Lecture 4 - Security, commissioning, sleepy devices, and OTA

**Course:** [Zigbee guide](../Guide.md) | **Phase 2 - Embedded Software, IoT**

**Previous:** [Lecture 03 - The Zigbee stack: ZDO, APS, endpoints, and clusters](Lecture-03.md) | **Next:** [Lecture 05 - ESP32-C6 practical path: devices, NCP, gateway, and Jetson context](Lecture-05.md)

---

## Security is not optional

Zigbee is often deployed in products that control:

- lighting
- locks
- sensors
- building devices

So the **security model** is not decorative. It is **core system behavior**.

Official reference: [Silicon Labs Zigbee security concepts](https://docs.silabs.com/zigbee/9.0.0/zigbee-security/02-concepts)

At a high level, Zigbee security includes:

- network-level security
- key handling
- trust-center responsibilities
- secure admission and device authorization

---

## Network key and trust-center thinking

You do not need every detail of the spec on day one, but you do need the right mental model:

- a Zigbee network has shared security material
- joining is controlled
- the trust center is central to admission and security policy

That means **secure onboarding** is part of firmware and gateway design, not an afterthought.

---

## Commissioning

Commissioning is the process of:

- allowing a device to join
- giving it the right credentials and policies
- integrating it into the network

If you come from Thread, think of this as the Zigbee-side version of secure onboarding, but with its own trust-center and application-ecosystem conventions.

The exact user experience may differ across products, but the engineering principle is the same:

- a device should not simply appear on the network because it is nearby

---

## Sleepy end devices

One of Zigbee's big value points is **battery-powered operation**.

Sleepy end devices:

- turn radios off aggressively
- do not route traffic
- depend on a parent/router relationship
- save energy by trading latency and simplicity

This is similar in spirit to sleepy children in Thread, even though the stacks are different.

The key embedded consequence is that **low power is never free**. You pay for it with:

- parent dependence
- polling behavior
- longer latency for inbound traffic
- more careful state handling

---

## OTA updates

**OTA** is especially important for IoT products because deployed devices do not stay static.

For Zigbee products, OTA matters because you may need to:

- fix bugs
- patch security issues
- update cluster behavior
- maintain compatibility with ecosystem changes

So even if OTA looks like an application feature, it is also part of long-term product maintainability.

---

## The practical design lesson

When you design a Zigbee product, you are really choosing among several competing goals:

- battery life
- responsiveness
- routing resilience
- join simplicity
- secure onboarding
- field maintenance

That is why Zigbee work is **real embedded-systems engineering** and not just wireless configuration.

---

## Lab

Take one hypothetical product:

- battery sensor
- smart plug
- room controller

For that product, write:

- should it be sleepy or always-on?
- should it ever be a router?
- what is the security risk if onboarding is weak?
- why would OTA matter after deployment?

---

**Previous:** [Lecture 03 - The Zigbee stack: ZDO, APS, endpoints, and clusters](Lecture-03.md) | **Next:** [Lecture 05 - ESP32-C6 practical path: devices, NCP, gateway, and Jetson context](Lecture-05.md)
