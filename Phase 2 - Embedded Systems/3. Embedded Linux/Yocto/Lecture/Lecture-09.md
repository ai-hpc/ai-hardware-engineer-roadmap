# Lecture 9 — Module 7: Kernel, bootloader, device tree (integrator level)

**Course:** [Yocto guide](../Guide.md) | **Phase 2 — Embedded Linux, Yocto**

**Previous:** [Lecture 08](Lecture-08.md) | **Next:** [Lecture 10 — Module 8](Lecture-10.md)

---

## 1. What Yocto expects you to know

At this stage you are **not required to be a kernel maintainer**. You are responsible for:

- Selecting the right **kernel provider** / recipe for your BSP.
- Carrying **board device trees** and any **kernel config fragments** your hardware needs.
- Understanding **where** boot artifacts come from in the deploy directory.

## 2. Typical customization paths

- **Config fragments** — preferred for maintainable kernel option changes when supported.
- **Patches** — for drivers, DTS fixes, or backports (with review and upstreaming plan).
- **Out-of-tree modules** — sometimes the right boundary for proprietary or fast-moving drivers.

## 3. Lab 7 — Trace boot artifacts

For your MACHINE, list which tasks produce:

- Kernel image / device tree blobs (if applicable).
- Bootloader binary(ies).

**Done when:** you can open the deploy directory and point to each file the flashing script would use.

---

**Previous:** [Lecture 08](Lecture-08.md) | **Next:** [Lecture 10 — Module 8](Lecture-10.md)
