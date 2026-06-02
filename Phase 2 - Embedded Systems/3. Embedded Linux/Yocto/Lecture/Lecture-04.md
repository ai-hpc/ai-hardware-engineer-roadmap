# Lecture 4 — Module 2: Architecture in one picture

**Course:** [Yocto guide](../Guide.md) | **Phase 2 — Embedded Linux, Yocto**

**Previous:** [Lecture 03](Lecture-03.md) | **Next:** [Lecture 05 — Module 3](Lecture-05.md)

---

## 1. Data flow (mental model)

**Metadata** in layers (conf/, recipes-*) flows into **BitBake**, which parses recipes plus MACHINE, DISTRO, and images. That produces a **task graph**: fetch, unpack, patch, configure, compile, install, package, then image. Output artifacts include RPM/IPK/DEB packages, rootfs, kernel, bootloader, and SDK.

## 2. Key objects

- **Recipe (.bb)** — how to build one piece of software (or fetch a binary).
- **Append file (.bbappend)** — your overlay changes without editing upstream recipes.
- **Layer** — a folder of metadata + configuration, versioned as a unit.
- **Image recipe** — defines what packages land on the root filesystem for this product image.
- **MACHINE** — selects board-specific kernel/bootloader/firmware assumptions.
- **DISTRO** — policy knobs: libc choices, init system preferences, security posture defaults (varies by distro layer).

## 3. Two configs you touch constantly

- **conf/bblayers.conf** — which layers BitBake is allowed to read.
- **conf/local.conf** — local machine settings: parallelism, download directory, extra image features, temporary tweaks.

**Plain English:** bblayers.conf is which rulebooks exist. local.conf is how your laptop or build server applies them.

## 4. Lab 2 — Draw your layer stack

On paper or a whiteboard, draw **Poky/OE-Core** at the bottom, your **BSP layer** when you have one, and your **product layer** on top. Mark which layer should own kernel tweaks, rootfs packages, systemd units, and version pins.

**Done when:** you can explain where a change should live before you open an editor.

---

**Previous:** [Lecture 03](Lecture-03.md) | **Next:** [Lecture 05 — Module 3](Lecture-05.md)
