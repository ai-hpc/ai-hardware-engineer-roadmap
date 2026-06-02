# Lecture 3 — Module 1: What Yocto is (and is not)

**Course:** [Yocto guide](../Guide.md) | **Phase 2 — Embedded Linux, Yocto**

**Previous:** [Lecture 02](Lecture-02.md) | **Next:** [Lecture 04 — Module 2](Lecture-04.md)

---

## 1. The problem

Embedded products need an OS that is:

- **Repeatable** (same inputs lead to same enough outputs for your release process).
- **Auditable** (licenses, versions, patches known and recorded).
- **Minimal** where possible (fewer packages, fewer CVEs and smaller update payloads).
- **Aligned to hardware** (kernel, firmware, boot chain, drivers).

## 2. What the Yocto Project provides

**Yocto Project** is an umbrella and governance model. Practically, you work with:

- **OpenEmbedded-Core (OE-Core)** — base recipes, classes, and policies.
- **BitBake** — the task scheduler / build engine.
- **Poky** — a reference integration (OE-Core + BitBake + tooling) you can build and learn from.

**Plain English:** Yocto is not a Linux distro. It is a **factory** that can produce distro-like artifacts (images, packages, SDKs) from metadata.

## 3. Yocto vs Buildroot (decision guidance)

| Dimension | Yocto / OpenEmbedded | Buildroot |
|-----------|----------------------|-----------|
| Philosophy | Layers, sharing metadata across products | Single-tree config, very direct |
| Package ecosystem | Large, community + vendor layers | Smaller default set; extensions possible |
| Learning curve | Steeper | Gentler |
| Multi-product reuse | Strong (layers, distros, configs) | Possible; often more manual |
| When vendors ship layers | Often the expected integration path | Less common (but not never) |

Neither is always better. If your **silicon vendor** maintains a mature Yocto layer, that often decides the default.

## 4. Lab 1 — Write your product requirements in Yocto terms

Answer in one page:

- **Target hardware** (SoC, machine name if known).
- **Connectivity** (Ethernet, Wi-Fi, USB gadgets, CAN, etc.).
- **Storage layout** (eMMC, NAND, NVMe, SD-only demo).
- **Update strategy** (A/B, single partition, package manager, image-based).
- **Must-have userspace** (containers or not, graphics or headless, real-time needs).

**Done when:** you can name what must be in the image vs what is nice to have.

### Worked example (optional)

A full one-page-style answer for an **edge AI camera on Jetson-class hardware** (hardware, connectivity, storage, OTA, userspace, must vs nice) with Yocto mappings and file sketches: **[Lab 01 — Worked example (edge AI camera)](Lab-01-Worked-Example-Edge-AI-Camera.md)**. Use it as a template; replace MACHINE names, layers, and packages with your BSP and release.

---

**Previous:** [Lecture 02](Lecture-02.md) | **Next:** [Lecture 04 — Module 2](Lecture-04.md)
