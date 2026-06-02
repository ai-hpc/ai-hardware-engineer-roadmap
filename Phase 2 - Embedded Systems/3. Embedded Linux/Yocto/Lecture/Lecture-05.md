# Lecture 5 — Module 3: First successful build

**Course:** [Yocto guide](../Guide.md) | **Phase 2 — Embedded Linux, Yocto**

**Previous:** [Lecture 04](Lecture-04.md) | **Next:** [Lecture 06 — Module 4](Lecture-06.md)

---

## 1. Get Poky (reference flow)

Exact branch names change; always prefer the **documentation for the release you chose**.

Typical sequence (conceptual): clone **Poky** at a named release branch; run the oe-init-build-env script for your build directory; set **MACHINE** to a reference machine (often QEMU for learning); build core-image-minimal first.

## 2. What success produces

You should be able to point to the **image artifacts** under the build directory deploy area (layout varies by version), and a **kernel** and **rootfs** suitable for the selected MACHINE.

## 3. Lab 3 — Minimal image, boot under QEMU (or hardware)

Build **core-image-minimal** for a QEMU-capable machine if your checkout supports it. Boot per the release docs. Log in and verify **kernel and OS identity** with standard shell commands.

**Done when:** you have repeated the build from a clean shell using only your notes (no random blog copy-paste).

---

**Previous:** [Lecture 04](Lecture-04.md) | **Next:** [Lecture 06 — Module 4](Lecture-06.md)
