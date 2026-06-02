# Lecture 8 — Module 6: Images, packages, and features

**Course:** [Yocto guide](../Guide.md) | **Phase 2 — Embedded Linux, Yocto**

**Previous:** [Lecture 07](Lecture-07.md) | **Next:** [Lecture 09 — Module 7](Lecture-09.md)

---

## 1. Image recipes vs packagegroups

- **Image recipe** — names the packages that become the rootfs for a flashable artifact.
- **packagegroup recipes** — bundles related packages for reuse across images.

## 2. IMAGE_INSTALL and friends

Teams usually extend images with:

- **IMAGE_INSTALL:append** — additive, review-friendly in many setups.

**Avoid editing Poky images directly**; extend via bbappend or a custom image recipe in your layer.

## 3. Image features (conceptual)

Features like debugging helpers, tools, or init tweaks are often gated as **image features** (exact names vary by distro/release). Treat them as **policy switches**, not random variables.

## 4. Lab 6 — Custom image

Create recipes-core/images/mycourse-image.bb that:

- requires or inherits appropriately from a minimal base image for your release.
- Adds two packages you need (for example, strace and your favorite tiny shell utility).

**Done when:** bitbake mycourse-image produces a new deployable artifact and you can identify its size difference vs minimal.

---

**Previous:** [Lecture 07](Lecture-07.md) | **Next:** [Lecture 09 — Module 7](Lecture-09.md)
