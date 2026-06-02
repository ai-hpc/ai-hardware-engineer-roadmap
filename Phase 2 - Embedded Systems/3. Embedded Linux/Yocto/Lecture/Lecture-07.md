# Lecture 7 — Module 5: Layers: how metadata stays maintainable

**Course:** [Yocto guide](../Guide.md) | **Phase 2 — Embedded Linux, Yocto**

**Previous:** [Lecture 06](Lecture-06.md) | **Next:** [Lecture 08 — Module 6](Lecture-08.md)

---

## 1. Why layers exist

Layers let teams **separate concerns**: BSP with hardware knowledge, corporate distro policy across products, product-specific tweaks isolated and reviewable.

## 2. Creating a custom layer (pattern)

Use the official **bitbake-layers create-layer** workflow for your release. You should end with conf/layer.conf, recipes-* trees, and optional README for layer intent and maintainers.

## 3. bbappend discipline

If you did not write the upstream recipe, prefer **.bbappend** in your layer. Good: add a patch, tweak PACKAGECONFIG, add a runtime dependency, install a config via FILESPATH patterns. Bad: **fork a recipe** to change one variable you could append.

## 4. Lab 5 — Your own layer, minimal change

Create meta-mycourse, add it to bblayers.conf, append a trivial change to a recipe (harmless PACKAGECONFIG toggle or FILES tweak).

**Done when:** bitbake-layers show-layers lists your layer in the expected priority order and the build still succeeds.

---

**Previous:** [Lecture 06](Lecture-06.md) | **Next:** [Lecture 08 — Module 6](Lecture-08.md)
