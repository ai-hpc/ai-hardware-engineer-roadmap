# Lecture 10 — Module 8: SDK and application workflow

**Course:** [Yocto guide](../Guide.md) | **Phase 2 — Embedded Linux, Yocto**

**Previous:** [Lecture 09](Lecture-09.md) | **Next:** [Lecture 11 — Module 9](Lecture-11.md)

---

## 1. Why the SDK exists

The image build proves the system fits together. The **SDK** gives application developers a cross-toolchain and sysroot-matched headers/libs **without** building the whole world in one tree.

## 2. Standard pattern

- Build `meta-toolchain` or the image-specific SDK target your docs recommend.
- Install the SDK shell script output into a predictable path.
- `source` the environment script and build your app with the provided `CC`, `PKG_CONFIG_PATH`, etc.

## 3. Lab 8 — Cross-compile "hello"

Cross-compile a trivial C program against the **SDK sysroot** and run it on the **target image**.

**Done when:** you trust the SDK version string matches the image you flashed (no silent ABI skew).

---

**Previous:** [Lecture 09](Lecture-09.md) | **Next:** [Lecture 11 — Module 9](Lecture-11.md)
