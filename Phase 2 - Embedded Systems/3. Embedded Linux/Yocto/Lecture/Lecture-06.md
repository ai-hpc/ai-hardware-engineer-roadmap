# Lecture 6 — Module 4: Recipes: the atoms of the system

**Course:** [Yocto guide](../Guide.md) | **Phase 2 — Embedded Linux, Yocto**

**Previous:** [Lecture 05](Lecture-05.md) | **Next:** [Lecture 07 — Module 5](Lecture-07.md)

---

## 1. Anatomy of a recipe

A **recipe** answers predictable questions: where **sources** come from (Git, tarball, license file); how it is configured and compiled (autotools, CMake, Meson, Makefile, prebuilt); what files are installed into staged roots; how it is split into **runtime packages** (dev, dbg, etc.).

## 2. Inheritance and classes

Instead of copying boilerplate, recipes **inherit** behavior: autotools, cmake, meson, kernel, image, and related classes. Classes encode how this kind of software is usually built.

## 3. Variables you will see constantly (conceptual)

Names evolve slightly, but the ideas do not: **SRC_URI** (what to download; patches often listed); **S** (unpacked sources); **B** (out-of-tree build dir); **WORKDIR** (per-recipe scratch); **PN, PV, PR** (name/version/revision).

## 4. Lab 4 — Inspect a real recipe

Pick a small recipe in **OE-Core**. Trace which **class** it inherits, which tasks exist (do_compile, do_install, etc.), and what package names it produces.

Use bitbake -e with your recipe name for effective environment (large output: learn to search it). Optionally bitbake -c menuconfig linux-yocto when you are ready for kernel exploration.

**Done when:** you can name the tasks between source and package for that recipe.

---

**Previous:** [Lecture 05](Lecture-05.md) | **Next:** [Lecture 07 — Module 5](Lecture-07.md)
