# Lecture 3 — Module 1: what Yocto is and what it produces

**Course:** [Yocto guide](../Guide.md) · **Previous:** [Lecture 02](Lecture-02.md) · **Next:** [Lecture 04](Lecture-04.md)

An embedded product needs more than a kernel binary. It needs an agreed set of userspace programs, libraries, firmware, configuration, and boot artifacts, built for a particular target and tracked over time.

## Name the parts

**BitBake** reads metadata and schedules tasks. **OpenEmbedded-Core** supplies recipes, classes, and base configuration. **Poky** is the reference integration/distribution used by this course. The **Yocto Project** brings together the tools, documentation, testing, and ecosystem around this work. A Yocto-based image is an output of that system, not a universal image you install on every board.

This is useful when several products share software but differ in hardware or image policy. It also has costs: initial learning, build resources, long-term source and layer maintenance. Compare a supported vendor image or Buildroot against your actual requirement rather than choosing a build system for its name. See the [Yocto concepts manual](https://docs.yoctoproject.org/5.0.20/overview-manual/yp-intro.html) and [Buildroot manual](https://buildroot.org/downloads/manual/manual.html).

## Worked requirements: observation replay appliance

| Product requirement | Course decision | Yocto object |
|---|---|---|
| Learn without a board | Boot an x86-64 QEMU target | `MACHINE = "qemux86-64"` |
| Validate a capture-to-completion age | Include `observation-check` | Application recipe/package |
| Keep the first image small | Start from `core-image-minimal` | Custom image recipe |
| Rebuild from known inputs | Save exact upstream and product-layer revisions | Release manifest |
| Let another developer build the same app | Generate SDK from the image configuration | `populate_sdk` output |
| Add a real camera later | Select supported capture hardware and driver | BSP and runtime integration, outside this baseline |

The age checker does not capture a sensor or synchronize clocks. Its known input/output cases make it useful as a packaging smoke test.

## Your turn

Write five requirements for a named robot or edge-AI device. For each, say whether it belongs in hardware/BSP, a runtime package, image policy, or the deployment/update process. Keep an optional-feature list separate so a first image does not turn into a full desktop installation.

For a richer design example, read the [edge AI camera requirements](Lab-01-Worked-Example-Edge-AI-Camera.md). Treat its board-specific sketches as integration exercises, and use the pinned QEMU sequence here for the first build.

**Self-check:** if an image contains your program, does it automatically provide an update/rollback strategy? **Answer:** no. Packaging selects software; rollback requires a boot/storage/update design and demonstrated failure recovery. Similarly, smaller images can reduce installed software but are not automatically secure or correct.

**Artifact:** a one-page requirement-to-build-object map. Carry the same product name and required runtime behavior into the next lesson.

**Previous:** [Lecture 02](Lecture-02.md) · **Next:** [Lecture 04](Lecture-04.md)
