# Embedded Linux for a Physical AI SoC

<div class="course-identity embedded-linux" markdown="1">
<div class="course-identity__icon">LNX</div>
<div markdown="1">
<p class="course-identity__eyebrow">Module 3 · Embedded Linux</p>
<p class="course-identity__title">Customize kernels, root filesystems, device trees, BSPs, and production Linux images.</p>
<p class="course-identity__meta">Artifact: bootable image or device-tree patch · Measure: boot time, probe logs, footprint</p>
</div>
</div>


Your model runtime needs a bootable machine, a driver contract, and a recovery path. This module connects the MCU and bus work in [embedded software](../2.%20Embedded%20Software/Guide.md) to a Linux host that can acquire observations, schedule accelerator work, exchange status over wireless, and recover after a failed update.

**Prerequisites:** [OS lectures 5, 12–19](../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Guide.md#lecture-index), a cross-compiled C program, and an interface map identifying registers, interrupts, clocks, reset, and DMA capability. Begin on QEMU or a supported development board; porting a new SoC is a later integration project.

## A sequence that produces one working system

| Step | Learn and implement | Evidence required before continuing |
|---|---|---|
| 1. Boot a baseline | Identify ROM/firmware, bootloader, kernel, device tree, root filesystem, PID 1 | Saved serial log, artifact checksums, target and source revisions |
| 2. Own the image | Add one application through a product layer and image recipe | Application exists on target; its build is reproducible from recorded inputs |
| 3. Describe one device | Match device-tree resources and driver probe to the schematic | Expected compatible/resources, successful probe, controlled missing-device error |
| 4. Submit and complete work | Trace a command, DMA buffer lifetime, interrupt, and userspace completion | Sequence-numbered transfer plus timeout/cancellation behavior |
| 5. Run the observation service | Bound queues, set startup dependencies, capture logs and timing | Observation age and queue high-water mark under CPU/I/O pressure |
| 6. Recover the product | Add watchdog/health policy, signed update inputs, and a tested recovery path | A failed service or test image cannot leave the device in an unexplained state |

Each stage consumes the previous stage’s artifacts. Do not begin performance tuning while image identity, driver ownership, or buffer correctness is uncertain.

## Choose one build system

Start with the [Yocto course](Yocto/Guide.md) for layers, recipes, images, SDKs, and release records. Use the [Buildroot manual](https://buildroot.org/downloads/manual/manual.html) when a whole-image configuration suits the project better. Buildroot package definitions build software into an image; Buildroot is not a conventional distribution package manager for incremental target updates. PetaLinux is a vendor-specific integration route for supported AMD adaptive-computing platforms; keep its release aligned with the hardware export and BSP.

OpenEmbedded-Core supplies metadata, BitBake executes tasks, and the Yocto Project supplies the surrounding tools and integration practices. They are related parts of a workflow, not three independent operating systems to learn in succession.

## Read a complete driver path

The [Jetson ESP-Hosted host-code course](Jetson%20ESP-Hosted%20Host%20Code/Guide.md) traces a real fork from board setup through SPI/IRQ work into Wi-Fi and Bluetooth subsystems. Reconstruct that path from a pinned source revision before adapting it. GPIO identifiers and bus names in a board report are observations about that board/kernel combination, not universal wiring instructions.

For a custom inference accelerator, transfer the method: identify device resources; trace submission and completion; document memory ownership; then validate reset and timeout paths. Wireless control/status and accelerator commands need explicit separation so a reconnect cannot publish an old action as a fresh one.

## Capstone: a replayable observation service

Start with the runnable [observation freshness lab](../Labs/Observation-Freshness.md) to establish queue/drop policy, age checks, and reset behavior on a CPU. Its JSON trace and negative cases form an acceptance fixture for the service. Replacing simulated timestamps with real capture provenance is part of integration, not an automatic consequence of passing that fixture.

Package a small service that reads a recorded observation stream and sends work to a CPU reference backend. Define the backend interface so a GPU, FPGA, or future NPU can replace it later. Every request carries an ID, capture timestamp, input shape/dtype, and deadline; every result carries the same ID, completion timestamp, output shape, and error/status.

**Pass conditions:** a fresh checkout rebuilds the package; the target starts it through its configured init system; a missing backend produces a bounded error; a delayed request is marked stale; restarting the service releases its buffers. Record latency percentiles and maximum observed age with the load and sample count. These observations do not establish a hard real-time guarantee.

Carry the service contract, image manifest, and driver map into [FPGA runtime/driver development](../../Phase%204%20-%20Track%20A%20-%20Xilinx%20FPGA/5.%20Runtime%20and%20Driver%20Development/Guide.md) and [Jetson deployment](../../Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/1.%20Nvidia%20Jetson%20Platform/Guide.md). The eventual SoC must satisfy this software-visible contract as well as the compiler’s tensor contract.
