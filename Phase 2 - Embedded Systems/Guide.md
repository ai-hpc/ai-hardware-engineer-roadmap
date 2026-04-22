# Phase 2: Embedded Systems

> *Move from abstract computing models to real boards, buses, boot flows, and constrained systems that must work outside the lab.*

**Layer mapping:** Primarily **L4** (firmware, RTOS, BSP, embedded Linux), with direct connections into **L3** (drivers, DMA, device interfaces) and **L1** (edge AI deployment).

**Role targets:** Embedded Software Engineer · RTOS Engineer · BSP Engineer · Embedded Linux Engineer · Edge AI Engineer

**Prerequisites:** [Phase 1 — Digital Foundations](../Phase%201%20-%20Foundational%20Knowledge/Guide.md)

**What comes after:** [Phase 3 — Artificial Intelligence](../Phase%203%20-%20Artificial%20Intelligence/Guide.md), then one of the Phase 4 tracks: [Xilinx FPGA](../Phase%204%20-%20Track%20A%20-%20Xilinx%20FPGA/1.%20Xilinx%20FPGA%20Development/Guide.md), [NVIDIA Jetson](../Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/1.%20Nvidia%20Jetson%20Platform/Guide.md), or [ML Compiler](../Phase%204%20-%20Track%20C%20-%20ML%20Compiler%20and%20Graph%20Optimization/Guide.md)

---

## Why This Phase Exists

AI hardware does not live in isolation. It ships inside systems that must:

- boot reliably
- talk to sensors and peripherals
- survive power and thermal limits
- update safely in the field
- integrate with Linux, RTOS, and board-specific software stacks

This phase teaches the engineering layer between raw hardware knowledge and deployable products.

---

## Phase Structure

| # | Module | What you learn | Why it matters |
|---|--------|----------------|----------------|
| **1** | [Schematic Capture and PCB Design](1.%20Schematic%20Capture%20and%20PCB%20Design/Guide.md) | board-level design, interfaces, layout basics, bring-up concerns | AI products still depend on real hardware integration |
| **2** | [Embedded Software](2.%20Embedded%20Software/Guide.md) | Cortex-M, FreeRTOS, interrupts, DMA, buses like SPI/I2C/UART/CAN | The control plane around sensors, peripherals, and low-level devices |
| **3** | [Embedded Linux](3.%20Embedded%20Linux/Guide.md) | Yocto, BSP customization, rootfs, kernel integration, production Linux | The software foundation for Jetson, robotics, and edge devices |
| **4** | [Product Design](4.%20Product%20Design/Guide.md) | product role, trust cues, room behavior, setup, and system boundaries | Good embedded systems still fail if they become weak products |

**Recommended order:** `1 → 2 → 3 → 4`

If you are using development kits instead of custom boards, Module 2 can start in parallel with parts of Module 1. Module 4 is most valuable once you already have enough hardware and software context to reason about tradeoffs, not just features.

---

## What You Should Produce

By the end of this phase, you should have at least:

- one hardware or interface-oriented artifact such as a schematic review, interface map, or bring-up checklist
- one MCU/RTOS artifact such as a peripheral driver, ISR design note, or FreeRTOS project
- one embedded Linux artifact such as a Yocto image build note, device-tree change, rootfs customization, or BSP write-up
- one product artifact such as a V1 design brief, trust model, or placement-and-controls rationale

These outputs matter because embedded credibility comes from working systems, not generic familiarity.

---

## Exit Criteria

You are ready to continue when you can:

- explain how a peripheral reaches software through buses, interrupts, and drivers
- reason about RTOS tasking vs bare-metal vs Linux tradeoffs
- identify where BSP, bootloader, device tree, and rootfs customization fit in a deployment stack
- explain at least one product tradeoff involving controls, trust, acoustics, placement, or setup
- produce at least one reproducible bring-up or embedded Linux artifact

---

## Who Should Prioritize This Phase

- **Jetson / robotics / edge AI targets:** treat this phase as mandatory
- **Compiler-first learners:** do enough to understand deployment realities, then move to Phase 3 and Phase 4C
- **FPGA / accelerator learners:** focus on the Linux and interface pieces that connect custom hardware to usable systems
- **Consumer device / home AI / voice product targets:** do not skip Module 4, because room-role and trust decisions can invalidate otherwise solid engineering

---

## Next

→ [**Phase 3 — Artificial Intelligence**](../Phase%203%20-%20Artificial%20Intelligence/Guide.md) · [**Phase 4 Track A — Xilinx FPGA**](../Phase%204%20-%20Track%20A%20-%20Xilinx%20FPGA/1.%20Xilinx%20FPGA%20Development/Guide.md) · [**Phase 4 Track B — NVIDIA Jetson**](../Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/1.%20Nvidia%20Jetson%20Platform/Guide.md) · [**Phase 4 Track C — ML Compiler**](../Phase%204%20-%20Track%20C%20-%20ML%20Compiler%20and%20Graph%20Optimization/Guide.md)
