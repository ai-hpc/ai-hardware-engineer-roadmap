# Lauterbach TRACE32® Debug (Autonomous Driving — advanced tooling)

<div class="course-identity auto-course" style="--course-accent: #65a30d; --course-accent-rgb: 101, 163, 13;" markdown="1">
<div class="course-identity__icon">LTDA</div>
<div markdown="1">
<p class="course-identity__eyebrow">Deep Dive · Specialization</p>
<p class="course-identity__title">Specialized course identity for Lauterbach TRACE32® Debug (Autonomous Driving — advanced tooling).</p>
<p class="course-identity__meta">Artifact: specialization case study · Measure: performance, reliability, role fit</p>
</div>
</div>


**Parent:** [Phase 5 — Autonomous Driving](../Guide.md) · Optional professional depth

**Prerequisites:** [Phase 1 — Operating Systems](../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Guide.md) (boot, JTAG concepts), [Phase 2 — Embedded Linux](../../../Phase%202%20-%20Embedded%20Systems/3.%20Embedded%20Linux/Guide.md) and [Phase 2 — Embedded Software](../../../Phase%202%20-%20Embedded%20Systems/2.%20Embedded%20Software/Guide.md) (MCU, RTOS, bring-up). Strong overlap with **automotive ECU**, **functional safety**, and **silicon validation** roles.

**TRACE32®** is a registered trademark of **Lauterbach GmbH**. This roadmap page is **educational**; feature names and training follow **vendor documentation**.

---

## Why this belongs in Autonomous Driving

**TRACE32** is not a replacement for **GDB** on a healthy Linux app—it is a **class of in-circuit debug and trace** used when:

- You need **hardware-assisted** run-control on **bare-metal**, **RTOS**, or **early bootloader** code where no OS debugger exists.
- You must capture **non-intrusive instruction trace** (ETM, PTM, Nexus, etc.) for **timing**, **coverage**, or **post-mortem** analysis after a fault.
- You work on **automotive** or **industrial** SoCs (**AURIX™ TriCore**, **ARM Cortex-R/M/A**, **Renesas RH850**, **RISC-V**, and many others) where **OEM/Tier-1** flows standardize on **Lauterbach** or equivalent **ICE** tools.
- **Safety** arguments (e.g. **ISO 26262** evidence) require **structural coverage** or **execution trace** that software-only tools cannot provide alone.

Treat this track as **tooling literacy** for **senior embedded / firmware / validation** engineers—not a substitute for official **Lauterbach training** or your project’s **safety plan**.

---

## What TRACE32 is (mental model)

| Layer | Role |
|-------|------|
| **Debug probe** | Hardware interface to the target (**JTAG**, **cJTAG**, **SWD**, **DAP**, vendor-specific) — connects PC host to SoC **debug port**. |
| **TRACE32 software** | **IDE + debugger + trace viewer + scripting** (Practice / **PRACTICE** language) — load symbols, breakpoints, memory, peripherals, multicore sync. |
| **Trace (optional)** | **On-chip trace buffer** (ETB, **ETF**) or **parallel trace port** (**TPIU**) + **trace probe** — reconstruct program flow, timing, and bus activity **without** stopping the CPU (configuration-dependent). |

**Compared to open-source stacks:** **OpenOCD** + **GDB** cover many **MCU** bring-up scenarios at low cost. **TRACE32** typically wins where **vendor silicon** ships **complex trace**, **multicore lockstep**, **automotive** ecosystem support, or **dedicated FAE** engagement is required.

---

## Core skills to build

1. **Target connection** — Power sequencing, **reset**, **debug connector** pinout, **adapters**, **target voltage**, **JTAG chain** discovery.  
2. **Run-control** — Step, go, halt, **breakpoints** (hardware vs software), **watchpoints**, **conditional** breaks.  
3. **Memory and registers** — Core **GPRs**, **special registers**, **MMIO**, **cache** / **MPU** awareness (invalidate vs “stale” views).  
4. **Multicore** — **SMP** / **AMP** / **lockstep** views; **synchronized** run-control where the architecture requires it.  
5. **Trace** — Enable **ETM** (or equivalent), size **buffer**, **trigger** (e.g. around fault), **export** for analysis; understand **trace clock** and **pin** requirements on **your** PCB.  
6. **Scripting** — Automate **regression** debug (flash, boot, test, capture trace) with **PRACTICE** or host-side scripts.  
7. **RTOS awareness** — **Task-aware** debugging when the **kernel** exposes the right **symbols** and **plugins** exist for your RTOS.

---

## Suggested learning path

| Stage | Action |
|-------|--------|
| **1** | Read **Lauterbach** “Debugger Basics” / **Getting Started** for **one** architecture you use (e.g. **ARM**, **TriCore**). |
| **2** | On a **lab board** with known-good **Blinky**, connect probe, verify **CPU** detection and **flash** programming if applicable. |
| **3** | Exercise **breakpoints** in **interrupt** and **main**; add **watchpoint** on a **status register**. |
| **4** | If hardware supports it, complete one **trace** lab: **trigger** on function entry, **decode** **PC** timeline. |
| **5** | Map **TRACE32** usage to **your** **V-model** phase: **unit** vs **integration** vs **HIL** (often **trace** is constrained or disabled in final **ECU** builds—know your **OEM** rules). |

**Official entry points (verify current URLs):**

- [Lauterbach — TRACE32](https://www.lauterbach.com/trace32.html) — product overview and documentation index.  
- Architecture-specific **PDF manuals** (ARM, PowerPC, TriCore, RH850, RISC-V, …) from Lauterbach’s documentation portal after registration if required.

---

## Hardware checklist (for your own carrier / ECU)

- **Debug connector** on **schematic** (**Tag-Connect**, **MIPI-10/20**, **Samtec**, **OEM-specific**) — **do not** rely on **handsolder** wires for **trace**-grade signals.  
- **Dedicated** **JTAG/SWD** **pins** **not** multiplexed with **GPIO** used in production unless **straps** allow safe **bring-up**.  
- For **parallel trace**: **length-matched** **trace** **lines**, **reference** **VTREF**, **ground**, and **shielding** per **SoC** and **probe** manual.  
- **Reset** and **power-good** **observable** on **scope** when **debugger** “cannot attach.”

---

## Relationship to other roadmap modules

| Module | Connection |
|--------|------------|
| **Phase 5 — Autonomous Driving** | **ECU** / **VCU** firmware, **openpilot**-adjacent stacks often pair **application** debug (Linux) with **MCU** **JTAG** on separate chips. |
| **Phase 4 — L4T / Jetson** | **Jetson** bring-up is usually **UART**, **USB recovery**, and **Linux** debuggers; **TRACE32** is more typical on **ARM/RISC-V MCUs** and **automotive** **SoCs** than on **Orin** application processors—still valuable if you touch **safety** **companion** **MCUs** or **customer** **silicon**. |
| **Phase 5 — AI Chip Design** | **Silicon bring-up** and **RTL/verification** teams often use **industry** **ICE** tools alongside **simulation**. |

---

## Projects (optional)

- **Bring-up script** — One **PRACTICE** script: attach, load **ELF**, set **breakpoint** on **`main`**, run, print **stack**.  
- **Trace mini-report** — Capture **10 ms** of **trace** around a **timer ISR**; annotate **ISR** **latency** in a short markdown note.  
- **Comparison note** — One page: **OpenOCD+GDB** vs **TRACE32** on the **same** **MCU** board for **your** pain points (attach time, **RTOS** plugins, **trace**).

---

## Summary

**Lauterbach TRACE32®** is **advanced embedded debug and trace** tooling for **serious** **SoC** and **ECU** work. Add it to your roadmap when you move from **“printf and GDB”** to **multicore**, **silicon validation**, **automotive**, or **safety**-adjacent **evidence**—and budget for **probe hardware**, **licenses**, and **vendor training**.
