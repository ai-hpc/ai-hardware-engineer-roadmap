# Digital Design and Hardware Description Languages

<div class="course-identity digital-design" markdown="1">
<div class="course-identity__icon">RTL</div>
<div markdown="1">
<p class="course-identity__eyebrow">Module 1 · Digital Design & HDL</p>
<p class="course-identity__title">Turn Boolean logic and sequential circuits into testable RTL.</p>
<p class="course-identity__meta">Artifact: Verilog/SystemVerilog block · Measure: simulation coverage, timing, resource use</p>
</div>
</div>


**Phase 1, section 1** — combinational and sequential logic, memory technologies at a digital level, and **Verilog (and related HDL practice)** so you can express hardware you will later map to FPGAs and accelerators.

This module merges what were previously separate **digital fundamentals** and **HDL** tracks into one numbered section. Work through the two guides below in order (or in parallel if you already know basic logic).

---

## 1. Digital design fundamentals

* **[Digital Design Fundamentals](Digital%20Design%20Fundamentals/Guide.md)** — Number systems, Boolean algebra, combinational and sequential logic, memory (SRAM, DRAM, ROM), timing intuition, and project ideas (breadboard through small RTL blocks).

## 2. Hardware description languages

* **[Hardware Description Languages (HDLs)](Hardware%20Description%20Languages%20%28HDLs%29/Guide.md)** — Verilog syntax and semantics, modeling styles, testbenches, synthesis-oriented thinking, and small RTL projects.

---

## Next in Phase 1

**[Computer Architecture and Hardware](../2.%20Computer%20Architecture%20and%20Hardware/Guide.md)** — ISA through memory hierarchy and real platforms, so you know *what* your gates and RTL will sit next to in a system.

## From arithmetic intent to a streaming block

Begin with a two-input signed multiply-accumulate: predict overflow, then implement a registered datapath. Add `valid`/`ready` and define acceptance as both high on a clock edge. Hold output data stable while valid is high and ready is low. Distinguish input acceptance, pipeline latency, and output acceptance in the waveform.

The integration deliverable is a one-page interface contract plus a self-checking testbench. The architecture course can then ask how many such blocks fit the bandwidth budget without changing the numerical contract.

**Carry forward:** Clock/reset and ready/valid contract with boundary-case waveforms. Continue with [the next course](../2.%20Computer%20Architecture%20and%20Hardware/Guide.md) and reuse that evidence.
