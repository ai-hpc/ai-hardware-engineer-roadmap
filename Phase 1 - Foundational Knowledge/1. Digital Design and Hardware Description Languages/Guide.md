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

## Practice and assessment

**Entry diagnostic (prerequisite check):** Trace a truth table and one clocked state transition without a simulator. If this is unfamiliar, use the prerequisite material above or the [roadmap diagnostics](../../Learning-Guide.md#diagnose-before-reading) before starting the lab.

1. **Baseline lab:** Build a small counter or ready/valid block and a self-checking testbench; predict its waveform before simulation.
2. **Guided variation:** Vary reset timing and apply backpressure; demonstrate that the checker catches one deliberately incorrect transition.
3. **Independent transfer:** repeat the method on a new input or configuration and explain where your original assumptions stop applying.

**Evidence to submit:** Waveforms, assertion results, cycle latency, and a stated clock/reset contract. Keep raw results and reproduction steps; distinguish measured results from estimates or work not run.

**Self-check:** Which observation would disprove your explanation of the result? Identify a competing explanation and the smallest experiment that separates them. A passing answer names a concrete predicted difference, not only another tool to try.

**Exit criteria:** reproduce the baseline, demonstrate the negative/variation check, and meet the [shared completion rubric](../../Learning-Guide.md#completion-rubric). Use the [lesson cycle](../../Learning-Guide.md#a-repeatable-lesson-cycle) for retrieval practice and spaced review.
