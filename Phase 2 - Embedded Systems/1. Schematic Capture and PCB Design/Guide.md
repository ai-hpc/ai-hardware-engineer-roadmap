# Schematic Capture and PCB Design

<div class="course-identity pcb-design" markdown="1">
<div class="course-identity__icon">PCB</div>
<div markdown="1">
<p class="course-identity__eyebrow">Module 1 · Schematic & PCB Design</p>
<p class="course-identity__title">Read, review, and design the board-level interfaces that make AI hardware usable.</p>
<p class="course-identity__meta">Artifact: schematic review or interface map · Measure: constraints, risks, bring-up checks</p>
</div>
</div>


**Phase 2, section 1** — turn a **netlist intent** into a **fabricated board** you can power, probe, and later program. This sits after Phase 1 (digital + HDL + architecture intuition) and before [**ARM MCU, FreeRTOS, and protocols**](../2.%20Embedded%20Software/Guide.md), where you assume a working PCB or dev kit.

---

## 1. Tooling and libraries

* **EDA choice:** KiCad (open, common in community projects), Altium/Cadence (industry), or comparable — pick one stack and learn its library, schematic, and board editors deeply.
* **Symbols and footprints:** Verified footprints vs datasheet dimensions; 3D models for mechanical check; revision control for project + libraries (Git LFS for large assets if needed).
* **Design rules:** Capture electrical rules (net classes, clearances) early so the layout tool enforces them consistently.

---

## 2. Schematic capture

* **Block diagram first:** Power tree, processors/MCUs, sensors, comms (USB, Ethernet, CAN), regulators, clocks, reset, and test points.
* **Power integrity on paper:** Decoupling strategy per rail (bulk + ceramic placement budget); sequencing if you use multiple supplies.
* **Interfaces:** Level shifting, ESD where connectors leave the board, series resistors for debug/series termination where appropriate.
* **Design for test:** Test pads, 0 Ω jumpers, optional unpop footprints for rework.

---

## 3. PCB layout (introductory through intermediate)

* **Stackup:** Layer count, plane assignment (ground/power), controlled impedance when you run high-speed signals (USB, DDR, Ethernet PHY).
* **Placement:** Short critical paths (switching regulators, crystal to MCU, high-speed differential pairs), thermal vias under hot parts.
* **Routing:** Differential impedance, length matching where specs demand it, return paths and splits in ground (avoid crossing gaps).
* **DFM / DFA:** Courtyard clearances, fiducials, panelization awareness, assembly-friendly footprints (hand vs reflow).

---

## 4. Outputs and handoff

* **Fab package:** Gerbers/OBD++, drill files, pick-and-place if applicable, fab drawing with stackup notes.
* **BOM:** Manufacturer part numbers, alternates, lifecycle awareness.
* **Bring-up plan:** Power rails first (current-limited supply), then clocks/reset, then programming/debug connectors — feeds directly into firmware work in **section 2**.

---

## Resources

* Vendor app notes for your MCU/PMIC/PHY (layout examples are often the best teaching).
* *High-Speed Digital Design* / *Signal and Power Integrity* references when you outgrow “rules of thumb.”

---

## Next in Phase 2

**[Embedded Software — ARM MCU, FreeRTOS, protocols](../2.%20Embedded%20Software/Guide.md)**
