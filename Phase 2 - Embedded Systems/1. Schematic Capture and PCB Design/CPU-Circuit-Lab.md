# CPU lab: a pull-up that passes nominal values and fails the real budget

**Reviewed and executed 18 October 2026.** Use Python's standard library to analyze a small sensor bus, generate a simulated waveform and explain a failed design. No board, oscilloscope, KiCad or SPICE installation is required. This is an electrical model exercise; no fabricated board or measured waveform is claimed.

**Prerequisites:** [the schematic guide](Guide.md), Ohm's law and seconds versus nanoseconds. **Time:** 60–90 minutes. **Deliverable:** a JSON budget, a CSV waveform, one fault explanation and a component-selection decision with assumptions.

## 1. Predict the failure before running the program

The fictional board has one 3.3 V pull-up rail. Every connected output can guarantee `V_LOW ≤ 0.4 V` while sinking 3 mA under the stated operating conditions. The rail spans 3.135–3.465 V, resistors have ±5% tolerance, and total bus capacitance spans 80–120 pF. The required rise time is at most 300 ns; the required high level is at least 2.31 V. These are exercise inputs, not specifications for an identified MCU or sensor.

For a resistor charging a capacitor with negligible leakage:

```text
V(t) = V_rail × (1 − exp(−t / (R × C)))
t_30_to_70 = R × C × ln(0.7 / 0.3) ≈ 0.8473 × R × C
```

A 3.3 kΩ resistor and 100 pF produce about 280 ns. Predict what happens at **3.465 kΩ and 120 pF**. The circuit does not promise that all components will take nominal values. The pull-up calculation and 30–70% timing convention follow the [TI application note](https://www.ti.com/lit/an/slva689/slva689.pdf); the selected I2C mode and device limits must also agree with [NXP UM10204](https://community.nxp.com/pwmxy87654/attachments/pwmxy87654/nxp-designs/931/1/UM10204.pdf).

## 2. Run all five cases

From the repository root:

```bash
python3 'Phase 2 - Embedded Systems/1. Schematic Capture and PCB Design/labs/circuit_budget.py' --scenario baseline
python3 'Phase 2 - Embedded Systems/1. Schematic Capture and PCB Design/labs/circuit_budget.py' --scenario nominal_only
python3 'Phase 2 - Embedded Systems/1. Schematic Capture and PCB Design/labs/circuit_budget.py' --scenario four_breakouts
python3 'Phase 2 - Embedded Systems/1. Schematic Capture and PCB Design/labs/circuit_budget.py' --scenario long_cable
python3 'Phase 2 - Embedded Systems/1. Schematic Capture and PCB Design/labs/circuit_budget.py' --scenario leakage_fault
```

[The implementation](labs/circuit_budget.py) calculates independent worst-case corners. All pull-ups share one rail; a pull-up to a different voltage is outside this model.

| Case | Nominal no-leak rise | Worst rise | Current at 0.4 V, worst corner | Result within this model |
|---|---:|---:|---:|---|
| One 2.2 kΩ | 186.4 ns | 234.9 ns | 1.47 mA | Pass |
| One 3.3 kΩ | 279.6 ns | 352.3 ns | 0.98 mA | Rise fails |
| Four 2.2 kΩ in parallel | 46.6 ns | 58.7 ns | 5.87 mA | Sink guarantee exceeded |
| One 2.2 kΩ, 160–240 pF cable case | 372.8 ns | 469.7 ns | 1.47 mA | Rise fails |
| One 3.3 kΩ, 300 µA sink-leakage fault | 279.6 ns ignoring leakage | Never reaches 70% | 0.98 mA | High level and rise fail |

The current column is the current the output must sink **at the guaranteed maximum low level**, not a prediction of its exact operating voltage or absolute maximum current. Exceeding that guarantee means the provided evidence no longer establishes a valid low level. It does not let this model predict when the device will be damaged.

A lower resistance improves charging but increases low-state current. Adding breakout boards changes both parallel resistance and capacitance. A change that improves one margin may break another.

## 3. Understand the leakage case

The optional fault draws a constant current from the line. Its steady-state voltage becomes:

```text
V_final = V_rail − I_leak × R
```

At the worst corner, `3.135 − 300e-6 × 3465 = 2.0955 V`. This is below the 2.31 V high requirement and below 70% of the rail. A finite “30–70% rise time” does not exist. The JSON uses `null` plus `reaches_70_percent: false`, rather than exporting an invalid JSON infinity or silently calling the line healthy.

This constant-current fault is a deliberately simple teaching model. Real leakage, clamps and powered-off devices can be voltage-dependent. Check the selected component's powered-off I/O behavior and the entire network before applying this calculation to hardware.

## 4. Generate and inspect a numerical waveform

```bash
python3 'Phase 2 - Embedded Systems/1. Schematic Capture and PCB Design/labs/circuit_budget.py' \
  --scenario baseline --waveform-csv /tmp/i2c-baseline.csv
python3 'Phase 2 - Embedded Systems/1. Schematic Capture and PCB Design/labs/circuit_budget.py' \
  --scenario baseline --probe-pf 40 --waveform-csv /tmp/i2c-probed.csv
```

The first command integrates `C × dV/dt = (V_rail − V)/R − I_leak` with a small fixed step and interpolates its threshold crossings. Its numerical rise should approach the analytic 234.9 ns value. The second adds an assumed 40 pF probe/load: worst rise becomes about **313.2 ns**, so the probe changes the result. Forty pF is an exercise load, not a claim about every oscilloscope probe.

Open the CSV in a plotting tool and label time in ns and voltage in V. Compare the 30% and 70% crossings with the calculation. The test suite reduces the integration step and checks convergence against the independent exponential solution. A numerically smooth waveform does not establish that the lumped model captures transmission lines, transistor output behavior, ringing or a real cable.

## 5. Tests and guided variations

```bash
python3 -m unittest discover -s \
  'Phase 2 - Embedded Systems/1. Schematic Capture and PCB Design/labs' -v
```

Seven tests exercise nominal-versus-corner disagreement, independent component corners, parallel pull-ups, probe loading, unreachable thresholds, invalid inputs and numerical convergence. [Recorded execution](labs/validation-2026-10-18.json) and [source snapshot](labs/source-evidence-2026-10-18.json) are included.

1. **Select a resistor for the long-cable case.** Ignoring leakage, rise at 240 pF requires actual `R ≤ 1475 Ω`. Sink current at the upper rail requires actual `R ≥ 1022 Ω`. With ±5% resistors, the nominal window is roughly **1075–1405 Ω**. A nominal 1.2 kΩ candidate fits this simplified window. Use `dataclasses.replace` on `scenarios()['long_cable']` and assess it; explain the new margin. Verify availability and actual component limits separately.
2. **Make capacitance 480 pF at the upper corner.** The rise bound demands actual `R ≤ 738 Ω`, below the 1022 Ω current bound. No resistor alone satisfies these assumptions. Reduce capacitance, use a suitable buffer/topology, or select a supported lower-speed mode and recalculate. Do not change only the software clock and assume all electrical limits disappear.
3. **Run a random tolerance sample.** It can illustrate likely behavior under a chosen distribution, but zero failures in a finite sample does not prove every bounded corner passes. State the distribution and keep the corner calculation as a separate result.

## 6. Carry the model into the schematic review

List every pull-up already present on modules, the shared rail, expected cable/load capacitance, the weakest sink guarantee and the power-off behavior. Mark the assumptions you still need to obtain from datasheets or measurements. Keep analog rise time, protocol ACK, correct identity register and fresh physical measurement as separate bring-up observations.

The exercise is complete when you can explain why each failed case fails, reproduce one change that fixes the relevant model margin, and name at least two hardware effects the model does not cover. Hand the final pin map and electrical constraints to the firmware lesson; do not hand over only a nominal resistor value.
