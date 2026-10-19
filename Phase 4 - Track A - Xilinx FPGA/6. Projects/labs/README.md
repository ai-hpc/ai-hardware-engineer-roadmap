# RTL Lab: Count Clock Cycles Before Calling Them Time Slots

A scheduling gate can permit transmission only during part of a local frame. It is a useful FPGA exercise, but it is not a synchronized wireless MAC. This lab checks the local counter, wrap, guard interval, and reset behavior in an actual RTL simulator.

**Needs:** Icarus Verilog (`iverilog` and `vvp`) and Python 3. **Time:** 30–45 minutes.

```bash
python3 -m unittest discover -s . -p 'test_slot_gate.py' -v
```

The review used Icarus Verilog 12.0. Two Python tests run ten simulator variants: four valid configurations, five invalid configurations, and one deliberately broken counter. [validation.json](validation.json) records source hashes. Missing Icarus is a skip, not a simulated result. No synthesis, timing analysis, RF transmission, or board test was performed.

## Define the local contract

Read [slot_gate.sv](slot_gate.sv). A frame contains `NUM_SLOTS` slots, each exactly `SLOT_CYCLES` clock periods. During `OWN_SLOT`, the gate is high only after a leading guard and before a trailing guard of `GUARD_CYCLES` each. Parameters are fixed at elaboration. Invalid values terminate simulation; the guard interval must leave at least one active cycle.

The counters use a synchronous reset, and the output is suppressed whenever reset is asserted. In hardware, the reset signal must already satisfy the chosen clock-domain/reset contract. Removing reset directly from an unrelated asynchronous source can produce an unsuitable output transition; this local module does not synchronize it. Resetting separate radio devices at unrelated times does not align their schedules.

With `SLOT_CYCLES=10`, `NUM_SLOTS=4`, `OWN_SLOT=1`, `GUARD_CYCLES=2`, the frame has 40 cycles. The second slot occupies phases 10–19. The enabled interval is phases **12–17 inclusive**, six cycles:

```text
frame phase:  0..........9 10 11 12 13 14 15 16 17 18 19 20.........39
gate:          all low     low  |------ high -----| low   all low
```

The testbench compares the output with an absolute frame-phase interval, checks active-cycle totals, and resets during a later active interval. It also covers non-power-of-two slot counts, a one-cycle/one-slot configuration, and no-guard operation. The latter is a counter boundary test, not a recommended RF timing design.

## Convert units explicitly

At an **assumed implemented clock** of 100 MHz, one period is 10 ns. A 100 µs slot needs `100,000 ns / 10 ns = 10,000 cycles`. Setting the counter limit to 100 creates a 1 µs slot, not 100 µs. The simulation's `#5` half-period makes a 100 MHz testbench clock; it does not prove that an FPGA implementation meets 100 MHz.

The counter resets at `SLOT_CYCLES - 1`, so its states are `0 ... SLOT_CYCLES-1`. The deliberate mutation compares against `SLOT_CYCLES` instead; it makes a slot too long or breaks wrap when that value cannot fit. The phase oracle must reject it.

**Check:** 10,000-cycle slots, four slots, and 200 guard cycles at each end produce how much enabled time for one owner's slot at 100 MHz? **Answer:** `(10000 - 400) × 10 ns = 96 µs` per 400 µs frame, a 24% local duty fraction.

## Explain what the gate cannot do

A real transmitter must also decide whether a whole packet fits before the closing boundary. A local “allowed now” signal cannot safely interrupt an already transmitted symbol or guarantee that an ACK exchange completes before another slot starts. Queuing, radio ramp time, propagation, processing delay, and other devices' clock error all matter.

For example, a packet exchange needing 80 µs cannot be admitted with only 30 µs remaining in the active window. Add an admission rule based on bounded remaining airtime, rather than just sampling `tx_enable`. Explain what happens when the airtime estimate is wrong or a peer misses synchronization. Do not replace a working 802.11 MAC with this module and claim a TDMA network.

Continue to the [wireless video project](../Wireless-Video-FPGA.md) with this limited evidence: local digital phase/guard behavior was simulated. Distributed synchronization, radio correctness, implementation timing, and video performance are separate milestones.
