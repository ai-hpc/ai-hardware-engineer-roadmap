# A streaming RTL stage that survives backpressure

Build and test a small piece of an AI camera pipeline: accept an unsigned byte, add one with saturation, and hold the result until the next stage can accept it. This takes about 60–90 minutes with a simulator installed. No FPGA board, GPU, Vivado license, or Python package is required.

**Before you start:** recognize an 8-bit signal, a rising clock edge, and a nonblocking assignment (`<=`). Read the [HDL introduction](../../../Phase%201%20-%20Foundational%20Knowledge/1.%20Digital%20Design%20and%20Hardware%20Description%20Languages/Hardware%20Description%20Languages%20%28HDLs%29/Guide.md) if these are new. The [numbers and bytes lab](../../../Phase%201%20-%20Foundational%20Knowledge/Labs/Numbers-Shapes-and-Bytes.md) explains saturation in software.

**You will leave with:** a passing transaction scoreboard, one waveform, and evidence that three deliberate faults are detected. The supplied RTL and testbench were executed with Icarus Verilog 12.0 on Ubuntu 24.04, Python 3.12, on 2026-10-18. This validates the simulated cases. Synthesis, implementation, static timing, power, and board operation have not been run for this example.

## 1. Understand the agreement between stages

Imagine passing a labeled box to a colleague. `valid` means “I am holding a real box”; `ready` means “I can take it now.” At a rising clock edge, a transfer happens only when both are 1. If the recipient is not ready, the sender holds the same box and keeps `valid` asserted. The sender must not wait for `ready` before presenting an available item.

Our stage has input `s_*` signals and output `m_*` signals. It stores one output item. Both sides use the **same clock**; this circuit does not cross clock domains.

| Contract | Exact behavior |
|---|---|
| Input transfer | `s_valid && s_ready` at a rising edge outside reset |
| Computation | `result = min(unsigned(input) + 1, 255)` |
| Output transfer | `m_valid && m_ready` at a rising edge outside reset |
| Output stalled | Preserve `m_valid=1` and `m_data` until acceptance or reset |
| Capacity | One item; accept into an empty slot or replace an item leaving now |
| Reset | Active high; suppress transfers while asserted; clear stored item and sticky flag at a rising edge |
| Overflow flag | Set on an **accepted** input of 255; remain set until reset |

The reset is a synchronous reset for the stored state, with combinational masking of interface handshakes. Assert it across at least one rising edge. This teaching contract does not implement every AXI interface/reset requirement. Ready/valid flow control is the foundation; complete AXI4-Stream integration also needs its specified reset behavior, optional sideband signals, and interface timing rules. See [AMD's transfer explanation](https://docs.amd.com/r/en-US/pg256-sdfec-integrated-block/AXI4-Stream-Interface) and [stable payload rule](https://docs.amd.com/r/2021.2-English/ug1483-model-composer-sys-gen-user-guide/AXI4-Stream-Support-in-Model-Composer).

There are two different meanings of overflow here. **Arithmetic overflow** would make 255 + 1 wrap to 0 in eight bits; our numerical contract clamps it to 255. **Buffer overflow** means storing more items than fit; this stage prevents that by lowering `s_ready`. An upstream device that cannot pause needs a larger buffer and an explicit loss policy.

## 2. Predict the first transfers

Values below are sampled immediately **before** each rising edge, after reset. A dash means the value is irrelevant because its valid bit is 0.

| Edge | Input valid/data | Input ready | Output valid/data | Output ready | What happens at the edge |
|---|---|---|---|---|---|
| A | 1 / 10 | 1 | 0 / — | 0 | Accept 10; store 11 |
| B | 1 / 255 | 0 | 1 / 11 | 0 | No transfer; both senders hold their items |
| C | 1 / 255 | 0 | 1 / 11 | 0 | Still stalled; nothing may be overwritten |
| D | 1 / 255 | 1 | 1 / 11 | 1 | Deliver 11 and accept 255 at the same edge |
| E | 0 / — | 1 | 1 / 255 | 1 | Deliver saturated result 255; slot becomes empty |

Notice that `s_ready` can be 1 while the output slot is occupied: if its old item leaves at this edge, a new item can replace it. This gives a possible steady rate of one item per clock with a ready sink. The minimum input-to-output-transfer latency is one clock. These are cycle properties, not measured MHz or a timing-closure result.

Open [stream_increment.sv](stream_increment.sv). Find the expression `!full || m_ready`. Explain why changing it to `!full` preserves correctness but inserts a bubble between transfers. Find the absent final `else` in the clocked block: it means the registers retain their values during a stall.

## 3. Run the actual simulator

Install Icarus Verilog using its [official installation instructions](https://steveicarus.github.io/iverilog/usage/installation.html). On Ubuntu, the distribution package is `iverilog`; on macOS, upstream documents Homebrew's `icarus-verilog` package. Use an installation appropriate for your machine. Both `iverilog` and `vvp` must be on `PATH`, or set `IVERILOG` and `VVP` to executable paths. A Python-only test cannot replace an HDL simulator here.

From the repository root:

```bash
cd 'Phase 4 - Track A - Xilinx FPGA/1. Xilinx FPGA Development/labs'
iverilog -V
python3 run_stream_lab.py --seed 1 --output /tmp/stream-report.json --vcd /tmp/stream.vcd
python3 -m unittest discover -s . -p 'test_*.py' -v
```

The runner compiles real SystemVerilog with `iverilog -g2012 -Wall`, then runs the compiled simulation with `vvp`. Temporary compilation files are removed automatically. The two unit tests run three random seeds and three fault cases. A missing simulator is reported as a **skip** by unittest and as an error by the direct runner; a skip does not establish correctness.

Seed 1 produced:

```text
accepted=481 delivered=480 flushed=1 stalls=438 replacements=412 saturated=3 cycles=1013
```

Why are 481 accepted but only 480 delivered? The directed reset test deliberately discards one queued item. The accounting rule is `accepted = delivered + reset_flushed + still_queued`. At the end, the queue must be empty. Compare your output with this record before changing the design.

For direct tool practice, the equivalent normal compile/run is:

```bash
iverilog -g2012 -Wall -s tb_stream_increment -o /tmp/stream.vvp \
  stream_increment.sv tb_stream_increment.sv
vvp /tmp/stream.vvp +SEED=1 +VCD=/tmp/stream.vcd
```

The waveform is optional. If GTKWave is installed, open `/tmp/stream.vcd`; add `clk`, `rst`, both valid/ready/data groups, and `overflow_seen`. Locate the A–E sequence above. Inspect signal values just before each rising edge, then the registered updates after it.

## 4. Know what the testbench proves

[The testbench](tb_stream_increment.sv) drives inputs at falling edges and checks at rising edges before the DUT's nonblocking assignments update its registers. This avoids a driver/checker race. The reference model works from accepted inputs, widens the arithmetic, clamps to 255, and tracks slot occupancy independently of the RTL register names.

It checks order, data, capacity, valid/ready behavior, stable stalled output, the producer's own hold behavior, overflow status, reset flushing, and final accounting. Directed cases cover the tricky transitions; 1,000 additional cycles use reproducible pseudorandom backpressure. Random testing extends coverage but does not prove every possible signal sequence.

Prove the checker notices mistakes:

```bash
python3 run_stream_lab.py --inject-fault overwrite-stall
python3 run_stream_lab.py --inject-fault wrap-overflow
python3 run_stream_lab.py --inject-fault retain-on-reset
```

These commands alter a temporary copy of the RTL. Expected JSON status is `fault_detected`, with assertions `STALL`, `DATA`, and `VALID`, respectively. The broken simulation exits nonzero; the Python experiment succeeds only if the intended check catches it. A compiler syntax error does not count as successful fault detection.

## 5. Change one thing, then transfer the idea

**Small variation:** run `--seed 17` and `--seed 2026`. The transfer counts may change, but the accounting invariant and all checks must still hold. Save a small table of accepted, delivered, flushed, and stalled counts. Do more stalls imply dropped items? Explain using your evidence.

**Design exercise:** implement a two-slot queue as a separate module and adapt the reference to a queue. Keep data and overflow contracts unchanged. Test empty, full, simultaneous enqueue/dequeue, reset while full, and sustained backpressure. Predict whether it can remove downstream stalls permanently before running it.

**AI/robotics transfer:** replace the byte with `{frame_id, capture_timestamp, pixel_value}`. Every field associated with an item must stay stable through a stall and pass through the same queue. A correct hardware queue can still deliver a frame too late for a robot's control deadline. Add an age acceptance policy at the system boundary; reuse the [observation freshness lab](../../../Phase%202%20-%20Embedded%20Systems/Labs/Observation-Freshness.md).

## 6. Check your understanding

1. A source presents 10, then changes to 20 while `valid=1, ready=0`. Is this allowed? **Answer:** no; the original unaccepted item must remain stable. The testbench checks its own producer for this error.
2. Can the output send the input accepted at edge A at the same edge? **Answer:** no; this stage registers the result. Its earliest output transfer is the next rising edge.
3. A value 255 appears while `s_ready=0`. Should it set the sticky overflow flag? **Answer:** no; it has not been accepted yet.
4. Does a 10 ns testbench clock establish a 100 MHz FPGA implementation? **Answer:** no; simulation schedules ideal events. Device mapping, placement, routing, constraints, and static timing determine whether an implementation meets that period.
5. Does a two-slot queue solve a consumer that is permanently slower than its producer? **Answer:** no; it absorbs only a finite burst. The system needs throttling, more service capacity, or an explicit drop policy.

If compilation fails, check tool availability and the `-g2012` flag first. If a `STALL` failure appears, inspect the first stalled transfer and the register enable. If the test hangs, inspect whether the producer waits for `ready` before asserting `valid`. If unexpected data remains after reset, inspect both valid state and the reference's reset epoch.

Return to [FPGA development Unit 3](../Guide.md) to synthesize and constrain your design. Report simulation results separately from utilization, timing, and hardware evidence. The ready path is combinational in this example; long chains can need registered ready/skid buffers and additional storage, which require fresh correctness and timing checks.
