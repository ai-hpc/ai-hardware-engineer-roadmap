# Lab: Define the Arithmetic and Memory Contract Before HLS

**Time:** 60–90 minutes. **Needs:** Python 3 and a C++17 compiler (`g++` for the test harness). No FPGA or AMD tools are required. This lab runs ordinary host C++ and checks analytical memory-bank examples. It does not run AMD HLS C simulation, RTL co-simulation, synthesis, implementation, or hardware.

A matrix-vector accelerator computes `y[row] = sum(W[row][column] × x[column])`. The arithmetic looks simple; the hardware contract still needs exact shapes, widths, rounding, storage, and ownership. An optimization that silently drops the last three columns is incorrect even if its estimated initiation interval improves.

## 1. Run the checked baseline

From this directory:

```bash
python3 -m unittest discover -s . -p 'test_*.py' -v
g++ -std=c++17 -O2 -Wall -Wextra -Werror verify_contract.cpp -o /tmp/course-matvec-contract
/tmp/course-matvec-contract
```

Expected successful program output:

```text
verified_shapes=512 rounding_cases=240 bank_cases=6
```

The three Python tests compile optimized and AddressSanitizer/UndefinedBehaviorSanitizer variants and two deliberately broken variants. Missing `g++` is an explicit skip, not a successful native run. The review host used GCC 13.3.0; [validation.json](validation.json) records source hashes and scope.

Read [matvec_contract.h](matvec_contract.h) before [verify_contract.cpp](verify_contract.cpp). The first contains the candidate arithmetic; the second contains an independent scalar reference and boundary checks. It uses division/remainder for rounding rather than copying the candidate's add-half-and-shift implementation.

## 2. Write the contract in plain language

- Signed 8-bit weights and inputs range from −128 through 127.
- Active shape is 1–8 rows and 1–64 columns; physical storage always has the fixed maximum shape.
- Accumulation uses signed 32-bit integers. Four partial accumulators handle groups of four columns; bounds checks preserve incomplete final groups.
- Divide by `2^shift`, with `shift` in 0–15, rounding to nearest with halfway cases **away from zero**; then saturate to signed 16-bit output.
- Invalid shape/shift returns `false` before modifying output. Inactive output rows remain unchanged.
- The caller provides distinct, live input/output arrays. The function is synchronous; it is not a DMA submission API.

The internal `requantize` helper assumes `shift <= 15`; the public `matvec` boundary enforces it. Its temporary magnitude is 64-bit so negating `INT32_MIN` is defined. Do not replace this with `abs(int32_value)` and assume the minimum signed value fits a positive signed 32-bit result.

**Worked fixture:**

```text
W = [ 1  2  3 ]    x = [ 4 ]    raw sums = [ 15 ]
    [-1  0  2 ]        [-2 ]               [  6 ]
                       [ 5 ]
```

At shift 0, output is `[15, 6]`. At shift 1, output is `[8, 3]`; `15/2 = 7.5` rounds to 8. A separate `−3/2` case rounds to −2. A framework using ties-to-even would have a different contract for some inputs. Select a deployment quantization rule explicitly rather than labeling every rule “INT8.”

## 3. Prove an accumulator bound

The largest product magnitude is `(-128) × (-128) = 16,384`. A conservative bound for 64 products is `64 × 16,384 = 1,048,576`. A signed 21-bit value has maximum 1,048,575, so **22 signed bits** are sufficient for that positive bound; signed 32-bit accumulation comfortably covers it. Bias terms or larger shapes require a new bound.

Do not narrow to 16 bits before accumulation and hope output saturation repairs it. C++ signed overflow can be undefined; a hardware truncation can wrap. Neither is equivalent to saturating the final exact sum. The all-extreme tests exercise both output limits. They also check every shape, including 1, 3, 5, 63, and 64 columns, and ensure inactive output rows retain their sentinel.

**Predict a failure:** the drop-tail mutation skips a final group smaller than four. Which fixture detects it immediately? **Answer:** the 2×3 fixture. The wrap-output mutation narrows without saturation; the all-`−128` product case must detect that error.

## 4. Count memory demand before requesting unroll

Four column lanes need four weight values and four vector values per group. If the arrays are separate memories with two usable read ports each, the ideal read-service lower bound is `ceil(4/2) = 2` cycles per group for each array. Count both arrays, but do not add their service times if they can be read in parallel. Sharing one physical memory changes the model.

For the lab's simple cyclic banking model, `bank(address) = address mod bank_count`. Each request is counted separately; there is no broadcast, cache reuse, write traffic, or arbitration overhead.

| Word addresses requested together | Banks | Read ports/bank | Read-service lower bound |
|---|---:|---:|---:|
| 0, 1, 2, 3 | 4 | 1 | 1 cycle |
| 0, 4, 8, 12 | 4 | 1 | 4 cycles |
| 0, 4, 8, 12 | 4 | 2 | 2 cycles |
| 0, 64, 128, 192 | 4 | 1 | 4 cycles |

The last row illustrates four row starts in a row-major array with stride 64. Partitioning the wrong dimension can leave every parallel access in the same bank. These are lower bounds under explicit assumptions, **not synthesized initiation intervals**. Real memory binding, read-during-write behavior, dependencies, and timing can be stricter. AMD explains the available choices in [array partitioning](https://docs.amd.com/r/en-US/ug1399-vitis-hls/Array-Partitioning).

## 5. Include recurrence and interface limits

Four partial accumulators reduce a single long dependency into four lanes, but each lane still depends on its own previous value. If a hypothetical pipelined add has latency three cycles and each lane is updated every outer-loop iteration, that recurrence can require group II ≥ 3 unless the design changes. Having enough memory ports does not remove a true data dependency. Never use a “false dependency” directive to deny a dependence that the algorithm really has.

At an **assumed implemented** 100 MHz, four elements per group and II=2 imply an ideal rate of 200 million products/s during the active inner loop. This is neither whole-row throughput nor whole-invocation throughput. Include fill/drain, final reduction, output conversion, control, and transfers. See [HLS scheduling and binding](https://docs.amd.com/r/en-US/ug1399-vitis-hls/Understanding-High-Level-Synthesis-Scheduling-and-Binding).

For 8×64 int8 weights, one int8 vector, and eight int16 outputs, the active payload is `512 + 64 + 16 = 592 bytes` if each input is transferred once. The result is not “512 MACs divided by peak MAC/s” alone. Reusing weights across vectors changes transfer cost; state that reuse assumption.

## 6. Continue to an actual HLS experiment

AMD UG1399 currently resolves to the 2026.1 documentation reviewed on 2026-10-18. Use the tool/platform release supported by your selected board and export flow; this lab has not validated a vendor installation.

1. Make the arithmetic function the candidate top, and separate testbench code from design code. Keep the same numerical and invalid-input contract, or document a hardware error/status interface that preserves it.
2. Run vendor C simulation with the fixtures. Save tool version, target part, clock constraint, sources, and output.
3. Compare baseline, four-column unroll, and an appropriately banked local-memory variant. Interface arrays on AXI memory ports may require loading into local arrays before partitioning has the intended effect.
4. Read achieved II, recurrence/resource warnings, operator binding, memory ports, and estimated resources. A pragma is a request, not evidence of success.
5. Run RTL co-simulation with valid/ready stalls and explicit packet/length behavior, then implementation timing for the selected device. Host tests do not cover these interfaces.

A useful experiment can report that the optimized variant lost: extra banking may cost too much area, routing may lower the achieved frequency, or transfer overhead may dominate. Submit the numerical tests, one actual report per tool stage you ran, and a short explanation of the limiting resource. Mark unrun stages clearly.
