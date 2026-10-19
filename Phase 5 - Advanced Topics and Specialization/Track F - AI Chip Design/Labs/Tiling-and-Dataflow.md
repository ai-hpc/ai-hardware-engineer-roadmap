# Lab: follow one matrix product from tiles to CPU instructions

A compiler and an accelerator must agree on three things: which numbers an operation produces, where those numbers live, and when they are ready. This lab makes all three visible. You will compute a rectangular matrix product, count the bytes moved by an explicit tiling schedule, follow operand tokens through a small systolic array, and optionally execute the same mathematical operation through a real MLIR compiler pipeline.

Allow 60–90 minutes. You need Python 3.10 or newer for the first three parts, with no additional packages or accelerator. The last part requires MLIR tools; it was executed with **MLIR 20.1.2** on an x86-64 CPU. The Python cycle counts are an analytical model, not timings measured from an FPGA, GPU, or chip.

Read [Lecture 3](../Lectures/Lecture-03.md) for operations and dialects and [Lecture 4](../Lectures/Lecture-04.md) for tensors, buffers, and loops. If matrix dimensions are unfamiliar, begin with [numbers, shapes, and bytes](../../../Phase%201%20-%20Foundational%20Knowledge/Labs/Numbers-Shapes-and-Bytes.md).

## 1. Know the answer before choosing the implementation

For `C = A @ B`, A has shape M×K, B has shape K×N, and C has shape M×N. K is the reduction dimension: every output adds K products. With

```text
A = [1 2 3]    B = [ 7  8]
    [4 5 6]        [ 9 10]
                  [11 12]

C[0,0] = 1*7 + 2*9 + 3*11 = 58
C      = [ 58  64]
         [139 154]
```

there are 12 multiply-accumulates (MACs). Counting one multiplication and one addition as two operations gives 24 operations. That counting convention does not say whether the hardware has a fused instruction or how many cycles execution takes.

From the repository root, run:

```bash
python3 'Phase 5 - Advanced Topics and Specialization/Track F - AI Chip Design/Labs/tiling_lab.py'
python3 -m unittest discover -s 'Phase 5 - Advanced Topics and Specialization/Track F - AI Chip Design/Labs' -p test_tiling_lab.py -v
```

The default problem is deliberately ragged: M=5, N=7, K=3 with tile sizes 2×4×2. The last row, column, and reduction tiles are smaller. Every output is checked against an independent dot-product reference; a checksum alone is insufficient because different wrong matrices can share a sum.

The [implementation](tiling_lab.py) uses Python integers to avoid rounding ambiguity in the functional exercise. Its byte accounting *models* 2-byte inputs and 4-byte accumulators. Python objects do not occupy those sizes, and the model does not emulate overflow or floating-point rounding. A real INT16/INT32 or FP16/FP32 implementation needs its own numerical contract.

## 2. Count what one schedule actually moves

The implemented schedule is output-stationary: hold one C tile locally while looping over K tiles. Each reduction step loads its A and B tiles once. Write C once after the full reduction. C starts at zero, so there is no initial C read. The model assumes no cache reuse between different output tiles and no spill of C during the K loop.

For a full tile with dimensions TM×TN×TK, the working storage is

```text
input_bytes * (TM*TK + TK*TN) + accumulator_bytes * TM*TN
```

At TM=2, TN=4, TK=2 this is `2*(2*2 + 2*4) + 4*(2*4) = 56` bytes. This includes one A tile, one B tile, and C. It excludes DMA descriptors, alignment, double buffering, instruction storage, and other live tensors. The code evaluates the actual boundary tile sizes as well.

For the default shape, there are three M tiles and two N tiles. A is read once for each N tile; B is read once for each M tile:

| Transfer | Calculation | Logical bytes |
|---|---|---:|
| A reads | `M*K*ceil(N/TN)*2` | 60 |
| B reads | `K*N*ceil(M/TM)*2` | 126 |
| C writes | `M*N*4` | 140 |
| Total | A + B + C | **326** |

There are 105 useful MACs. Arithmetic intensity for this schedule is `210/326 ≈ 0.644` operations per logical byte. It is not a measured DRAM intensity: real cache lines and memory transactions can change the byte count.

Try a capacity boundary:

```bash
python3 'Phase 5 - Advanced Topics and Specialization/Track F - AI Chip Design/Labs/tiling_lab.py' --capacity 56
python3 'Phase 5 - Advanced Topics and Specialization/Track F - AI Chip Design/Labs/tiling_lab.py' --capacity 55
```

The first passes; the second reports that the tile needs 56 bytes. Next compare `--tile 2 4 1` with `--tile 2 4 3`. Predict which count changes before running.

**Answer:** peak storage changes from 44 to 68 bytes. Total logical traffic stays 326 bytes because the same elements are loaded over more or fewer K chunks while C remains resident. If a different implementation spills C after every chunk, that conclusion no longer holds.

The zero-dimension contract is explicit. M=0 or N=0 produces no output and no transfers. K=0 with nonempty M and N produces a zero matrix and still writes C. A compiler must preserve this behavior even when it removes an empty reduction loop.

## 3. See why a systolic array needs time to fill

The `systolic` function simulates registered neighbor links. Each processing element (PE) can perform one MAC per cycle. A flows right and B flows down. Input row i and column j are delayed so matching reduction indices arrive at PE(i,j) together:

```text
A[i,k] enters row i at cycle k+i
B[k,j] enters column j at cycle k+j
Both meet at PE(i,j) at cycle k+i+j
```

The simulator carries the reduction index with every operand token and checks their alignment. For a physical 2×2 array and K=3, the active MAC counts are:

| Cycle, starting at zero | 0 | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| MACs performed | 1 | 3 | 4 | 3 | 1 |

The total is 12 MACs in five cycles. The array has 20 available PE-cycles, so useful utilization is `12/20 = 60%` for this isolated tile. The fill/drain formula is `K + array_rows + array_cols - 2`; the first multiplication is cycle zero. A 1×1 array with K=1 therefore needs one cycle, not zero or three.

Smaller output tiles are padded to the physical array dimensions. A 1×1 output on this 2×2 array still issues 12 MACs, but only three contribute to the requested output. It has `3/20 = 15%` useful utilization under this model. Masking unused PEs may save energy; it does not magically shrink the physical pipeline or its loading schedule.

The default report models each reduction tile as a separate fill/compute/drain episode on a 2×4 array, retaining partial C between episodes. Six output tiles each have K chunks of depths two and one: `6*((2+2+4-2)+(1+2+4-2)) = 66` ideal compute cycles. At an assumed 16 bytes/cycle, the logical transfer floor is `ceil(326/16) = 21` cycles.

Perfect overlap gives a lower bound of `max(66,21) = 66` cycles; fully serialized ideal compute and transfers give 87. The latter is **not an upper bound on real latency**. Both exclude address generation, DMA startup, bank conflicts, stalls, clock crossing, arbitration, instruction issue, and control overhead. No clock frequency is assumed. A continuous streaming design can also avoid some modeled restart costs.

**Exercise:** double the modeled bytes/cycle. Does the default perfect-overlap lower bound improve? **Answer:** no; compute remains 66 cycles. This tells you what the model assumes is limiting. It does not prove the real implementation is compute-bound.

## 4. Execute Linalg through LLVM on the CPU

The complete [matmul.mlir](matmul.mlir) program initializes C, runs a 2×3 by 3×2 product, verifies **all four elements**, deallocates C, and returns 415 on success or −1 on a value mismatch. It starts with buffer semantics so the first compiler exercise can focus on lowering without also needing tensor bufferization and ownership passes.

Install a matching set of `mlir-opt`, `mlir-translate`, and `mlir-runner` tools. Some distributions provide version-suffixed names, such as `mlir-opt-20`; the helper discovers those. Package names and availability depend on the distribution. You can instead point to an isolated toolchain:

```bash
export MLIR_OPT=/path/to/llvm-20/bin/mlir-opt
export MLIR_TRANSLATE=/path/to/llvm-20/bin/mlir-translate
export MLIR_RUNNER=/path/to/llvm-20/bin/mlir-runner
python3 'Phase 5 - Advanced Topics and Specialization/Track F - AI Chip Design/Labs/run_mlir_lab.py' --output-dir /tmp/course-mlir-stages
```

Replace the three `/path/to` values with your installation paths. If the tools are already on PATH, omit those exports. The [runner](run_mlir_lab.py) preserves four stages when an output directory is supplied:

1. `01-verified.mlir`: parsed operations with valid shapes and types.
2. `02-loops.mlir`: Linalg expanded to scalar loads, arithmetic, stores, and `scf.for` loops.
3. `03-llvm-dialect.mlir`: explicit loop branches, pointer descriptors, and LLVM dialect operations.
4. `04-matmul.ll`: translated LLVM IR, followed by actual native CPU JIT execution.

The tested lowering flags are:

```text
--convert-linalg-to-loops --lower-affine --convert-scf-to-cf
--convert-arith-to-llvm --convert-index-to-llvm --finalize-memref-to-llvm
--convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts
```

Each conversion has a specific job. For example, converting structured control flow to `cf` does not itself convert those branches to the LLVM dialect; omitting `--convert-cf-to-llvm` leaves operations this runner cannot execute. These flags are a tested pipeline for this fixture, not a universal recipe for every MLIR module. A tensor program, vector contraction, GPU launch, or external function requires additional handling.

Run the real compiler checks:

```bash
python3 -m unittest discover -s 'Phase 5 - Advanced Topics and Specialization/Track F - AI Chip Design/Labs' -p test_mlir_lab.py -v
```

Three tests were executed successfully with MLIR 20.1.2: the valid program lowers/translates/runs; a changed A value reaches execution and fails the value check; and incompatible K dimensions fail verification before lowering. If tools are absent, the optional unittest suite clearly skips. A present compiler rejecting the fixture is a test failure, not a skip.

**Important distinction:** the CPU MLIR example has not applied the Python tiling schedule or emitted a systolic accelerator instruction. The two exercises expose different contracts for the same operation. Connecting them requires an implemented and verified lowering that preserves boundary, initialization, layout, and synchronization behavior.

## Finish with an engineering explanation

Keep one report containing the chosen shape, tile dimensions, input/accumulator sizes, calculated traffic, peak storage, numerical result, and tool version. Explain why the last K tile cannot be dropped and why a large tile can reduce traffic while exceeding scratchpad capacity. Include the negative shape or wrong-value test as evidence that your checker rejects a real error.

Then continue to the [programmable VLA accelerator lab](../Programmable%20VLA%20Inference%20Accelerator/Lab-Guide.md) to study a broader working set and deadline schedule, or the [RTL handshake lab](../../../Phase%204%20-%20Track%20A%20-%20Xilinx%20FPGA/1.%20Xilinx%20FPGA%20Development/labs/README.md) to implement valid/ready behavior under backpressure. The present systolic model assumes no stalls; a streaming RTL implementation must define what every operand, valid bit, and accumulator does when progress stops.

Primary references checked 2026-10-18: [Linalg operations](https://mlir.llvm.org/docs/Dialects/Linalg/), [structured Linalg tutorial](https://mlir.llvm.org/docs/Tutorials/transform/Ch0/), [bufferization and ownership](https://mlir.llvm.org/docs/Bufferization/), and [Vector dialect](https://mlir.llvm.org/docs/Dialects/Vector/).
