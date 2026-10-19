# Lecture 02 — Choose a kernel language by the work it exposes

[Previous: measurement](Lecture-01.md) · [Course index](README.md) · [Next: compilers and runtimes](Lecture-03.md)

A GPU kernel describes work performed by many threads. A kernel language decides
which details you express directly and which the compiler chooses. A **tile** is
a block of data processed together; it is not automatically a hardware
instruction, a cache line, or a group of independent threads.

**Prerequisites:** matrix dimensions, row-major indexing, and the distinction
between global and shared memory. Your goal is to choose a language for one
operator and state which correctness and performance questions remain. Allow
60–90 minutes, using a CPU reference if you have no GPU.

## Start with a tiny matrix product

For `C=A@B`, let A have shape `[5,7]` and B shape `[7,3]`. The result has shape
`[5,3]`; each result sums seven products. A tile of four output rows and four
output columns needs two row tiles and one column tile. The second row tile has
only one valid row, and every tile has one invalid column.

If the reduction tile has width four, its second iteration contains three valid
K positions and one invalid position. Invalid loads must contribute zero to the
sum and invalid stores must not write beyond C. A program that works only when
all dimensions divide the tile size has not implemented general matrix
multiplication.

**Work by hand:** how many multiply-add contributions belong to the mathematical
result? How many lanes exist in the padded tile computation?

<details markdown="1">
<summary>Answer</summary>

The useful result contains `5×3×7=105` products. The padded logical iteration
space contains `8×4×8=256` product positions. This does not prove the hardware
executes exactly 256 instructions: masking, lowering and Tensor Core shapes
change the actual instruction sequence. It does show why shape padding matters.

</details>

## What the common programming systems expose

| System | Typical control you express | Questions to verify |
|---|---|---|
| CUDA C++ | Threads, blocks, memory operations and synchronization | Which instructions, layouts and async operations does the target support? |
| CUTLASS / CuTe | Structured layouts and reusable matrix/kernel components | Does the selected architecture/template match the operand layout and dtype? |
| Triton | Blocks of tensor values and program-level scheduling | How are masks, strides, warps and shared resources lowered? |
| TileLang | Tile computations and explicit scheduling features | Which backend, pass pipeline and target were used? |
| cuTile Python | Tile operations lowered through NVIDIA Tile IR | Does the installed compiler support this exact GPU generation? |
| Mojo GPU programming | A language and GPU abstractions integrated with its toolchain | Which compiler, runtime components and target backends are available? |

These are overlapping choices, not a universal ranking from easy to fast. A
high-level implementation may compile into excellent instructions; explicit
low-level control can also encode an inefficient schedule. Compare the same
operator, shapes, dtype, layout and error tolerance.

The [cuTile source pinned October 16](https://github.com/NVIDIA/cutile-python/blob/770b4937164f7cf98b5e19b0f55d688854b8ada5/README.md)
describes `cuda.tile`, the `cuda-tile` package, and Tile IR tooling. Its README
still describes a narrower `tileiras` 13.2 target set. The [current quickstart](https://docs.nvidia.com/cuda/cutile-python/quickstart.html)
lists compute capabilities 8.x, 9.x, 10.x, 11.x and 12.x, driver r580 or later,
and matching major/minor versions for the Tile IR compiler, NVCC and NVVM
packages. Record that documentation difference rather than repeating the older
README's “Hopper forthcoming” statement as a current limitation. Match the
actual compiler, driver and GPU before promising a path on a Jetson or server.

## Layout and synchronization are still your problem

A tensor can have the expected shape and unexpected strides. A transposed view
may share storage while changing its index-to-address map. Write down which
inputs your kernel accepts: contiguous only, arbitrary strides, or a constrained
packed layout. Reject unsupported cases explicitly.

Likewise, async copying does not make an input ready immediately. Name who
produces a tile, which operation establishes completion, and when the storage
can be reused. A barrier that coordinates only one block cannot synchronize
all blocks in a grid. Persistent kernels still need a correct dependency and
resource protocol.

## Run a reference before choosing a fast implementation

The [CPU compiler lab](../../../Phase%204%20-%20Track%20C%20-%20DL%20Inference%20Optimization/00%20-%20CPU%20Compiler%20Lab/Guide.md)
and [native TVM lab](../TVM%20Deep%20Dives/labs/README.md) give actual compiled
CPU exercises with shape and tail checks. They establish a useful workflow:
reference output → transformed program → comparison → inspect generated code.
CPU correctness is valuable evidence, but it does not validate a GPU lowering.

For an attention operator, use the [FlashAttention CPU lab](../../../Phase%204%20-%20Track%20C%20-%20DL%20Inference%20Optimization/02%20-%20Kernel%20Engineering/FlashAttention%20Course/CPU-Attention-Lab.md).
Its masks, empty rows and rectangular shapes expose errors before GPU tuning.
Then choose one supported GPU backend as an optional extension, record compile
and warmup separately, and compare with the same reference.

**Exit check:** explain one tail, one stride and one synchronization failure.
Choose a language for the operator with a reason tied to supported hardware and
required control, not a promised speedup. Current tool claims checked October
18, 2026; [source revisions](sources/verification-2026-10-18.json) are retained.
