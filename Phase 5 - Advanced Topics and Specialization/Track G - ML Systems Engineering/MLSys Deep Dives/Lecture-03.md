# Lecture 03 — Separate compilation from runtime scheduling

[Previous: kernel languages](Lecture-02.md) · [Course index](README.md) · [Next: recurrent and hybrid models](Lecture-04.md)

A compiler transforms a program into executable operations. A runtime allocates
memory, schedules those operations, and manages requests and state. Some systems
bundle both. Keeping their jobs separate helps explain why a faster kernel may
not improve a whole application.

**Prerequisites:** the matrix-product contract from Lecture 02 and the timing
boundaries from Lecture 01. You will compare an actual compiled CPU graph with
an uncompiled reference, then reason about launch and scheduling overhead.
Allow 60–90 minutes plus the linked lab's environment setup.

## Follow one operation through the stack

Consider `y=relu(x@W+b)`. A graph representation describes its data dependencies.
Compiler passes may choose layouts, simplify shapes, fuse operations, or select
an implementation. Lowering translates the resulting operations into a form
that the backend can compile for a CPU or GPU. The runtime still needs the right
input shapes, dtypes, memory ownership and execution dependencies.

Fusion can avoid writing and rereading an intermediate tensor. It can also
increase register pressure or duplicate work. A graph with fewer nodes is not
by itself evidence of fewer bytes transferred or lower latency. Inspect the
lowered program and measure the relevant execution boundary.

In the [native TVM lab](../TVM%20Deep%20Dives/labs/README.md), run the Relax
matrix/bias/ReLU example and its odd-shape checks. It builds actual LLVM code and
exports/reloads a shared library. The pinned release uses `tvm.tirx` and
`tvm.s_tir`; moving documentation can show a different namespace. Follow the
lab's pin as a complete environment rather than mixing examples across versions.
Its unsuccessful MetaSchedule experiment is recorded as unsuccessful. Producing
a tuning log does not establish that any valid schedule was found.

## Four different ways to choose implementations

A hand-written schedule makes decisions explicitly. A heuristic selects a
reasonable implementation from known rules. Search compiles and measures
candidates under a time budget. A library dispatches among specialized kernels.
Real compilers combine these approaches. Search can fail, overfit to a shape,
or cost more than the saved execution time for a short-lived workload.

If compilation takes 20 seconds and saves 0.2 milliseconds per call, the simple
break-even count is `20/0.0002=100,000` calls. That illustrative calculation
assumes the saving persists and ignores other startup costs. A server that
reuses the artifact may cross that threshold; a one-shot script may not.

## Understand the runtime's timeline

Suppose a trace shows 100 kernels, each taking 5 microseconds, with an additional
3 microseconds of nonoverlapping gap per kernel. The illustrated critical path
is 800 microseconds. Eliminating every gap would reduce it to 500 microseconds,
a maximum speedup of 1.6× under this model. It would not make the kernels execute
faster. Overlap in a real trace changes the arithmetic, so do not simply sum
all displayed durations from different streams.

CUDA graphs can reduce repeated host submission overhead for captured work.
They do not remove mathematical dependencies, guarantee bitwise numerical
identity, or make every dynamic allocation and shape change capturable.
Persistent execution keeps scheduling work on the device and can overlap
fine-grained tasks. It also needs correct dependency tracking, resource limits,
progress guarantees and request-state ownership.

The [TileRT revision checked for this course](https://github.com/tile-ai/TileRT/blob/0ea19371f120977761bfa0cf6a2d08415d8fce0e/README.md)
describes a tile-based runtime and public release `0.1.6.post2` with specific
CUDA and ROCm backends. The wheel has exact PyTorch ABI requirements, and its
README documents a NVIDIA CLI argument bug with a Python-API workaround.
Source wrappers, installed backend binaries and the underlying compiler are
different artifacts. A public repository does not establish that every backend
component can be rebuilt from that repository.

## Compare toolchains by their boundary

TVM has its own Relax and tensor-level infrastructure. IREE uses MLIR-based
compilation. Mojo is a programming language with a compiler; MAX includes model
and serving components. Sharing some interfaces does not make every system the
same compiler or imply that their artifacts are interchangeable.

The [pinned Modular repository](https://github.com/modular/modular/blob/9d6a4ca8fc378f2cd5f1f70479bac27b8b3c2615/README.md)
identifies the Mojo compiler as available in source since August 2026 and lists
other open components. MAX distribution has a separate license boundary. Check
the relevant component and target instead of describing the entire ecosystem
with one “open” or “closed” label.

## Exercise: classify a proposed speedup

A patch reduces a matrix kernel by 10%, but the matrix accounts for 40% of a
nonoverlapping request timeline. What improvement can it explain on its own?

<details markdown="1">
<summary>Answer</summary>

Normalized time becomes `0.6+0.4×0.9=0.96`, a 4% time reduction or about 1.042×
speedup. A larger measured gain needs another explanation, such as changed
launch overhead, synchronization, cache state, workload or measurement error.

</details>

**Exit check:** retain a CPU reference comparison, exported artifact, compile
cost and warmed execution boundary. Name which runtime behavior was not tested.
The GPU runtime descriptions here are source-reviewed on October 18, 2026;
TileRT, Mojo/MAX and CUDA graph execution were not run for this lesson.
