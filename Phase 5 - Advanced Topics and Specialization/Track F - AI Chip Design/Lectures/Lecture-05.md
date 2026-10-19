# Lecture 5: ML-to-hardware compilation pipelines

A working compiler connects a model to kernels, memory, synchronization, and a runtime. Producing plausible assembly is only one part. This lecture compares TVM, MLIR-based systems, and tinygrad by following the same small operation: `Y = ReLU(A @ B + bias)`.

By the end, you should be able to identify where shapes are checked, where kernels are grouped, where storage is assigned, and where device-specific instructions are selected. Begin with the [CPU tiling and lowering lab](../Labs/Tiling-and-Dataflow.md); it provides a complete executed example before the larger compiler architectures below.

## 1. Follow one operation through the stack

Let A have shape 5×3, B have shape 3×7, and bias have shape 7. The output has shape 5×7. Broadcasting bias means adding `bias[j]` to each row's output column j; it does not mean allocating five independent bias vectors.

```text
Model operation: Y[i,j] = max(sum_k A[i,k]*B[k,j] + bias[j], 0)
    ↓ frontend import and semantic checks
Graph: matmul → bias add → activation
    ↓ partitioning and legal fusion
Kernels or library calls, with explicit inputs and outputs
    ↓ scheduling and memory planning
Tiles, layouts, loops, buffers, dependencies
    ↓ target lowering and compilation
CPU instructions, device kernels, or accelerator command streams
    ↓ runtime
Allocate → transfer → dispatch → wait → consume result
```

At each arrow, ask what facts must remain true. The K reduction must include all three elements. Bias is added once per output. ReLU comes after the complete reduction. A 2×4 tile needs a ragged final row and column. Buffer ownership must survive asynchronous execution. These are correctness requirements before any performance comparison.

There is no universal one-pass order: a compiler can interleave scheduling, fusion, layout selection, and lowering. A library call may bypass a compiler's custom kernel generator. A model may mix generated kernels, vendor libraries, and CPU fallback.

## 2. Apache TVM: graph programs and tensor schedules

TVM separates what a tensor program computes from how it executes. Its current documentation presents **Relax** for graph-level programs, tensor-level IR and schedules, and explicit runtime/external-library interfaces. The documentation checked on **2026-10-18** includes `tvm_tirx` and `tvm.s_tir` interfaces; older Relay, `te.create_schedule`, AutoTVM, and Ansor tutorials belong to particular releases. Use examples matching an installed release or source commit. [TVM IRModule tutorial](https://tvm.apache.org/docs/get_started/tutorials/ir_module.html).

An IRModule can contain graph functions, tensor programs, and calls to externally implemented operations. That distinction matters: seeing a matmul in the input graph does not prove TVM generated the inner GEMM kernel itself. Inspect the lowered call and runtime module to identify which implementation executes.

A schedule might split the M/N/K loops, reorder them, copy tiles to a scratchpad, vectorize a suitable dimension, or bind work to GPU threads. These transformations have legality conditions. A vectorized dimension needs supported access and dependence behavior; a reduction transformation must preserve the chosen numerical contract. A tile must fit all simultaneously live storage, not just its weight matrix.

**Worked schedule decision:** in the course lab, changing TK from one to three while holding the 2×4 output tile increases modeled resident bytes from 44 to 68. It does not change the 326 logical bytes transferred because C stays resident across K chunks. It reduces the number of reduction chunks. A hardware design that spills C or double-buffers A/B has a different cost model. This is the sort of information a useful scheduler needs.

Search-based tuning explores permitted schedules under a budget. It can find a better candidate than the starting schedule; it does not establish a global optimum. Record target model, input shapes, dtypes, search space, tuning budget, correctness tolerance, and measurement noise. Compilation and tuning cost may dominate a workload run only a few times. [TVM scheduling and tuning documentation](https://tvm.apache.org/docs/deep_dive/tensor_ir/index.html).

For a new accelerator, first decide whether to generate its instructions, call a runtime/library, or partition supported subgraphs to an external compiler. Each route needs operator coverage, layouts, memory and synchronization rules, and a fallback policy. Changing a target string does not automatically supply those pieces. Microcontroller deployment also needs a runtime, memory planning, supported operators, and board integration; choosing a C target alone does not guarantee CMSIS-NN calls or a particular SRAM footprint.

**Optional investigation:** follow the current official TVM import tutorial with a small linear layer. Save the exact TVM/PyTorch versions and imported module, then identify a graph function, a tensor computation, and an external call if present. This lecture's TVM integration was source-reviewed, not executed during the CPU validation session.

## 3. MLIR-based systems solve different parts of the problem

MLIR is infrastructure for extensible operations and transformations, not one complete compiler with one fixed pipeline. A `linalg.matmul` operation preserves useful structure, but a project must still choose schedules, implement supported conversions, and connect the result to a runtime.

**torch-mlir** provides import and lowering paths for PyTorch computations. Its APIs and supported capture mechanisms depend on the project revision. Exporting a graph, lowering Torch operations, bufferizing tensors, generating device code, and executing that code are separate steps. Consult the [torch-mlir project examples](https://github.com/llvm/torch-mlir) for a pinned revision rather than mixing an old `torch_mlir.compile` snippet with current PyTorch packages.

**IREE** combines a compiler and runtime. Its architecture includes workload formation, asynchronous resource management, hardware abstraction, and target executables. This makes buffer lifetimes and dispatch dependencies part of the compilation story, rather than leaving every application to reconstruct them. Supported input forms, target backends, and runtime drivers must match the selected build. [IREE developer overview](https://iree.dev/developers/general/developer-overview/).

Consider an asynchronous command that reads A and writes C. The runtime must retain A until the read finishes, prevent a consumer from reading unfinished C, and avoid recycling C's storage while that consumer is active. Correct arithmetic in the kernel cannot compensate for a premature free or missing event dependency.

**Triton** is a kernel programming/compiler system. A frontend such as a framework compiler may choose to generate Triton kernels, but a standalone Triton kernel does not import and execute an arbitrary model by itself. Its IR stages expose block operations, layouts, and target-specific lowering. A `tl.dot` does not promise a particular matrix instruction on every GPU: target generation, operand precision, layout, launch configuration, and compiler decisions all matter.

### A safe boundary pattern for a Triton tile

This is a **kernel fragment**, not a complete launchable program. `rm`, `rn`, and `rk` are row, column, and reduction indices; A and B are contiguous row-major buffers. It illustrates the masks that a ragged implementation must preserve:

```python
# Inside a Triton kernel, for one K chunk:
a = tl.load(A + rm[:, None] * K + rk[None, :],
            mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
b = tl.load(B + rk[:, None] * N + rn[None, :],
            mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)
acc = tl.dot(a, b, acc)
# After ALL K chunks and the final bias/activation:
tl.store(C + rm[:, None] * N + rn[None, :], acc,
         mask=(rm[:, None] < M) & (rn[None, :] < N))
```

Zero-filled invalid K elements preserve the reduction. The final store mask protects the output edges. A complete kernel also needs imports, valid block sizes, accumulator initialization, strides or an explicit contiguous-layout contract, a K loop, launch configuration, dtype policy, and a host reference check. Start from the [official complete Triton matmul tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html); the fragment above was not executed on a GPU in this review.

**Question:** is an unmasked load safe if its result is never stored? **Answer:** no. An invalid read is already an error; masking only the output does not protect the inputs.

## 4. tinygrad: inspect a smaller compiler stack

tinygrad exposes tensor operations, UOp graphs, scheduling, kernel optimization, rendering, and device execution in a relatively compact codebase. Its internal names and backend organization evolve. `LazyBuffer` and “exactly 12 UOps” describe older or simplified accounts; neither is a stable current architecture contract. Use the [developer overview](https://docs.tinygrad.org/developer/developer/) and [layout documentation](https://docs.tinygrad.org/developer/layout/) with the pinned source revision.

The useful conceptual sequence is:

```text
tensor expressions and UOp graph
    → graph rewrites and scheduling
    → kernel optimization and lowering
    → program rendering/compilation
    → buffer allocation and execution
```

A UOp graph is not identical to a linear instruction list. Linearization is a later representation choice. Scheduling decides kernel boundaries; optional BEAM search explores kernel implementation choices. Neither guarantees that matmul plus ReLU becomes one kernel, that the selected tile is globally optimal, or that all backends follow the same compilation route.

Use the existing [CPU compiler lab](../../../Phase%204%20-%20Track%20C%20-%20DL%20Inference%20Optimization/00%20-%20CPU%20Compiler%20Lab/Guide.md) for the pinned, executed tinygrad example and saved evidence. Follow it into the [tinygrad deep dive](../../../Phase%204%20-%20Track%20C%20-%20DL%20Inference%20Optimization/06%20-%20tinygrad%20Deep%20Dive/Guide.md). That sequence includes numerical checks and device selection, so you can interpret a real trace instead of guessing backend names from an old tutorial.

A new backend needs more than a renderer. It needs allocation, transfers, a compiler or binary loader, kernel invocation, synchronization, supported dtypes/operations, and error handling. A short implementation can be valuable for exploration, but line count is not evidence of coverage, robustness, or performance. Avoid comparing compilers with unsourced “percentage of vendor peak” tables.

## 5. Choose an integration boundary by evidence

Instead of ranking compilers by hardware brand, compare the work each boundary requires:

| Integration route | What you must establish |
|---|---|
| Call an existing vendor library | Supported shapes/dtypes, layout, stream behavior, workspace, numerical contract |
| Generate a custom GPU kernel | Legal operations, tails, synchronization, resource use, target compatibility |
| Compile a graph with a supported stack | Operator coverage, capture constraints, fallback, runtime and deployment support |
| Add a custom accelerator backend | Instruction/runtime contract, memory plan, dependencies, numerical and failure tests |
| Emit C for a microcontroller | Bounded memory, supported kernels, cross toolchain, board runtime, measured deadlines |

A supported backend can still be unsuitable for a particular operator or latency requirement. Evaluate a small representative slice before importing a large model. Include unsupported cases in the report so fallback costs remain visible.

For a hypothetical 32×32 INT8 array, answer these questions before defining a dialect operation: where do INT32 partial sums live; what happens at K tails; is DMA completion explicit; how are strides expressed; what happens when a command fails; can a CPU consumer observe an incomplete output? Your IR must represent enough information to implement those answers.

**Worked memory check:** a 32×32×64 tile uses 2,048 bytes for A, 2,048 bytes for B, and 4,096 bytes for an INT32 C accumulator. That is 8,192 bytes before padding, DMA metadata, or other state. Double-buffering A and B makes it 12,288 bytes. Increasing K alone does not enlarge C, but does enlarge the staged inputs. If the architecture stores accumulators in a separate memory, budget each bank separately rather than reporting only a combined total.

## 6. Build one reviewable vertical slice

1. Run the [functional tiling and real MLIR CPU lab](../Labs/Tiling-and-Dataflow.md). Keep both the successful result and a deliberately rejected wrong value or shape.
2. Trace the same operation through one additional compiler using its pinned course lab or current official tutorial. Identify the actual kernel/library and runtime path.
3. State the supported shapes, layouts, dtypes, aliasing, and numerical tolerance. Include empty or ragged cases where the interface allows them.
4. Separate compilation/tuning time, first execution, steady-state kernel time, transfers, and end-to-end completion. A host timer around an asynchronous launch measures submission unless it includes the relevant completion boundary.
5. Explain one failed optimization: a tile exceeding storage, a misplaced activation, a layout mismatch, or a missing dependency. Show the test that catches it.

For a robotics workload, include observation-to-action delay and missed-deadline behavior in addition to kernel throughput. A faster average kernel can still make a worse controller if it adds batching delay or unpredictable compilation pauses. This connects compiler decisions to the [programmable VLA accelerator course](../Programmable%20VLA%20Inference%20Accelerator/README.md).

The required artifact is a short compatibility and evidence report: exact versions, target, supported operator contract, generated representation, correctness checks, measured quantities, and limitations. Optional vendor integrations above are documented study paths; the course's CPU labs identify what was actually executed.
