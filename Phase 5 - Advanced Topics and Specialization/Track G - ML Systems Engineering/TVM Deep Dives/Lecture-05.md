# Lecture 05 — Ship a compiled artifact with its contracts

A compiled function is only part of a deployment. The consumer needs a compatible runtime/ABI, target instructions, tensor contracts, parameters, and lifecycle rules. The [CPU lab](labs/README.md) exports a real Relax executable to a shared library, reloads it, and checks outputs. That is the concrete deployment result for this course.

## What the CPU export proves

The test compiles `relu(XW+b)` for LLVM, calls `export_library`, loads the resulting `.so` with `tvm.runtime.load_module`, creates a Relax virtual machine, and runs the same numerical reference check. A host C/C++ linker is required for the library step. Temporary output files are removed after the test.

It proves that this pinned TVM build can package and reload this CPU program in the tested environment. It does not prove the file is portable across arbitrary operating systems, CPUs, TVM versions, or C++ ABIs. The export includes no pretrained weights because this model receives W and b as inputs. A model with bound weights needs a different packaging account.

See the [current export/load tutorial](https://tvm.apache.org/docs/how_to/tutorials/export_and_load_executable.html) for development APIs. Match that documentation to your compiler snapshot; mixing current Relax output with a legacy GraphExecutor loading example is not a valid deployment recipe.

## Write a deployment manifest

Record compiler/package and runtime revisions, target triple/CPU features, build/linker options, library hash, function names, shapes/dtypes/layouts, parameters and their hashes, and required external libraries. Include numerical tolerance, warmup/startup behavior, memory budget, and error policy.

If compilation targets an ISA newer than the deployment CPU, a valid shared library can still fail with an illegal instruction. If a required library is missing, loading may fail before inference. If a tensor is on the wrong device or has unsupported strides, a call may reject it or need an explicit conversion. Test those boundaries deliberately.

DLPack can exchange tensor storage without a mandatory copy, but ownership, lifetime, device, stream synchronization, and supported layout remain contracts. The tail test's NumPy-backed view stays alive during execution. Zero-copy is not permission to free or mutate a buffer while another runtime uses it.

## Runtime choices are version-specific

Relax VM is the path exercised here. Relay GraphExecutor, older AOT executors, AutoTVM, and microTVM tutorials describe historically important workflows. Use an archived compatible source/toolchain if studying them; do not advertise an untested current microcontroller deployment by pasting older build flags into a new wheel.

An MCU deployment additionally needs a supported board/runtime, cross compiler, linker script, memory placement, operator coverage, quantization contract, and physical flashing/debug evidence. Static allocation can simplify budgeting, but stack, scratch, DMA buffers, interrupts, and runtime metadata still count. A CPU compilation is not an MCU timing result.

## MLC-LLM is a separate integration exercise

MLC uses TVM-related compiler infrastructure to build supported LLM paths for several targets. Follow its [compiler installation guidance](https://llm.mlc.ai/docs/install/tvm.html) and pin its own package/source combination. An arbitrary latest Apache TVM wheel is not guaranteed compatible with an MLC release or target backend.

A full MLC artifact includes the model architecture/configuration, converted weights/quantization, tokenizer/chat template, target library, runtime, and cache/sampling settings. Test short and long prefixes, incremental decoding, context limits, and the chosen quality criteria. Runtime support for a backend does not establish that every model/format runs efficiently there.

## Benchmark the delivered boundary

Measure loading, initialization, first call, steady-state calls, and peak memory separately. Include required copies and preprocessing/postprocessing in application latency. Keep a kernel-only measurement too when it helps diagnosis, but label its boundaries. For async accelerators, use appropriate completion timing rather than only the host enqueue duration.

A useful deployment report contains both the successful reference comparison and a failure case: wrong shape, missing dependency, unsupported target, or invalid input handled according to policy. Do not require an arbitrary ±10% reproduction band without accounting for measurement noise and machine differences.

**Exit exercise.** Hand another learner the manifest, source, requirements, and tests. They should be able to build/reload the CPU module, explain the 37-element tail, and identify what remains untested for a GPU or embedded target. A measured limitation is part of a complete report.

Return to the [course overview](README.md) or continue into [AI inference engineering](../AI%20Inference%20Engineer%202026/README.md).
