# Test the local runtime components

The CPU tests are the reproducible starting point. This checkout does not have a passing pretrained-model/Jetson acceptance record. A separate upstream project's benchmarks and release names are not local test results.

## CPU build and execution

From the course repository root:

```bash
cmake -S Projects/jetson-llm-runtime -B /tmp/jllm-course -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/jllm-course -j2
ctest --test-dir /tmp/jllm-course --output-on-failure
```

This builds actual C++17 code, without CUDA headers/libraries. The six CTest entries run native sampler, memory-planning, allocator-control, attention-oracle and decode-scratch tests plus a Python-driven set of **18 native integration tests**. The latter also invokes the sampler and validates native parser/tokenizer behavior using independently encoded binary fixtures. It checks every truncated prefix of one complete tensor fixture and a deterministic malformed-input corpus.

The test runner keeps assertions active by using explicit exceptions in the CPU checks. The optional legacy GPU tests receive `-UNDEBUG`, so Release configuration cannot silently remove their `assert` checks.

The allocator test compiles the production pool/cache control code against a test-only host-memory shim. It checks ownership, capacity, indexing, overflow and cleanup, including allocation failure. This target does not call a CUDA runtime or execute on a GPU; never treat its result as managed-memory or device-coherence validation. The shim include path is private to this target.

## Sanitizers

```bash
cmake -S Projects/jetson-llm-runtime -B /tmp/jllm-course-sanitize -DCMAKE_BUILD_TYPE=Debug -DJLLM_SANITIZE=ON
cmake --build /tmp/jllm-course-sanitize -j2
ctest --test-dir /tmp/jllm-course-sanitize --output-on-failure
```

**Actual review host:** Ubuntu 24.04.4 x86-64, GCC 13.3.0, Python 3.12.3. The [validation record](labs/validation.json) records the Release and ASan/UBSan results, tested sources and boundaries. Sanitizer success covers these executions; it is not a general proof of parser or engine correctness.

## Component timings and script behavior

`scripts/test_plan.sh [build-directory]` runs the CPU configure/build/CTest workflow. `scripts/bench.sh [build-directory]` emits reproducible parser/sampler microbenchmark JSON using a temporary fixture. Both propagate failures; neither downloads a model or changes device power/OS settings. See the [timing exercise](labs/README.md#8-measure-a-component-without-mislabeling-it).

`profile.sh <new-output-prefix> -- <executable> [arguments...]` is an optional Nsight Systems wrapper. The installed profiler and selected workload must already work. It does not select a model, suppress a failed workload, or claim that a captured trace proves correctness. Trace capture was not executed in this review.

## Interpret the fixtures

The [lab](labs/README.md) generates a 1 KiB-scale GGUF file with zero-valued embedding data and a custom toy vocabulary. It intentionally omits transformer layer weights. Parsing it proves that the supported layout is accepted. It does not constitute a functional model or a pretrained tokenizer test.

The parser supports a bounded little-endian GGUF v3 profile. It rejects unknown metadata/tensor types, nested arrays, excessive counts/lengths, duplicate names/keys, invalid UTF-8, malformed metadata keys, tensor names beyond 64 UTF-8 bytes, unsupported alignment/ranks, nonintegral quantization blocks, overlapping tensors and out-of-file payloads. Unsupported does not necessarily mean invalid under every version of the full GGUF ecosystem; it means this teaching reader cannot process the input faithfully.

## Optional CUDA source build

On a compatible Jetson development environment:

```bash
cmake -S Projects/jetson-llm-runtime -B /tmp/jllm-jetson -DJLLM_BUILD_JETSON=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build /tmp/jllm-jetson -j2
ctest --test-dir /tmp/jllm-jetson --output-on-failure
```

No unused `GGML_CUDA` option is needed: this project has its own CMake targets. Choose the CUDA toolkit compatible with the BSP; do not install a wildcard set of CUDA versions. Building is a separate milestone from device execution. The validation record enumerates C++/CUDA translation units compiled individually with nvcc 13.4.92 and `-arch=sm_87` on an x86-64 host. This is compile-only evidence: the Jetson executable was not linked or run. The GPU CTest targets use skip code 77 for absent devices/drivers, and report other enumeration errors as failures.

The older kernel checks remain limited smoke tests. The new
[attention parity harness](labs/Attention-Layout-Lab.md) has independent numerical
references, negative-score and GQA fixtures, partial tiles, poisoned output and
checked CUDA errors. It compiled, but its local execution returned **77**: GPU
parity was not run. The CPU oracle and scratch checks are separate executed
evidence. Extend comparable references to the remaining kernels, including
nonzero RoPE positions and packed quantization layouts, before model acceptance.

## Pretrained-model acceptance remains open

A real model needs all of these to agree: architecture operations and metadata, tensor encodings, tokenizer/pretokenization/merges, special tokens and chat template, cache positions/scales, logits, sampling policy, and stopping rules. Current local tokenizer loading rejects ordinary model tokenizer profiles rather than treating greedy string matching as BPE.

Use a pinned independent reference to compare exact token IDs first, then per-layer/whole-model values under stated tolerances, then greedy generation and task-level quality. Cover Unicode, whitespace, special tokens, mixed tensor types, tied embeddings, context limits, repeated calls, and cold/warm state. A coherent sentence alone is insufficient evidence.

For timing, define prompt/output lengths, prefill boundary, time to first token, decode intervals, concurrency, queue delay, clocks, power mode, temperature, model/input identities and repeated-run statistics. The experimental engine's prefill timing excludes the final prompt-token forward pass; it must not be reported as a full prefill or TTFT benchmark.
