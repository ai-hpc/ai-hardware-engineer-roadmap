# Read and test a real engine without a GPU

This lab uses the **actual SparkInfer-K3 source** at commit `7689cc77e0e18d4b4f1937cf2a189dcd5a8d90f3`. You compile its CPU partition/metadata components and run its benchmark-policy tests. No model weights, CUDA toolkit, rented node, or network access during the tests is required.

The source checkout needs Git/network once; the tests need Python 3.10+ and a C++17 compiler. The course review used Python 3.12.3 and GCC 13.3 on Linux. Allocate roughly 100 MB for source and temporary binaries. The upstream top-level CMake project requires CUDA, so this exercise compiles only its explicitly CUDA-free components directly.

## Start with four small counterexamples

From the course repository root:

```bash
python3 -m unittest discover -v -s 'Phase 5 - Advanced Topics and Specialization/Track G - ML Systems Engineering/AI Inference Engineer 2026/Part 4 - Optimizing a Real Engine/labs'
```

These four tests reuse the [Part 1 scalar expert-combine fixture](../Part%201%20-%20Fundamentals/labs/README.md). They model the pinned kernel's skip predicate, **not** GPU execution or observed K3 routing.

Predict each result before reading the output:

| Input | Exact combine | Thresholded combine |
|---|---:|---:|
| 16 experts, each weight 1/16, each output 1; position 16383 | 1 | 1: depth gate inactive |
| Same inputs; position 16384, threshold 0.08 | 1 | 0: every weight is below threshold |
| Weights 0.01/0.99, outputs 1000/1; gate active | 10.99 | 0.99 |
| Weights 0.01/0.99, outputs 99/−1; gate active | 0 | −0.99 |

The answer is the lesson: small **weight mass** is not a guaranteed small **relative output error**. The actual source defaults the threshold to 0.08 and its depth gate to 16384. Byte comparisons at 8192 do not execute it. See [Lecture 04](Lecture-04.md) for the correct error bound and required real-prefix evaluation.

## Fetch the reviewed source

Use a new disposable directory, not an existing working checkout:

```bash
git clone --filter=blob:none --no-checkout https://github.com/gittensor-ai-lab/sparkinfer-k3.git /tmp/course-sparkinfer-k3
git -C /tmp/course-sparkinfer-k3 checkout --detach 7689cc77e0e18d4b4f1937cf2a189dcd5a8d90f3
```

If that directory already contains your work, choose another path and pass it to `--repo`. Inspect the pinned source before executing it; this runner is not a sandbox for arbitrary repositories.

Run from the course repository root:

```bash
python3 'Phase 5 - Advanced Topics and Specialization/Track G - ML Systems Engineering/AI Inference Engineer 2026/Part 4 - Optimizing a Real Engine/labs/run_upstream_cpu.py' \
  --repo /tmp/course-sparkinfer-k3 \
  --output /tmp/sparkinfer-cpu-report.json
```

The runner verifies the full commit and rejects tracked modifications. It compiles into a temporary directory, deletes binaries on exit, and writes commands, output, tool versions, and source hashes to the report. It runs these upstream components:

| Component | What it checks | What it cannot establish |
|---|---|---|
| Shard dimensions | Valid rank/shape combinations and tiled bands | GPU kernel indexing on real allocations |
| Backend selection | Capability and policy decisions | Actual peer/NCCL communication |
| Weight plan/residency | Tensor-name rules, block alignment, replication, offsets | Model fit on a particular GPU |
| K3 config/manifest | Required metadata, hybrid layers, multi-shard fixtures | Quality of downloaded model weights |
| Decode plan | Operation ordering and collective placement as data | CUDA stream/graph synchronization |
| Baseline plumbing | Pins, node/quant identifiers, missing shards, dry runs, policy wiring | GPU throughput or real-node isolation |
| Label policy | Accuracy rejection, significance and tier calculations | Whether the input measurement was truthful |

The native programs print thousands of internal assertions; these are **not thousands of independent GPU experiments**. Seven passing executables and two Python suites are the meaningful execution units. Passing source-consistency tests also does not validate the source's scientific claims: its legacy active-parameter label and marketing-unit memory arithmetic still require independent review.

## Work through one real failure

Read `runtime/tests/tp_weight_residency_cpu_test.cpp` and find the leading-dense `ffn_down` case. At TP=8, width 33792/8 = **4224 elements**. A 256-element quantization block does not divide 4224: 4224/256 = **16.5 blocks**. A column cut there would split packed data/scales. The planner must reject that split or choose a valid representation/policy; integer truncation is not a repair.

Next inspect the band-tiling checks. Each rank copying the right number of bytes is insufficient: two ranks can copy the same band while another band is absent. Global coverage and no overlap are independent invariants.

**Challenge:** in a separate scratch copy, deliberately change the shard offset or disable the block-alignment guard. Compile the affected upstream test using the command stored in your report. It must fail. Keep this mutation out of the clean pinned checkout; the runner deliberately rejects edits.

## If a command fails

- **Wrong revision or modified source:** use a fresh checkout at the full SHA. The refusal is intentional; it prevents results from silently describing another tree.
- **`g++` not found:** install your platform's C++ compiler or pass `--cxx` with a compatible compiler path. This exercise does not require NVCC.
- **An upstream assertion fails:** keep the JSON report, compiler version, failing test name, and stderr. Do not turn a failed source contract into an ignored warning.
- **No GPU:** that is expected for this lab. The GPU benches in the upstream README are a separate, much larger exercise.

## Finish with an evidence table

Write three rows: **executed here**, **reported by the author**, and **not established**. Place the seven native runs and policy tests in the first row. Put the published 60.17 tok/s in the second. Put real-prefix 128k accuracy, full-model GPU equivalence, sustained thermal behavior, and deployment p99 in the third unless you actually measure them.

The [validation record](labs/validation.json) includes source hashes and our actual results. Continue with [Lecture 02](Lecture-02.md), then [Lecture 03](Lecture-03.md). With a GPU later, reuse the same evidence discipline and add hardware-specific measurements; the CPU lab is a foundation, not a substitute for them.
