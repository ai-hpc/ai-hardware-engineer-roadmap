# Lecture 10 — Capstone: preserve a contract while changing an implementation

**Course:** [FlashAttention](Guide.md) · **Reviewed:** 18 October 2026

**Goal:** produce a small, reproducible engineering change. **Artifact:** a patch, correctness evidence and a report that clearly separates CPU results, source review and optional GPU measurements.

## 1. Choose an achievable scope

| Route | Concrete change | Required evidence |
|---|---|---|
| CPU mathematical path | Add a documented sliding-window mask to the tiled reference | Explicit allowed-pair examples, dense/tiled forward comparison, gradient tests, empty/tail cases |
| CPU inference path | Add a bounded cache append plus partial-page fixture | Full-prefill versus incremental agreement, capacity/position failures, no double rotation |
| GPU kernel path | Modify one supported tile/dispatch/mask path in a pinned implementation | Same correctness contract, GPU memory/synchronization checks, controlled before/after timing |
| API/documentation path | Correct a reproducible interface/shape/mask defect | Minimal failing case, corrected behavior and exact version boundary |

The CPU routes are complete learning outcomes. They do not claim that a Python loop is an optimized FlashAttention GPU kernel. Publishing an upstream PR is optional; a reviewable local patch is enough.

## 2. Write the acceptance rule first

For the window-mask route, define positions before code. With lower-right alignment, query i has aligned position `i + S − L`; a causal left window of width W permits keys from `position−W+1` through `position`, clipped to valid keys. State whether W counts the current token. A different convention can also be valid, but all implementations and tests must agree.

Use at least:

- A square sequence and a rectangular decode sequence.
- A query longer than available keys, producing fully masked rows under the chosen alignment.
- Tile tails and tile size one.
- A wrong mask/alignment case that the test actually rejects.
- Float64 forward and gradient checks, then one lower-precision comparison on identical quantized inputs.

Do not call a passing finite fixture a proof over all tensors. The derivation establishes the mathematical invariant; tests catch implementation defects over declared coverage.

## 3. Run the baseline before the change

Use [the CPU lab](CPU-Attention-Lab.md) and retain its test/report files. Freeze the input seeds and comparison tolerances. Make one change and rerun the affected checks. If a new requirement changes the semantics, update the oracle and explain why; do not quietly weaken thresholds to turn a failure green.

For a GPU extension, use a workload that fits available memory. A float64 dense score oracle at 64k length may itself require tens of GiB per head; compare smaller cases or selected rows with a justified reference strategy. Do not allocate a huge oracle just because the fused candidate fits.

## 4. Keep performance claims tied to their boundary

CPU reference timing can reveal Python/array overhead, but cannot predict CUDA performance. GPU kernel timing excludes tokenizer, model layers and serving queues unless deliberately included. Full-model tokens/s requires the full model path and request workload. Time-to-first-token includes more than an attention prefill call.

For a kernel report, collect warmups, repeated measurements, a spread statistic, device/runtime versions, exact shapes and useful FLOP convention. Distinguish requested backend from observed executed kernel. If a supported backend is unavailable, report the reason instead of relabeling fallback execution.

A slower change can still complete the course when the evidence explains why. No arbitrary target speedup, “within 20% of upstream,” or universal reproducibility percentage is required.

## 5. Submit a report another learner can reproduce

```text
Question and baseline:
Input/mask/position/precision contract:
Revision, environment and exact commands:
One implementation change:
Correctness tests, thresholds and failed cases:
Measured results and timing boundary:
Source-reviewed or unexecuted extensions:
Limitations and next experiment:
```

Include the patch, raw JSON/CSV output and one figure only if it helps explain the result. Keep a numerical counterexample or mutation that demonstrates your checks detect the intended bug.

**Example conclusion:** “The added window mask matches the dense float64 reference for these shapes and gradients; a lower-right/upper-left mutation fails the decode fixture. CPU execution was measured. No GPU throughput or full-model quality is claimed.”

Continue to [the compiler stack](../../03%20-%20Compiler%20Stack/Guide.md) with the exact mask/layout/precision contract your backend must preserve. The meaningful artifact is a tested boundary, not a guessed benchmark table.
