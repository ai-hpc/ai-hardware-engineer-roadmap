# Lecture 10 — Capstone: an evidence-based deployment decision

**Prerequisites:** Lectures 01–09. **Time:** 4–6 hours of integration and review, plus bounded experiments. **Outcome:** recommend, reject, or defer a specific DeepSeek V4.1 Flash deployment with reproducible evidence.

## Project brief

Your team needs a service for input-heavy engineering conversations, with a smaller interactive workload sharing capacity. Choose a checkpoint and node configuration. Define input/output distributions, image usage, quality budgets, TTFT/TPOT targets, and a maximum experiment cost before running.

There is no required speedup. A justified rejection of an unsupported or quality-regressing conversion is better engineering than a fast but incorrect demonstration.

## Deliverables

| Artifact | Minimum content |
|---|---|
| Identity manifest | Model and runtime revisions, conversion provenance, protocol and image preprocessing |
| Compatibility report | Architecture, serialization, loader, kernel, and hardware evidence |
| Capacity model | Per-rank weights/state/workspace, host RAM and disk, startup peak |
| Baseline | Launch, smoke tests, raw request records, quality result |
| Optimization record | At least three hypotheses, including a failed or neutral one if observed |
| Load report | Three repeated runs, latency distributions, failures, goodput, and memory |
| Quality report | Held-out paired results and examined regressions |
| Decision | Recommended operating envelope, unsupported features, limitations, reproduction steps |

Do not fabricate neutral experiments to fill a quota; document what actually happened.

## Cost worked example

For an illustrative node costing C dollars/hour and useful throughput G output tokens/second:

```text
dollars_per_million_good_output_tokens = C * 1_000_000 / (3600 * G)
```

If G is zero, cost per useful token is undefined/infinite, not zero. Record whether C includes idle time, storage, networking, and operational costs. Keep input-heavy and output-heavy workload accounting separate. Use an actual quoted price only with its date and source.

## Review sequence

1. Reproduce the metadata audit from a clean directory.
2. Trace the chosen loader through one expert and one unchanged component.
3. Explain the largest capacity-model discrepancy.
4. Re-run a quality negative control.
5. Recompute metrics from raw records using the supplied summarizer or an equally specified method.
6. Explain why the best configuration changes under a different traffic mix.
7. Present a one-page decision with uncertainty and an explicit rollout/rollback criterion.

A reviewer should be able to reject your recommendation without disputing what you measured. That is the point of preserving raw evidence.

## Analysis-only completion

Without the required node or a validated loader, submit the manifest, memory model, format exercises, exact support gap, and a ready-to-run experiment design. Clearly label GPU and full-model fields **not run**. This earns the analysis milestone; the deployment milestone remains open.

## Self-check and rubric

Use the [roadmap rubric](../../../../Learning-Guide.md#completion-rubric). In addition, the deployment milestone requires actual measured quality and performance for the same checkpoint/runtime tuple. Passing CPU fixtures cannot substitute for those results.

Explain these three distinctions without notes:

- active computation versus resident storage;
- checkpoint-format support versus architecture support;
- raw tokens per second versus quality-preserving goodput.

[Previous](Lecture-09.md) · [Part overview](README.md) · [Inference course](../README.md)
