# Lecture 05 — Prefill, decode, and kernel profiling

**Prerequisites:** a passing Lecture 04 baseline; roofline analysis. **Time:** 3–4 hours on the serving node, or 90 minutes designing the experiment. **Outcome:** identify one limiting stage and test a causal explanation.

## Decompose the critical path

Separate tokenization/encoding, queueing, prefill, first-token selection, decode iterations, and transport. For multimodal requests add image preprocessing and the vision encoder. DeepSeek V4.1's execution path must be read from the actual trace; naming every stage “attention” hides actionable differences.

Measure two views: device work through GPU timing/traces and client-visible latency through a monotonic clock. Synchronizing after every operator changes overlap; use that only for diagnosis and label the result.

## Worked bottleneck calculation

Suppose a synthetic decode iteration spends 40% of time in routed-expert compute. A perfect 2× speedup of that component produces:

```text
overall_speedup = 1 / (0.60 + 0.40 / 2) = 1.25
```

It does not produce a 2× faster server. If queueing dominates under load, client latency may improve by a different amount; measure it.

Build a stage table: time share, bytes moved, kernel count, collective time, launch gaps, and the evidence for each value. Avoid calling a stage bandwidth-bound merely because it is decode: batch size and expert token distribution can change arithmetic intensity.

## Guided experiment

1. Select a short-input/long-output workload and an input-heavy workload.
2. Run unprofiled baselines, then a short [Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) capture using the tool version installed on the node.
3. Annotate CPU waits, GPU kernels, memory transfers, and collectives.
4. Choose one hypothesis: launch overhead, expert GEMM shape, gather traffic, cache attention, or communication.
5. Change one supported knob. Predict its effect on two metrics before running it.
6. Re-run correctness, then the unprofiled benchmark. Use a second trace to explain the change.

Do not compare profiled throughput with an unprofiled baseline; profiling overhead is part of the experiment.

## Failure clinic

| Symptom | Competing explanations | Discriminating experiment |
|---|---|---|
| Low GPU utilization | CPU gaps, small kernels, transfers, load starvation | Compare launch gaps and a higher admitted load |
| FP4 kernel is slower | Poor shape, scale conversion, layout packing, fallback | Inspect kernel dispatch and isolate packing time |
| Faster prefill, worse p99 | More interference with decode or larger queues | Hold offered load fixed and inspect queue time |
| Large average expert speedup | One rank becomes the straggler | Compare per-rank timing, not only a global mean |

**Independent transfer:** repeat the best change at a different batch/context mix. If it regresses, state the region where it should be enabled.

## Self-check and evidence

A profiler suggests a hypothesis; the controlled change tests it. Your artifact must include raw trace references, baseline and variant configuration, an Amdahl estimate, measured effect, and the quality gate. An optimization with no measurable gain is a valid result if you explain the ceiling it did not move.

[Previous](Lecture-04.md) · [Overview](README.md) · [Next: parallelism](Lecture-06.md)
