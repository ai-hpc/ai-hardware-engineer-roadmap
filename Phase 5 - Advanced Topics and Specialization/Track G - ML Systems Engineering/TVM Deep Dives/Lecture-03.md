# Lecture 03 — Tuning is measured search with a budget

A schedule search generates candidates, builds them, runs them, and records measurements. A learned cost model can rank promising candidates, but actual device measurements remain the evidence. Correctness is a separate gate: a fast wrong candidate is not a valid optimization.

Start with the [lab's four-candidate CPU sweep](labs/README.md). It measures tile sizes 1, 4, 8, and 16 for the same 37-element operation and checks every output before reporting time. This small manual search explains the process without pretending to be a large MetaSchedule experiment.

## Separate the costs

For each candidate record generation time, compilation time, measurement repetitions, and runtime. Warm up before collecting steady-state samples. Run on a controlled target with a fixed thread count/affinity policy where practical; background load, CPU frequency, thermal limits, caches, and tiny timing windows can reorder close candidates.

The lab's Python/FFI calls are deliberately small. Their duration includes call overhead, and a difference of a few nanoseconds can be noise or wrapper effects rather than useful throughput. No fixed winner or speedup threshold belongs in CI. Use larger realistic shapes and the runtime's evaluator when studying hardware throughput.

## What MetaSchedule adds

MetaSchedule organizes design spaces, search strategies, builders/runners, task scheduling, cost models, and persistent databases. The [current tutorial](https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/meta_schedule.html) documents whole-model Relax tuning passes and task extraction. The tested release exposes schedule APIs under `tvm.s_tir.meta_schedule`; moving development docs may import core components from `tvm_tirx` instead.

AutoTVM, Ansor, and MetaSchedule are distinct generations of TVM tuning infrastructure, not aliases for one API. A sketch calling an old `tune_relax` function without a version is not a complete executable recipe. Match the target construction, compiler snapshot, dependencies, input module, and database format.

A random cost model is useful for a tiny smoke test of build/run/database plumbing; it does not demonstrate that learning improved search. A four-trial run cannot establish a best schedule. In the recorded local [MetaSchedule attempt](labs/metaschedule_attempt.json), one candidate failed to build and no usable measured schedule was returned; the successful four-candidate lab sweep is manual search, not a claimed MetaSchedule success. An RPC runner additionally requires a real target service, transport/authentication policy, compatible runtime, and control over competing jobs.

## When does tuning pay back?

Let extra tuning/build time be C seconds, baseline runtime t0, and improved runtime t1. If `t1 < t0`, runtime-only break-even executions are:

```text
N > C / (t0 - t1)
```

For C=600 seconds and a saving of 0.2 ms/call, more than three million calls are needed to recover that wall-clock expenditure under this simple model. GPU/CPU costs, parallel tuning workers, developer time, and deployment frequency can change the economic calculation. If the candidate is slower, there is no positive break-even count.

Model-level benefit also depends on task frequency and the fraction of total latency affected. Saving half of a kernel that occupies 4% of a request saves about 2% overall if nothing else changes. Tune measured hot paths rather than the easiest isolated kernel.

## Database reuse is conditional

Retain workload IR/shape/dtype/layout, compiler revision, target features, schedule trace, measurement method, and run conditions. A database entry for one CPU ISA or GPU is not automatically correct or fast on another. Changes to lowering, runtime ABI, layouts, or permitted numerical transformations may invalidate reuse even when a model's name is unchanged.

XLA also supports [persisted GPU autotuning](https://openxla.org/xla/persisted_autotuning); tuning is not unique to TVM. Compiler comparisons should state version, search budget, baseline libraries, and measurement environment rather than claiming one family has no search capability.

**Exercise.** Two candidates measure 10.0 and 9.9 microseconds with run-to-run variation of 0.4 microseconds. What should you report? **Answer:** the evidence does not yet establish a reliable 1% improvement. Repeat under controlled conditions, report dispersion, and prefer a robust choice if the difference remains unresolved.

**Deliverable.** A correctness-checked candidate table, a stated search budget, and a break-even estimate. Next: [graph optimization](Lecture-04.md).
