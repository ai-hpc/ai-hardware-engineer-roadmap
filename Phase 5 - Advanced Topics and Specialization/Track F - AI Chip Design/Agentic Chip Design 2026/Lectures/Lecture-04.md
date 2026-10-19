# Lecture 04 — Measure generation and repair fairly

[Course guide](../Guide.md) · [Previous](Lecture-03.md) · [Next: Lecture 05](Lecture-05.md)

A score is meaningful only with a task set, checker, and sampling protocol. The [VerilogEval repository](https://github.com/NVlabs/verilog-eval) distinguishes original code-completion and newer specification-to-RTL tasks. Its revised protocol and tool requirements differ from the original release. Pin the benchmark revision; a current result should not silently reuse an older task name or pass@k definition.

## Separate the denominators

Imagine ten fixed tasks, one initial candidate each. Eight compile and five pass the functional checker. After up to three repair rounds, seven tasks pass. Report:

- Initial compile success: 8/10 = 80%.
- Initial functional success: 5/10 = 50%.
- Final success under the repair budget: 7/10 = 70%.

Do not report 5/8 as the main functional success rate: it excludes the two failed compilations. Do not call 70% “pass@1”; it includes additional proposals and checker feedback. Record unresolved tasks, timeouts, tool failures, and exceeded budgets.

## What pass@k means

For n independently sampled candidates for one task, c of which pass a fixed checker, a common estimator is `1 - choose(n-c,k)/choose(n,k)`, for k≤n. It estimates the chance that at least one candidate in a sample of k is correct under that sampling protocol. It is not an estimate of an adaptive repair agent that receives failure feedback between attempts.

**Worked example:** n=4, c=1, k=2 gives `1 - 3/6 = 0.5`. This does not mean a user can always select the correct candidate without a checker. A reported deployment workflow must include its candidate-selection procedure and cost.

With one small course module, a pass rate has very limited generality. Three random seeds are three test traces for that module, not three independent design tasks. Report the number of tasks, candidates, seeds, and successful checks separately.

## Coverage and checker quality

Code coverage asks which implementation statements or branches executed. Functional coverage asks whether meaningful behavior was exercised: empty/full transitions, stalls, replacement, saturation, reset while full. Neither automatically proves correctness. A checker that accepts a deliberately faulty implementation has an identified gap even if line coverage is high.

Keep visible development tests and held-out evaluation tests separate if claiming generalization. Public course tests are visible; do not relabel them hidden after using their failures to repair a candidate. For a small local study, it is acceptable to report only visible-test success, with that limitation stated.

## Build a result table

For each task and attempt, retain source hash, checker hash, seed, compile status, functional status, wall time, model/tool cost, and failure category. Freeze the first-attempt row before repairs. Aggregate only after every attempt has a defined terminal status.

**Exercise:** a timeout is rerun with a larger budget and passes. Which result belongs in the original comparison? **Answer:** the timeout under the original budget. The larger-budget rerun is a separate condition, though both can be reported.

Deliver a small table for the baseline and the three controlled faults from [Lecture 05](Lecture-05.md). This validates the scoring procedure; it is not a language-model benchmark.
