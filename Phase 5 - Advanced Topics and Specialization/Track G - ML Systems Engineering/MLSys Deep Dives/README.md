# MLSys deep dives: from model computation to useful work

This seven-session course explains how model structure, kernels, compilers,
runtime scheduling and hardware jointly determine an application's behavior.
The goal is a small optimization report whose correctness, quality and timing
claims another learner can check.

**Start after:** [AI math and training](../../../Phase%203%20-%20Artificial%20Intelligence/0.%20AI%20and%20Robot%20Learning%20Foundations/Guide.md)
and the [inference fundamentals](../AI%20Inference%20Engineer%202026/Part%201%20-%20Fundamentals/README.md).
You should be able to trace a matrix shape and distinguish model parameters from
request state. If robotics and AI are both new, begin with
[Start here](../../../Start-Here.md) before returning to this advanced survey.

## A manageable route

Study one lesson per session. Spend roughly one hour reading, predicting and
checking a worked example, then use the linked lab for a longer practical
session. You do not need to install every tool mentioned. Choose one functioning
CPU environment first; add a GPU backend only when it supports your workload.

| Session | Question you will answer | Practice to carry forward |
|---|---|---|
| [01 — Useful work and cost](Lecture-01.md) | Which numerator, denominator and quality gate make a result meaningful? | A request/task record with timing and resource boundaries |
| [02 — Kernel languages](Lecture-02.md) | Which details does a language expose, and which remain your responsibility? | A reference with shape, tail, stride and synchronization cases |
| [03 — Compilers and runtimes](Lecture-03.md) | Does the change improve computation, launch overhead or scheduling? | An actual compiled CPU result and an explained timeline |
| [04 — Recurrent and hybrid state](Lecture-04.md) | Which state is fixed-size, and which grows with history? | Output, final-state and continuation comparisons |
| [05 — Inspect an executable model](Lecture-05.md) | What do a pinned config and implementation establish? | A model/checkpoint/runtime record with explicit unknowns |
| [06 — Speculative decoding](Lecture-06.md) | When does verified drafting improve accepted progress? | A rejection-identity check and a complete cycle-cost calculation |
| [07 — Robot deadlines and energy](Lecture-07.md) | Does the output remain useful at actuation? | One paired baseline/candidate report with a failure explanation |

## Reuse the course's working experiments

The [native TVM lab](../TVM%20Deep%20Dives/labs/README.md) compiles and executes
LLVM/Relax examples. The [FlashAttention CPU lab](../../../Phase%204%20-%20Track%20C%20-%20DL%20Inference%20Optimization/02%20-%20Kernel%20Engineering/FlashAttention%20Course/CPU-Attention-Lab.md)
checks attention mathematics and gradients. The [hybrid-state lab](../GLM-5.3-Flash%20Architecture%20Mastery/CPU-Hybrid-State-Lab.md)
compares an independent reference with pinned upstream PyTorch KDA/KPool code.
The [timed robot-loop lab](../../../Phase%203%20-%20Artificial%20Intelligence/0.%20AI%20and%20Robot%20Learning%20Foundations/Lab-02-Timed-Robot-Loop.md)
shows how a fixed trained policy behaves under different simulated queues and
expiry rules.

These experiments cover different execution boundaries. A CPU component result
is not a full-model GPU benchmark, and a chosen simulator delay is not a measured
board latency. Keep that distinction in your own report instead of treating all
successful commands as the same kind of evidence.

## Your capstone report

Choose one operator, model component or small robot-policy task. Record the
baseline before changing it. State the single proposed change and predict which
term it affects: arithmetic, transferred bytes, cached state, launch overhead,
queueing, or accepted task progress. Fix the quality and correctness criteria
before comparing results.

Retain inputs/seeds, source revisions, environment, commands, raw output and
one deliberate failure case. Separate compile/install time from warmed execution.
Show distributions when timing varies; do not average per-run percentiles and
present them as a pooled percentile. If the candidate is worse or unchanged,
explain what that reveals about your original hypothesis.

You finish when another learner can reproduce the result and identify where its
claim stops. A model replacement needs its own quality evaluation. An operator
speedup needs an end-to-end explanation. A robot application needs observation
freshness and action semantics as well as inference performance.

## Currency and source boundaries

Reviewed **October 18, 2026**. The [source revision record](sources/verification-2026-10-18.json)
pins TileRT, cuTile, DFlash and Modular documentation/code used for current tool
claims. Individual lessons link the relevant research and hardware specifications.
A public repository, a released binary wheel and a locally executed implementation
are distinct artifacts. The survey's arithmetic examples are labeled illustrative;
it does not claim local execution of TileRT, Mojo/MAX, cuTile or a Thor workload.

Continue into [inference engineering](../AI%20Inference%20Engineer%202026/README.md)
for a deeper serving sequence, or use the [Physical AI Design Studio](../../../Physical-AI-Design-Studio.md)
to carry your validated component into a larger system. The [learning guide](../../../Learning-Guide.md)
provides the shared rubric for correctness, reasoning, measurement, reproduction
and transfer.
