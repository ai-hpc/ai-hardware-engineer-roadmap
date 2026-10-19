# Lecture 01 — Measure useful work and its cost

[Course index](README.md) · [Next: kernel languages](Lecture-02.md)

An inference system receives requests, computes outputs, and spends time, memory
and energy doing so. A useful optimization improves an outcome the application
needs while preserving its quality requirements. More generated tokens alone
can mean longer answers, repeated reasoning, or failed work.

**Before starting:** distinguish seconds, watts and joules, and explain why a
model's training loss does not measure service latency. In this session you will
calculate throughput, deadline-qualified throughput and cost over the same
observation interval. Allow 45–60 minutes.

## Begin with a workload contract

Write down the model revision, input/output lengths, concurrency, sampling
settings, cache state and quality checks. A single interactive conversation and
an offline batch are different workloads. A text request and an image request
can share a decoder but incur different preprocessing and encoder work.

For each request, record arrival, first output, completion, generated token count
and quality status. The **time to first token** includes queueing and preprocessing
if measured from client arrival. **Time per output token** describes subsequent
decode progress under its stated timing boundary. An average token interval does
not establish the worst gap experienced by an interactive user.

Distinguish offered load from completed work. If arrivals exceed service capacity,
a short run can report impressive compute throughput while an expanding queue
makes response times unacceptable. Include queued, failed and timed-out requests
in the report rather than silently dropping them.

## A worked interval

The following numbers are invented for arithmetic practice, not rental quotes
or measured model performance. During one hour, a service completes 100,000
output tokens. Its allocated infrastructure cost for that hour is 2 currency
units. Outputs totaling 80,000 tokens pass the chosen quality and deadline gate.

```text
raw throughput        = 100,000 / 3,600 = 27.78 tokens/s
qualified throughput  =  80,000 / 3,600 = 22.22 tokens/s
cost per million raw tokens       = 2 / 100,000 × 1,000,000 = 20
cost per million qualified tokens = 2 /  80,000 × 1,000,000 = 25
```

The denominator must match the question. For a robot task, report cost or energy
per successful task as well: a token count need not represent useful motion.
State whether partially correct answers can contribute any accepted tokens;
often quality belongs to the whole response, so a failed response contributes
none. Fix this rule before looking at the result.

**Predict:** a modification generates 120,000 tokens for the same hourly cost,
but only 60,000 pass. Did useful throughput improve?

<details markdown="1">
<summary>Answer</summary>

Raw throughput rises to 33.33 tokens/s, but qualified throughput falls to 16.67.
Cost per million qualified tokens rises to 33.33. The result is a regression for
the declared gate. Different answer lengths or task difficulty would require
additional controls before comparing the runs.

</details>

## Separate prices, engineering cost and energy

An API price includes commercial choices. A historical price decline does not
identify which hardware, model, kernel or utilization change caused it. For a
local system, define which costs you allocate: hardware amortization or rental,
hosting, energy, networking and operations. Avoid adding electricity twice if
it is already included in the rental cost used by your model.

Power is an instantaneous rate; energy is its integral. A hypothetical 40 W
system running for ten seconds uses 400 J. If it completes two successful tasks,
that is 200 J per success under that measurement boundary. Include failed
attempts and idle intervals when they belong to the application's operating
cycle. State whether measurement covers the module, host or wall socket.

## Connect cost to a physical lower bound

Suppose one decode step must read 4 GB of weights from a memory interface that
can sustain 100 GB/s. Weight traffic alone takes at least `4/100=0.04 s`, giving
an upper bound of 25 steps/s under those assumptions. This is neither a measured
throughput nor a bound that survives changing the traffic assumption. Batching,
expert selection, caching and speculation change how many useful outputs one
weight read can serve. KV reads, computation and launches add work.

A compiler optimization may remove an activation write without changing weight
traffic. It may matter greatly during prefill and barely affect batch-one decode.
Predict the affected term before collecting a speed number.

## Practice with an existing result

Use the [timed robot-loop lab](../../../Phase%203%20-%20Artificial%20Intelligence/0.%20AI%20and%20Robot%20Learning%20Foundations/Lab-02-Timed-Robot-Loop.md).
Compare accepted policy results, applied commands and final task outcomes.
Explain why the three counts differ. The fixture delays are simulated: its
numbers do not measure laptop or robot throughput. Then sketch the additional
timestamps and energy samples needed to measure a physical deployment.

For a serving experiment, follow the scenario and quality discipline in
[MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/).
A benchmark result is meaningful within its scenario, model, quality target and
system configuration; changing one of them changes the comparison.

**Exit check:** another learner can reconstruct your numerator, denominator,
timing boundary and quality rule from your report. They can also identify one
change that would invalidate the comparison. Reviewed October 18, 2026.
