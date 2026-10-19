# Lecture 08 — Budget latency and cost per accepted design

[Course guide](../Guide.md) · [Previous](Lecture-07.md) · [Next: Lecture 09](Lecture-09.md)

An agentic workflow consumes model inference, tool runtime, storage, and reviewer attention. A useful system improves the rate of accepted results under a defined budget. Counting generated modules or output tokens alone can reward a system that produces many failures.

## Start with an accounting boundary

Record model input/output usage, local or API inference cost under the chosen pricing assumptions, simulator/synthesis compute time, elapsed wall time, and human review time separately. Prices and hardware rates change; store the dated rate table used for a calculation. The arithmetic below uses hypothetical costs, not a current provider quote.

Suppose ten tasks cost $2 each to attempt and six pass the fixed checker. Total attempt cost is $20; cost per accepted task is `$20/6 ≈ $3.33`. Do not divide only the successful tasks' costs by six and discard the failures. If the accepted tasks still require review, include that cost in a separately defined “reviewed result” metric.

A repair policy costing an extra $8 and increasing successes from six to eight gives total `$28/8 = $3.50` per accepted task. It improves completion rate but increases this cost metric. Which policy is useful depends on deadlines and the value of completing otherwise failed tasks; one number cannot decide every objective.

## Separate service time from waiting time

One task's end-to-end latency includes queueing, model calls, tool execution, retries, and review waits. A GPU serving the model faster does not necessarily remove the EDA tool bottleneck. Parallel workers can improve total throughput while increasing memory demand, license contention, or per-task tail latency.

If two simulator workers each take one second per independent task, their ideal capacity is two tasks/second. Actual capacity can be lower because of compilation, I/O, scheduling, and shared-resource contention. This is a capacity estimate, not an observed benchmark. Measure the complete pipeline before choosing worker counts.

## Keep experiments repeatable

Use the same task set and termination policy for compared serving configurations. Cache reuse must be stated: a repeated prompt may benefit from caching, while a new task may not. Preserve first-attempt success and final success even if one configuration generates more candidates within the wall-time limit.

Do not measure a remote model's generation latency as if it were the complete chip-design latency. Report median and a tail percentile for sufficiently many tasks, plus timeout count and sample size. With ten observations, a tail estimate is coarse; show the raw task times rather than implying precise reliability.

**Exercise:** a configuration doubles tokens/second but increases failed repairs because a smaller model misses reset behavior. Which metric shows the regression? **Answer:** success under the fixed task/budget and cost per accepted result; raw generation throughput misses it.

Deliver an accounting sheet with explicit rates, all attempted tasks, terminal statuses, and two policy comparisons. Continue to [Lecture 09](Lecture-09.md) to assemble the full experiment.
