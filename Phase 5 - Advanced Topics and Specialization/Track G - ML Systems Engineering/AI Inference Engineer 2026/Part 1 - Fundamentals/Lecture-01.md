# Part 1 · Lecture 01 — Choose the workload and measure the service

An inference engineer turns a model into a service with a stated quality, latency, reliability, and cost target. Start with a concrete request: “For this prompt distribution and arrival rate, deliver correct responses with p95 time to first token below our agreed limit.” A GPU name or a tokens-per-second number alone is not a service specification.

Spend 30–45 minutes on this lesson. You need only arithmetic. Your artifact is a one-page benchmark contract; the following [CPU lab](labs/README.md) makes its memory assumptions executable.

## Four useful workload examples

These are examples, not an exhaustive taxonomy. Vision, speech, multimodal encoders, rerankers, and diffusion add other shapes.

| Workload | First questions | Possible constraint |
|---|---|---|
| Interactive chat | Prompt/output lengths? Reused prefixes? Concurrent users? | First-token delay, pauses between streamed tokens |
| Tool-using agent | Model time versus tool/network time? Valid and correct arguments? | End-to-end task success and latency |
| Offline generation | Completion deadline? Retry rate? Output quality? | Useful outputs per paid GPU-hour |
| Embedding/retrieval | Encoder or decoder-based model? Length distribution? Batch deadline? | Retrieval quality, vectors/s, p95 request latency |

A long prompt may dominate chat latency. A large decode batch may become compute-bound. An agent may spend most time waiting for a tool. Embedding models can be memory- or launch-bound at small shapes. Treat each bottleneck as a hypothesis to test. Prefix reuse helps only when token prefixes, model state, and cache policy really match.

## Put timestamps on a request

Let `a` be client submission, `f` the first output-token arrival, and `l` the last. With `N > 1` output tokens:

```text
TTFT = f - a
request-average TPOT = (l - f) / (N - 1)
end-to-end latency = l - a
```

An inter-token latency (ITL) is one adjacent arrival gap. Its distribution differs from a distribution of request-average TPOTs. State which one you report. For a one-token response TPOT is undefined; do not divide by zero or call it zero latency. Transport can deliver several tokens in one chunk, so client chunk timestamps may not resolve individual generation times.

TTFT includes transport, queueing, tokenization, prefill, sampling, and response delivery according to the chosen boundaries. Prefill normally produces the logits used for the first output token; do not automatically add another full decode pass. Report server-only and client-observed measurements separately when both are useful.

For arrivals at 0 ms and tokens at 120, 150, 195, 210 ms: TTFT is 120 ms, average TPOT is 30 ms, individual gaps are 30, 45, and 15 ms, and completion latency is 210 ms. A good average hides the 45 ms pause.

## Throughput, tails, and money

Count successful outputs during a defined measurement window. State whether throughput counts input tokens, output tokens, requests, or useful completed tasks. Per-GPU throughput divides replica throughput by every GPU reserved for that replica, including idle ranks.

```text
cost per million useful output tokens
  = total dollars spent / useful output tokens × 1,000,000
```

At a hypothetical $8/replica-hour and 400 useful output tokens/s, cost is `$8 × 1e6 / (3600 × 400) = $5.56/MTok`. At $12/hour and 500 tokens/s it is $6.67/MTok: higher throughput, higher unit cost. These are arithmetic examples, not cloud quotations. Include CPU/network/storage, idle capacity, and failed attempts when the product pays for them.

A p99 from 50 observations is essentially an extreme sample, not strong evidence about a rare tail. Keep sample count and quantile method; use a longer steady-state run and repeated trials. Report failures and timeouts alongside latency rather than dropping them. Closed-loop clients that wait for completion before issuing work can hide overload: an open-loop arrival schedule tests queue growth at an offered rate. Cap the test and record rejected requests.

## The benchmark contract

Record model/tokenizer revisions, chat template, precision and calibration, runtime/container versions, hardware topology, request lengths and arrival process, cache warm/cold state, sampling parameters, and correctness criteria. Keep raw request timestamps and outcomes. Separate model download, compilation, startup, warmup, and steady-state serving.

Change one variable, then check both quality and performance. Correctness can mean identical integer outputs, numerical tolerance, preserved sampling distribution, valid schema, or task-quality thresholds; choose the relevant contract before measuring. Schema validity alone does not establish that a tool call is appropriate.

**Exercise.** Compare a service with 1,000 completions and 50 timeouts against one with 990 completions and no timeouts over the same offered workload. What data is missing before choosing? **Answer:** offered request count, retry policy, timeout boundary, latency/quality distributions, and total cost. Completed-token throughput alone cannot decide.

Next: [transformer execution](Lecture-02.md). For timestamp and benchmarking guidance, consult [PyTorch benchmark utilities](https://docs.pytorch.org/docs/stable/benchmark_utils.html); a kernel microbenchmark is only one component of a service measurement.
