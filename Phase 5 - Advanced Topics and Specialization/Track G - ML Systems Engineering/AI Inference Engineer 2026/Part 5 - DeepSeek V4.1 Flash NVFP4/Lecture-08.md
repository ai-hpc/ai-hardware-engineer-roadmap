# Lecture 08 — Scheduling, load, and goodput

**Prerequisites:** a correct baseline and Lecture 05 timing boundaries. **Time:** 3 hours; CPU fixture first. **Outcome:** report useful service capacity without hiding slow or failed requests.

## Define the service contract

Choose a request mix, offered arrival process, input/output lengths, and latency targets before tuning. A closed-loop client sends new work after earlier work finishes; an open-loop client maintains its arrival schedule. Under saturation they answer different questions. Record scheduled arrival separately from actual dispatch if the client can fall behind.

For request i with n_i output tokens:

```text
TTFT_i = first_content_token_time_i - request_start_i
TPOT_i = (last_token_time_i - first_token_time_i) / (n_i - 1), n_i > 1
completed_output_throughput = completed_output_tokens / observation_window
goodput = output_tokens_from_completed_requests_meeting_SLOs / observation_window
```

TPOT is a per-request average. Its p95 is not p95 inter-token latency. To measure inter-token stalls, collect actual token timestamps; HTTP chunks may contain multiple tokens. Count tokens with the pinned tokenizer/runtime accounting, not whitespace.

## Run the supplied fixture

```bash
python3 labs/inference_metrics.py --demo
```

The demo contains three measured requests, including a failure, and one explicit warmup. Expected completed output throughput is **2 tokens/s**, goodput is **4/3 tokens/s**, and failure rate is **1/3**. These are synthetic arithmetic checks.

Read [the schema](labs/README.md) before exporting real request records. Failed requests stay in the window and failure denominator. Partial outputs from failed requests do not count as completed output. Single-token requests have no TPOT and are evaluated on TTFT alone.

## Scheduling experiment

1. Establish low-load latency, then increase offered rate in small steps.
2. Keep prompt mix, output limits, quality settings, and hardware fixed.
3. Sweep one supported scheduler control, such as prefill chunking or admission limits.
4. Track arrivals, completed requests, failures, queue growth, TTFT/TPOT distributions, goodput, and peak memory.
5. Stop after the predefined load/duration budget or sustained overload condition.
6. Repeat three times and publish raw records. Report run variability; do not average run percentiles.

A finite run should include its drain period. The supplied summarizer uses earliest measured start to latest measured end; for a steady-state service study, additionally report the fixed arrival window, drain duration, and any censored requests. Do not silently omit clients that timed out.

## Failure clinic

A scheduler can improve aggregate throughput by admitting more work while making interactive requests unusable. It can also appear faster by returning fewer tokens, dropping failures, changing reasoning effort, or using warm prefixes only in one condition. Include checks against each applicable mechanism.

**Independent transfer:** mix short interactive requests with long input-heavy requests. Compare fairness across classes as well as total goodput.

## Self-check

- Which rate belongs in a deployment cost model? Useful throughput for the defined workload and SLO, with failures and idle/drain costs included.
- Why can TPOT look good while streaming feels bad? An average can hide long individual stalls.
- Is zero observed failure in ten requests a reliability result? It is a smoke test, not strong statistical evidence.

[Previous](Lecture-07.md) · [Overview](README.md) · [Next: quality](Lecture-09.md)
