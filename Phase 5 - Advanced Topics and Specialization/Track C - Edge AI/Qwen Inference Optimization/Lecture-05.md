# Lecture 5: Speculation, routing, and capacity from explicit assumptions

**Reviewed 18 October 2026.** Input: evaluated model artifacts and a workload definition. Output: a reproducible planning calculation plus an experiment that could falsify it. All timing/capacity numbers below are **teaching inputs**, not measured Qwen hardware performance or market prices.

## 1. Accepted prefixes, not independent accepted-token totals

Suppose a draft proposes `K` tokens, and each next token has constant conditional acceptance probability `a`. Verification stops at the first rejection. Under this simplified model, the expected number emitted per round, including the corrected or bonus token, is

\[
E[N]=1+a+a^2+\cdots+a^K.
\]

For `a != 1`, this is `(1-a**(K+1))/(1-a)`; for `a=1`, it is `K+1`. Do not use `1+K*a`: later candidates count only if earlier candidates were accepted. EOS, variable acceptance, tree methods, and practical implementations can change the realized count.

The portable lab computes this sum and verifies it with a 50,000-round Monte Carlo simulation. At `a=.55,K=5`, expectation is **2.1607097** tokens/round; the recorded simulation gives **2.15674**. This is a probability experiment, not an observed model acceptance rate.

For sequential draft cost `td`, verification cost `tv(K)`, and other overhead `to`:

\[
R=\frac{E[N]}{K t_d+t_v(K)+t_o}.
\]

Using illustrative `td=5 ms`, `tv(5)=32.5 ms`, `to=0` gives **37.58 token/s**. A hypothetical target baseline of 30 ms/token gives 33.33 token/s, so this case is only about 12.7% faster. Lower acceptance or higher overhead can make speculation slower; the test includes such a case. Weight reuse does not make verification computation or its KV traffic free.

## 2. Preserve the target distribution and token contract

For greedy target decoding, prefix equality against target choices can preserve greedy output. For stochastic speculative sampling, acceptance/rejection and corrected sampling must follow the algorithm; “accept tokens the target would have produced” is insufficient. See the [speculative decoding paper](https://arxiv.org/abs/2211.17192).

Classic draft/target implementations commonly require compatible token IDs and distributions. Equal vocabulary row counts do not prove compatible tokenizers, and differing padded table sizes alone do not fully characterize token mapping. Compare tokenizer contents, special tokens, normalization and templates, then use the runtime's supported method. Universal assisted generation and other mapping methods have their own requirements.

Do not assert a 55% or 70% acceptance rate because two models share a brand. In particular, Qwen3-4B-Instruct-2507 is a non-thinking checkpoint, not the original mixed-mode Qwen3 variant. A Qwen2.5 draft/target pair still needs compatible artifacts and measured acceptance. Current [vLLM speculative examples](https://docs.vllm.ai/en/latest/examples/features/speculative_decoding/) distinguish methods and supported configurations. EAGLE, Medusa and related methods add model-specific components; they are not guaranteed drop-in or universally faster.

## 3. Routing is a quality/latency decision

A router selects an edge or remote path using task evidence, available resources and a declared product policy. Build a held-out set containing actual task outcomes, then compare always-local, always-remote, and routed strategies. A model's self-reported confidence is not automatically calibrated. If escalation occurs only after a complete local attempt, that local delay must be included in end-to-end latency.

Keep input-sharing boundaries explicit in the application. Record when a request takes the remote path, what happens when that path is unavailable, and which model/version generated the result. Local execution also consumes device energy and engineering resources; calling it “free” hides costs.

Percentiles do not average linearly. If 70 requests take 100 ms and 30 take 2,000 ms, the nearest-rank p95 of the combined 100 requests is **2,000 ms**, not `0.7*100+0.3*2000=670 ms`. The latter is the mean for this toy mixture. Use the actual mixed request distribution when assessing an SLO.

## 4. Work through capacity with headroom

Assume 1,000 daily users, 50 requests/user/day, 30% remote routing, 400 output tokens/request, and a peak factor of five:

```text
Remote requests/day = 1000 * 50 * .30 = 15,000
Output tokens/day = 15,000 * 400 = 6,000,000
Average output token/s = 6,000,000 / 86,400 = 69.444
Peak output token/s = 69.444 * 5 = 347.222
```

Now suppose a **measured** replica can serve this workload at 400 output token/s, and the planning policy limits demand to 70% of that rate:

```text
replicas = ceil(347.222 / (400 * .70)) = 2
```

The lab verifies this arithmetic. Neither 400 token/s nor 70% is a measured fact or universal rule here. Derive them from a representative load test and your latency/reliability target. A replica can itself use several GPUs; replicas and GPUs are not the same count. Redundancy during maintenance/failure can require additional capacity.

This output-rate estimate also omits input/prefill cost, length distributions, burstiness, network effects, retries and queueing. Validate those separately. Little's law `average in-flight requests = arrival rate * average time in system` is a useful steady-state consistency check, not a p95 latency predictor.

## 5. Observe work that completed

Measure request arrivals/completions/errors, prompt and generated tokens, TTFT, inter-token latency, end-to-end latency, running/waiting requests, memory/KV allocation, and the actual backend. Define whether TTFT includes network, queueing and tokenization. Report latency with workload/concurrency; throughput without the corresponding latency can reward unbounded queues.

Current [vLLM production metrics](https://docs.vllm.ai/en/latest/usage/metrics/) are exposed at `/metrics` by the server. Inspect the names and types for the installed version. There is no need to invent a `vllm metrics` command or an `--engine-metrics-enable` flag. On a running local optional server:

```bash
curl --fail http://127.0.0.1:8000/metrics
```

This command was source-checked, not run against a model server in this lesson. GPU utilization, KV occupancy, and collective-time fractions are diagnostic observations, not universal “healthy at 90%” thresholds. Trace the workload before assigning a cause.

## 6. A useful final experiment

Choose one supported speculation or routing configuration. Preserve raw responses and timing samples for an unchanged reference and candidate. Report task success, strict tool-call validity if relevant, paired failures, acceptance-prefix distribution, extra memory, TTFT, inter-token latency, completed throughput, and fallback behavior.

A faster configuration that changes the sampler, truncates outputs, drops failures from the denominator, or processes shorter prompts is a different workload. Explain such changes rather than presenting them as an optimization gain.

**Answers to common mistakes:** `1+K*a` overcounts accepted prefixes; a 70/30 routing mix does not determine p95; one 400-token/s replica at 347-token/s peak has little headroom; a lower-bit artifact need not use a faster kernel; a compatible tokenizer does not guarantee a high acceptance rate.

Next: [GEMM layouts and executable checks](Lecture-06.md).
