# Chapter 6: Production Serving — Quality, Goodput, and Recovery

**Previous:** [Kernel engineering](05-Blackwell-Kernel-Engineering.md) · **Course:** [Blackwell inference](README.md)

## Turn a benchmark into a service contract

Chapter 3 measures one serving group; Chapter 4 selects its placement. This chapter asks whether the group remains useful under a realistic arrival process, failures, and upgrades. The repository supplies no validated production Qwen/B200 throughput table. Keep your own measurements separate from vendor benchmark results and analytical bounds.

Specify input-length distribution, output-length distribution, arrival rate/burstiness, concurrency limits, quality target, TTFT, inter-token latency, completion-latency limits, and error budget. “p99 < 500 ms” is ambiguous until the measured interval is named.

## Select a runtime by a compatibility experiment

TensorRT-LLM, vLLM, and other engines change quickly. For each candidate save a release/commit or container digest, exact model revision, quantization format, GPU architecture, backend, and documented feature support. Confirm the actual kernel path and numerical results with a small run before a load test. A claim that one engine is universally fastest is not a selection method.

Use the maintained [TensorRT-LLM quantization matrix](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/quantization.md) and the selected release's configuration documentation. CUDA Graphs, prefix caching, and chunked prefill are workload-dependent options; vLLM's [graph design](https://docs.vllm.ai/en/v0.18.2/design/cuda_graphs/) describes supported modes. Record the mode actually used.

## Admission control prevents queueing from hiding overload

A server can report high aggregate tokens/s while most users miss their latency objective. Plot offered request rate against achieved goodput and p99 latency. Track queue time separately from execution time and include timeouts and rejected requests.

An illustrative service admits 8 requests/s and has an average completed-request residence time of 2 s. Under a stable regime, Little's law gives an average of 16 requests in the system. It does not tell you the p99 latency or guarantee sufficient GPU memory: long contexts and correlated bursts change those constraints. Use the actual distribution in the load test.

Compare closed-loop tests (clients wait before submitting again) with an arrival-driven test. A closed-loop client can reduce offered load when the server slows down, hiding overload. Record the generator protocol in every chart.

## Observability with a useful causal chain

| Signal | Possible explanation | Small follow-up |
|---|---|---|
| TTFT rises, GPU execution stable | Queueing or prefill scheduling | Separate queue and prefill timestamps |
| Inter-token tails rise at long context | KV traffic, cache pressure, contention | Hold concurrency fixed and sweep context |
| One rank runs out of memory | Uneven partition or replicated buffers | Compare per-rank allocated/reserved bytes |
| Faster kernel, slower service | Extra packing, graph recapture, synchronization | Profile the full request path |
| Quality changes after upgrade | Different kernels/scales/tokenizer behavior | Replay the frozen evaluation set |
| CPU/GPU transfer stalls | Offload or input pipeline placement | Trace transfer sizes and dependencies |

Measure sustained power and thermals at a named boundary if cost or efficiency matters. GPU power excludes host, network, storage, and facility overhead; rented pricing additionally depends on instance availability and contract. Date any price quote and calculate cost per **successful SLO-compliant** request.

## Recovery and upgrade lab

Run these experiments on an isolated service with saved test inputs:

1. Establish one warm baseline and one cold-start result, including model download/load, graph capture, and readiness time.
2. Interrupt a worker, restart it, and measure time to readiness. Document which in-flight requests fail, which retry, and how duplicate responses are prevented.
3. Send an overlength request and an overload burst. Confirm explicit rejection/backpressure rather than silent prompt truncation or unbounded queue growth.
4. Deploy a candidate runtime/checkpoint beside the baseline. Replay the same quality and load tests, then document rollback to the exact prior artifact.
5. Archive logs, traces, complete command/config files, raw results, and unresolved failures. A health endpoint is necessary but does not prove numerical readiness.

## Connect service engineering to robotics

Cloud chat goodput and a local robot control deadline are different objectives. For a VLA deployment, replace token-only metrics with observation-to-action age, valid action rate, action horizon, stale-action rejection, task success, and energy per completed task. A network outage must not leave old policy outputs looking current. Keep the deterministic controller and its deadline policy visible in the [physical-AI SoC design](../../../../Track%20F%20-%20AI%20Chip%20Design/Guide.md).

**Final artifact:** one deployment dossier connecting hardware inventory → checkpoint format → memory budget → runtime/kernel path → placement → quality/load/recovery results. It should let another learner reproduce the decision and see its limits.
