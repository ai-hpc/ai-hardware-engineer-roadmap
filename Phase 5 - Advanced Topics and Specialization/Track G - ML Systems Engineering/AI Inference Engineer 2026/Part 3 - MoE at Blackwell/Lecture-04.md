# Part 3 · Lecture 04 — When prefill/decode separation earns its transfer cost

Prefill and decode often favor different batch shapes and scheduling policies. Disaggregation assigns them to separate worker pools and transfers the state needed to continue a request. It can reduce interference and permit independent scaling; it also adds transfer, queueing, ownership, and failure-handling costs. It is a design to evaluate, not a feature to enable at an arbitrary GPU-count threshold.

## Draw the request lifecycle

```text
admit → prefill queue → prefill computation → transfer/acknowledge KV state
      → decode queue → decode steps → release state
```

The real system may overlap transfer with layer/chunk computation, reuse cached prefixes, or stream pages. Identify when the first token is sampled and delivered, who owns each page, and when the sender may release it. A timeout must not leave two workers generating duplicate outputs or free memory still being transferred.

Transferred state needs a compatibility contract: model/adapter revision, token prefix, position/RoPE configuration, KV dtype/scales/layout, page mapping, and shard topology. Different TP degrees can require reshaping or redistribution. Equal tensor element counts do not prove compatible cache semantics.

## Work a transfer bound

For the dense example's 320 KiB/token BF16 cache and 8192 prompt tokens, full unsharded payload is 2.5 GiB. At a hypothetical sustained **one-way** 25 GB/s transfer rate:

```text
2.5 * 2^30 bytes / 25e9 bytes/s ≈ 0.1074 seconds
```

That is a payload-only bound for a stated path. Protocol overhead, contention, staging, reshaping, queueing, and synchronization add time; overlap and cached-prefix reuse can reduce the exposed portion. If eight ranks each transfer one-eighth over independent links, use per-shard bytes and the actual simultaneous paths. Do not compare a global byte total with an aggregate bidirectional marketing rate.

For a compressed MLA layout, transfer payload differs; use the measured runtime representation. Compression savings may be offset by conversion or different sharding. The [CPU accounting lab](../Part%201%20-%20Fundamentals/labs/README.md) calculates ideal dense and MLA payloads, not network latency.

## Read the research as experiments

[Splitwise](https://arxiv.org/abs/2311.18677) studies phase splitting and resource allocation. [DistServe](https://arxiv.org/abs/2401.09670) frames allocation around goodput under latency objectives. [Mooncake](https://arxiv.org/abs/2407.00079) emphasizes KV-centric scheduling and storage/transfer. Their reported gains belong to their models, baselines, hardware, workload distributions, and paper versions. They are not universal expected speedups for your service.

For an implementation, inspect the [Mooncake project](https://github.com/kvcache-ai/Mooncake/) or the chosen runtime connector at a pinned revision. Connector availability is weaker evidence than a full correctness and failure test of your exact model/cache layout.

## Compare against a capable colocated baseline

The baseline should include an appropriate batching/chunked-prefill policy and capacity limit. Otherwise an apparent disaggregation win may simply fix an avoidable scheduling problem. Hold total reserved resources, input/output lengths, offered arrival rate, precision, quality threshold, and cache state constant.

Report useful completed requests or tokens meeting the chosen SLO, TTFT and ITL distributions, both pools' queue depth/utilization, transfer bytes and time, peak memory, rejected requests, retries, and cost. A pool can be underutilized while the other is saturated. Scaling ratios follow the measured workload, not a fixed prefill:decode rule.

## Failure exercise

A prefill worker finishes, sends half the pages, and dies. A correct system must distinguish incomplete transfer from complete acknowledged ownership, avoid decoding from partial state, and either retry/recompute or fail the request according to policy. Cache reuse must not cross model/tenant identities accidentally. Include cancellation during transfer, stale metadata, missing pages, and a decode worker failure in integration tests.

**Decision question.** If separation saves 60 ms of interference but exposes 100 ms of extra transfer/queueing, is it a latency win? **Answer:** no under that measured request path. It might improve throughput or tail isolation at another load, which requires another experiment.

Next: [speculation and production contracts](Lecture-05.md).
