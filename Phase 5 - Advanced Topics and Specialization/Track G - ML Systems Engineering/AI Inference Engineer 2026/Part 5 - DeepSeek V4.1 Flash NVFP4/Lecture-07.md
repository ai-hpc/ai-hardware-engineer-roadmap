# Lecture 07 — Long context and cache behavior

**Prerequisites:** Lecture 02 and a correct serving baseline. **Time:** 3 hours. **Outcome:** separate architectural cache savings from observed runtime allocation.

## Model state is not one flat KV tensor

The [DeepSeek technical report](https://arxiv.org/html/2609.19969v1) describes cross-layer reuse, hierarchical indexing, FP4 main KV, and bounded replay for sliding-window state. Its reported **890 bytes/token** concerns global cache under its architecture accounting. It is not the total memory cost of an arbitrary runtime or a promise that a million-token request fits your node.

Build a ledger for global cache, local/window state, index structures, page metadata, request buffers, vision state, and workspaces. Mark shared versus per-request allocations.

## Work the units

As a paper-based estimate, 890 bytes/token × 1,048,576 tokens is 890 MiB of global state. Add the other ledger terms and weights before deciding capacity. If your engine uses another representation, the estimate is only a comparison target.

For a generic page allocator, if each page holds P tokens, N requests with lengths L_i consume:

```text
allocated_token_slots = sum(ceil(L_i / P) * P)
```

This teaches internal fragmentation. Apply it only to the structures your runtime actually pages; do not assume all compressed V4.1 state follows a conventional decoder-only KV layout.

## Controlled sweep

1. Start with a supported short context and double input length while keeping output length fixed.
2. Stop before the predeclared memory or latency limit; do not force an OOM to find a maximum.
3. Measure TTFT, decode timing, peak memory per rank, and admitted concurrency.
4. Repeat with independent prompts and with a known shared prefix.
5. Separate first-use, warm cache, and cache-evicted conditions.
6. Verify outputs remain valid as the context grows.

For a reuse experiment, log tokenizer/encoder revision and every prefix-identity input. Different system messages, image content, or settings may alter the computation and invalidate reuse. A higher hit rate is useful only if the returned computation is still correct.

## Failure exercise

Design two conversations that differ in a meaningful prefix field. Demonstrate that your test would catch a cache erroneously shared between them. Use a local test harness; do not depend on unrelated users' traffic.

**Independent transfer:** optimize repeated-prefix input-heavy traffic, then re-evaluate a no-reuse workload. Separate improvement due to less work from improvement in kernel efficiency.

## Self-check

- Why can TTFT rise sharply before memory is exhausted? Chunking, queueing, indexing, and bandwidth can become limiting first.
- Is the maximum configured context a tested capacity? No; it is a configuration ceiling.
- Can per-token memory be inferred from one request? Not reliably: fixed buffers, page rounding, and lazy allocations require a sweep.

**Exit artifact:** memory/context and latency/context curves, a documented cache-validity test, and a service limit derived from measured behavior. On the analysis route, submit the ledger and equations with unmeasured terms explicit.

[Previous](Lecture-06.md) · [Overview](README.md) · [Next: useful throughput](Lecture-08.md)
