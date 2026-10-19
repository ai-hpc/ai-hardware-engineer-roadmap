# Part 2 · Lecture 05 — Serving kernels and the scheduler must agree

A fast attention kernel does not guarantee a responsive service. The scheduler chooses which tokens run together, the cache manager owns their state, and the kernel must interpret the supplied shapes and positions correctly. This lesson connects those contracts using the [tested cached-attention example](../Part%201%20-%20Fundamentals/labs/README.md).

## Continuous batching

Requests arrive and finish at different times. A continuous-batching scheduler admits new work between execution steps rather than waiting for every request in a fixed batch to finish. It can improve utilization, but the admission policy, token budget, cache capacity, and prefill/decode mix determine latency.

Distinguish maximum active requests from maximum tokens processed in an iteration. A request can contribute many prefill tokens and later one decode token; speculative verification changes that count again. Increasing a token budget may improve throughput while extending a decode user's wait. Measure the arrival process and queue depth as well as GPU execution time.

## Chunked prefill

A long prompt can be split into chunks, interleaving prompt processing with decode work. Correct chunking retains the prefix KV and absolute positions. The CPU reference proves that chunks of different sizes reproduce a full causal computation. Resetting query positions at each chunk is a correctness bug, even if the shapes look valid.

Chunking does not remove the total attention computation. It changes scheduling, temporary allocations, and kernel shapes. Small chunks can incur more launches and lose matmul efficiency; large chunks can delay existing decodes. A chunk size is a workload policy, not an architecture constant.

## Attention backends

FlashAttention avoids materializing the full attention score/probability matrix through tiled computation and online softmax. FlashAttention-3 specifically targets Hopper's asynchronous features; newer generations and other backends may be better for other devices/shapes. A paper's kernel speedup is not a guaranteed server speedup. Read the [FlashAttention implementation](https://github.com/Dao-AILab/flash-attention) and [FlashAttention-3 paper](https://arxiv.org/abs/2407.08608) for exact supported paths.

FlashInfer supplies inference-oriented attention and related kernels with ragged/paged layouts. The [API documentation](https://docs.flashinfer.ai/api/attention.html) specifies page tables, head dimensions, masks, and workspace contracts. “Backend installed” is weaker evidence than a trace showing that backend selected for the measured request.

For every backend change, test empty/short prefixes where supported, non-square cached attention, partial pages, varied head geometry, and mask semantics. In PyTorch SDPA, set dropout probability explicitly to zero for inference; calling a surrounding module's evaluation method does not automatically alter a nonzero functional argument.

## Prefix caching and identity

Reuse requires matching token IDs and the state that produced the cache: model/adapter, tokenizer/template, position scheme, precision/layout, and other relevant execution options. A string that looks similar may tokenize differently. Shared pages need correct lifetime/refcounts and copy-on-write. Cross-tenant reuse must respect the application's isolation policy.

Track hit rate by **tokens or bytes saved**, not only requests: reusing ten tokens is different from reusing ten thousand. Measure cold and warm distributions separately. Exact prefix reuse can save prefill work; it does not skip decoding a new answer.

## Captures, compilation, and shapes

CUDA graph capture can reduce repeated launch overhead, but requires a supported execution path and stable replay contracts. Runtimes may pad to capture buckets, keep several graphs, or fall back for unhandled shapes. Include capture memory and first-use compilation in startup/cold-latency accounting. Never assume that turning a feature on makes every request use it.

Use the installed runtime's help and matched documentation for flags. The current [vLLM engine argument reference](https://docs.vllm.ai/en/latest/configuration/engine_args/) is useful for interpreting token budgets, cache, and compilation options, but it must be paired with a release pin. No full 70B serving benchmark was executed for this lesson.

**Worked experiment.** Hold model, precision, prompt mix, offered load, and cache state constant. Vary one chunk/token budget; report TTFT, individual ITL tails, throughput, rejection rate, and peak memory. If throughput improves but the required latency SLO fails, that setting is not an acceptable optimum.

Next: [long context](Lecture-06.md).
