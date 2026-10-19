# Part 2 · Lecture 06 — Long context: capacity, position, and quality

“Supports 128K” combines several independent claims: the model's position recipe, the runtime's implementation, enough memory, and useful quality at that length. This lesson separates them. Use exact token counts: here 128K means 131072 tokens, not 128000 characters or bytes.

## Compute storage before allocating

For the dense 80-layer GQA geometry in Lecture 01, BF16 KV is 320 KiB per token. Thus 131072 tokens require 40 GiB for one request, before page rounding and metadata. FP8 values alone halve that to 20 GiB, but scales and layout overhead remain. The [CPU lab](../Part%201%20-%20Fundamentals/labs/README.md) checks the arithmetic and independent per-request page rounding.

Reserve space for weights/scales, active activations, kernel and communication workspace, graph captures, runtime allocations, and transient peaks. A GPU's product capacity is not an entirely free KV budget. For sharded caches, derive per-rank layouts; do not divide by TP unless the actual head and replication rules justify it.

Attention prefill work includes a component quadratic in sequence length. FlashAttention saves score-matrix traffic/storage but still computes the relevant attention interactions. During decode, a new query attends over the retained context; sufficiently long context can make KV reads dominate even when short-context decode was dominated by weight reads.

## Position extension must match the checkpoint

The pinned Qwen2.5-72B configuration in this course has `max_position_embeddings=32768`. Its [official model card](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) describes a YaRN recipe for extended context. Record the complete RoPE scaling configuration and the runtime's support for it. Setting a maximum length flag does not by itself reproduce the position transformation.

Llama 3.3 has its own position configuration; do not copy Qwen's YaRN settings onto it. A runtime may reject a requested limit, accept it with changed behavior, or require explicit extension options. Check configuration and evaluate the resulting model, not only whether allocation succeeds. The [YaRN paper](https://arxiv.org/abs/2309.00071) explains the method and its scope.

## KV quantization and chunking

Quantized KV adds scaling and conversion requirements. Check whether scale values are loaded, computed, or defaulted; an unsupported or poorly calibrated path can damage attention even though outputs remain finite. Follow the [runtime's quantized-KV contract](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/) for a pinned version. Test the longest intended contexts and relevant tasks separately from weight quantization.

Chunked prefill controls how much prompt work occupies an iteration. It does not shorten the mathematical context or eliminate its cache. Correctness requires absolute positions and consistent prefix state across chunks. The tiny CPU example is a fast way to understand this before debugging a 40 GiB cache.

## Prefix sharing changes physical allocation

If 100 requests share a 10,000-token prefix, a system with correct page sharing need not allocate 100 separate physical copies of every prefix page. It still needs per-request metadata and distinct suffix state. Page boundaries, copy-on-write, eviction, and tenant isolation constrain the savings. Measure physical allocated pages and cache misses instead of assuming a perfect 100× reduction.

**Worked budget.** With a hypothetical 100 GiB remaining KV budget and 40 GiB per unshared full-length request, at most two such caches fit by this simplified calculation, leaving 20 GiB. If the actual allocator requires a reserve, has partially used pages, or stores more metadata, even that count may be too optimistic. This is a planning bound, not an admission policy.

## Test usefulness, not just retrieval of one needle

Include retrieval at several positions, multiple relevant facts, distractors, instruction following, answer synthesis, and representative tool tasks. Compare short and extended configurations. A single needle-in-a-haystack result does not establish general long-context reasoning. [RULER](https://arxiv.org/abs/2404.06654) gives a broader research evaluation starting point; still add your application's tasks.

For a hardware experiment, sweep prompt lengths before maximum concurrency. Record token counts after the real tokenizer/template, TTFT, individual ITL, peak allocated/reserved memory, OOM/rejection behavior, and task outcomes. Keep failures in the report. No universal “128K recipe” can replace those measurements.

Next: [communication implementation](Lecture-07.md).
