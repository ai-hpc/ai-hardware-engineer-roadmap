# CPU lab: attention, bytes, and expert routes

In 60–90 minutes, prove four small inference contracts before trusting a large model deployment. Python 3.10+ is sufficient for the core exercises; an optional PyTorch comparison runs on CPU. No model weights, accelerator, service account, or network access is needed after obtaining these files.

## Run and predict

From this `labs` directory:

```bash
python3 -m unittest discover -s . -v
```

The standard-library suite contains 22 tests. Two PyTorch comparison tests run if PyTorch is installed; otherwise they explicitly skip. The course also executed all 24 tests with PyTorch 2.14.0+cpu. See [validation.json](validation.json) for the actual environment and source hashes.

Before opening [inference_contracts.py](inference_contracts.py), predict these answers:

1. How much BF16 KV payload does an 80-layer, 8-KV-head, 128-head-dimension model need for 131072 tokens?
2. If 16-token cache pages hold requests of lengths 1 and 17, how many slots are allocated?
3. A one-token decode query at position 2 has equal scores over values `[1,3,8]`. What should attention return? What does an upper-left one-row triangular mask return?
4. If one token selects two remote experts on the same rank, how many activation vectors must cross the network?

Answers: **40 GiB; 48 slots; 4 versus the incorrect 1; one vector if destination deduplication plus local fan-out is implemented, otherwise two.** Routing metadata and result/combine traffic are extra.

## Exercise 1: accounting with real configuration files

Read [model_sources.json](model_sources.json). It records three public configuration revisions and their SHA-256 hashes; these are configuration snapshots, not pretrained models. The gated Llama configuration was not downloaded.

`gqa_bytes` distinguishes explicit `head_dim` from a fallback `hidden_size/num_attention_heads`. Qwen3-235B-A22B has 4096 residual width, 64 query heads, and an explicit 128 head dimension. Using the fallback would undercount its cache by two. Its ideal BF16 KV payload is 192512 bytes/token. DeepSeek-V3.1's **compressed latent + RoPE-key layout** is 70272 bytes/token under the stated layout assumptions.

Change request lengths and page size. Explain why each request rounds separately. Then inspect `grouped_weight_bytes`: it counts row-packed payload plus one scale per partial/full group. At 129 four-bit weights and a group size of 128, a row needs 65 payload bytes plus four scale bytes. This is an explicitly defined teaching format, not a complete model-file encoder.

`quantize_int4_group` executes the symmetric [-7,7] rounding example with ties-to-even. Test an all-zero group and values exactly halfway between codes. Reuse the scale calibrated on `[-1,1]` for a held-out value 3: it clips to 1. The tests also saturate extreme finite values before rounding, so an overflowing intermediate ratio cannot reach `round(infinity)`. This illustrates calibration mismatch and numerical handling; it is not an AWQ/GPTQ implementation.

`roofline_seconds` uses exact rational arithmetic and returns a lower bound from FLOPs and bytes. It does not include queueing, launch, communications, or allocator overhead. A capacity estimate is not a timing measurement.

## Exercise 2: cached attention must preserve positions

`causal_gqa` uses lists shaped `[token][head][dimension]`, grouped-query mapping, a stable softmax, and an absolute query offset. It materializes scores and intentionally favors readability over speed. The small inputs are finite and moderate; this is not a production numerical kernel or a RoPE implementation.

The tests compare full-prefix execution with chunks of 1, 2, 3, and 7, verify future values cannot affect past outputs, and exercise different KV groups. The single-query counterexample is especially useful: both incorrect and correct outputs are finite, so checking only for NaNs misses the bug.

The optional [PyTorch comparison](test_torch_attention.py) expands K/V heads explicitly and uses an explicit boolean mask for cached chunks. It tests the documented non-square `is_causal=True` alignment against the intended absolute position. Set dropout to zero. Passing these cases establishes this tiny attention contract, not equivalence of a complete LLM or a GPU kernel.

## Exercise 3: draw the dispatch matrix

Use two ranks with owners `[0,0,1,1]`. Routes `[[0,2,3],[2,3]]` start at ranks `[0,1]`. With width 8 and two-byte values, remote assignment payload is 32 bytes; destination-deduplicated payload is 16. Expert loads total five assignments and rank loads are `[1,4]`.

The matrix counts remote vectors from source row to destination column. It excludes metadata, packing, headers, and combine traffic. Try a balanced route and a hotspot with the same assignment count. Explain why equal byte totals do not imply equal latency. `expert_reference` checks normalized weighted combination and token order using scalar toy experts; it does not implement DeepSeek's complete router.

## Exercise 4: verify a speculation probability identity

For target `p=[4/5,1/5]` and draft `q=[1/2,1/2]`, enumerate accepted and rejected mass using `speculative_one_step_distribution`. The exact result is p. The tests also cover disjoint support, identical distributions, and all 121 pairs on a two-token probability grid.

This is one proposal's rejection-sampling identity, not a multi-token speculative decoder. It does not manage logits processors, cache rollback, an all-accepted bonus token, or drafting/verification performance. Those remain requirements for a real serving implementation.

## Retain a learning artifact

Keep your predicted answers, one changed case, and the cached-mask failure. For a hardware follow-up, add model/runtime pins, actual memory allocations, a trace, and measured latency. Do not label these CPU arithmetic/reference tests as GPU performance evidence.
