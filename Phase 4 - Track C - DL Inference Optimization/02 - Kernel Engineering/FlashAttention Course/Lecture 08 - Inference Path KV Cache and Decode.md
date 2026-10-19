# Lecture 8 — Decode, cached positions and the two halves of KV

**Course:** [FlashAttention](Guide.md) · **Reviewed:** 18 October 2026

**Goal:** preserve positions and cache semantics while calculating decode resources. **Artifact:** CPU cache/mask/RoPE tests and a clearly bounded optional device experiment.

## 1. Describe the actual query positions

Prefill often processes many queries. Ordinary autoregressive decode often processes one new query, but chunked prefill and speculative decoding can have several. State query length L, live key length S, query heads Hq, KV heads Hkv and head dimensions separately.

A query for the newest token should attend to the intended cached history. With `L=1,S=5`, upper-left causal alignment allows only key 0; lower-right alignment allows all five keys. PyTorch SDPA's `is_causal=True` uses upper-left alignment for rectangular inputs. The pinned FA2 interface uses bottom-right alignment. The CPU tests expose this difference with real PyTorch calls. [SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention) · [pinned FA2 interface](https://github.com/Dao-AILab/flash-attention/blob/965c9212e6be33c821ee9f2b461564c5c91a2503/flash_attn/flash_attn_interface.py).

For an unpadded single current query with all keys valid and no future keys, noncausal attention over the live cache also permits all keys. If the cache tensor includes unused capacity or padding, mask it or supply correct live lengths; do not attend to the entire allocation.

## 2. Count both K and V

For equal K/V head dimensions D and element width b:

```text
KV capacity = 2 × batch × layers × live_tokens × Hkv × D × b
```

At 16,384 tokens, 8 KV heads, D=128 and BF16, one layer needs **64 MiB** for K+V. Across 32 layers that is **2 GiB**, before allocation metadata, padding or other tensors. The earlier 32 MiB value counted only one of K or V.

An illustrative bandwidth of 4.8×10¹² bytes/s would give about **0.447 ms** merely to read 2 GiB once. This is an idealized traffic floor, not a measured model decode time. A full model also reads weights, performs other operators, schedules work and may incur communication. A single attention-kernel duration is neither full-model tokens/s nor request time-to-first-token.

GQA uses fewer KV heads than query heads. The actual CPU test checks `enable_gqa=True` against explicit K/V head repetition. The capacity reduction follows Hkv; the realized bandwidth benefit depends on reuse. MQA/GQA are model architecture choices, not a universal quality ranking or an inference-only switch for arbitrary pretrained models.

## 3. Separate paging from attention semantics

Paging maps logical token positions to physical blocks. It can reduce allocation waste and support sharing; it does not inherently reduce the mathematical number of attended tokens or guarantee faster kernels.

The portable test reconstructs a sequence from a deliberately reordered page table and a partial final page. It checks metadata bounds. This is a **logical indexing fixture**, not an implementation of a GPU paging API.

Page-size restrictions are API-specific. In the pinned FA2 `flash_attn_with_kvcache` documentation, `page_block_size` must be a multiple of **256**. Do not copy a serving engine's 16-token page size into that interface and assume compatibility. Different FlashInfer/FA3/FA4/runtime APIs need their own documentation and tests. Cache update may mutate buffers; benchmark repeated fixed positions and growing generation as different workloads.

## 4. RoPE must preserve the position convention

Our NumPy helper implements one explicitly limited variant: adjacent dimension pairs, all dimensions rotated, base 10,000 and absolute integer positions. It checks a full causal pass against an incremental final-token pass. The cached keys are rotated once at their original positions; the new query/key use the new absolute position.

Resetting the decode query's position to zero produces a detectable disagreement. Rotating already-rotated cached keys again is another error. Model-specific scaling, partial rotation and split-half/interleaved layouts require their own config and vectors; this fixture is not a universal Llama/Qwen tokenizer/model compatibility test.

## 5. Design a fair device comparison

Compare identical live tokens, masks, dtype, head mapping, query positions and output semantics. Allocate metadata outside kernel-only timing; separately report end-to-end append/dispatch costs. Reinitialize or deliberately advance mutated caches. Record whether graph capture is used, and test the same path after capture.

**Answer check:** paging within 10–15% of a contiguous implementation is not a guaranteed result. Low achieved bandwidth does not identify one unique bottleneck. A trace and a controlled variation must establish whether launch overhead, insufficient parallelism, indirect accesses or another resource dominates.
