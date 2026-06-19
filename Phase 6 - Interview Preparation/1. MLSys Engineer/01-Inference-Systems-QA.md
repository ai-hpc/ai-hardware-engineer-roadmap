# MLSys Engineer — Inference Systems Q&A

**Collection:** [MLSys Engineer](README.md) | **Up:** [Interview Preparation](../README.md)

`#kv-cache` `#attention` `#speculative-decode` `#latency` `#trt-llm` `#edge`

---

> These answers are written at senior/staff level. Each one has a problem statement, a mechanism, the real engineering insight, and at least one production gotcha. In an actual interview, aim for 3–4 minutes per answer — long enough to hit the key tradeoff, short enough that the interviewer doesn't interrupt before you get there.

---

## Q1. How does PagedAttention work?

**The problem it solves:** naive serving reserves a contiguous `max_seq_len` KV buffer per request at allocation time. That wastes memory two ways: internal fragmentation (unfilled slots in the reserved block) and external fragmentation (you can't repack the space between requests). vLLM measured 60–80% effective memory waste in this model.

**The mechanism:** PagedAttention applies OS virtual-memory paging to the KV cache. The cache is partitioned into fixed-size **blocks** (typically 16 tokens of K and V per block). Each sequence has a **block table** — a mapping from logical token position → physical block index — that is maintained by a CPU-side block manager. The attention kernel is rewritten to gather K and V through that indirection rather than accessing a contiguous buffer.

**Why this helps:**

- Internal fragmentation is bounded by one block (≤15 wasted token slots per sequence instead of up to `max_seq_len`)
- Blocks need not be contiguous in physical memory → the allocator can reuse scattered pages
- Blocks with identical content can be **copy-on-write shared**: beam search branches share prefixes until they diverge; parallel samples share the prompt; persistent system-prompt prefixes are shared across all requests (**prefix caching**)
- KV growth is fully dynamic — no pre-commitment to a sequence length

**The kernel rewrite:** the core change is in the attention kernel's K/V access pattern. Where a contiguous kernel indexes `K[seq_offset + i]`, a paged kernel dereferences `K[block_table[i // block_size] * block_stride + (i % block_size)]`. This indirection costs a few registers but is negligible against the HBM bandwidth bound.

**Production caveat — prefix caching validity:** cache entries are keyed by token-id hashes. A single BOS/EOS or system-prompt variation invalidates the entire prefix. In multi-tenant serving, cache hit rate is heavily workload-dependent — benchmark it on your actual traffic before claiming prefix-cache speedup.

**Edge context:** on a single-stream edge device the multi-tenant win is smaller. I sized a pre-allocated pool to `actual_context_budget` instead of adopting paged blocks. But the prefix-sharing idea transfers directly: my persistent cross-turn KV buffer (hydrated from disk on warm start) is the same insight — don't re-prefill a prefix you already computed.

---

## Q2. How would you implement EAGLE-3 in TensorRT-LLM?

**What EAGLE-3 does:** EAGLE drafts at the feature level rather than the token level. A tiny draft head predicts the target's **hidden state** for the next position, then samples from the draft logits to build a tree of candidate tokens. The target verifies the entire tree in one forward pass using a tree attention mask. EAGLE-3 specifically drops the feature-regression loss constraint of EAGLE-1/2 and instead fuses hidden states from low, mid, and final target layers as the draft input — this raises acceptance length without the regression constraint's instability.

**Concretely in TRT-LLM, phased:**

**Phase 1 — expose target hidden states.** Modify the base model's TRT engine to emit hidden states from three layers (early ≈ L/4, mid ≈ L/2, near-final) as additional outputs. In TRT-LLM's Eagle implementation (`eagle_base`), this is the `emit_hidden_states` flag — the base engine exports concatenated hidden states from those checkpoints alongside the logits.

**Phase 2 — draft head.** A small 1–2 layer transformer (the EAGLE draft head) takes the concatenated hidden states as input (no context re-encoding) and autoregressively builds a draft token tree. The tree has configurable width and depth; typical: 4–6 candidates at depth 1, 2–3 at depth 2 (total 10–20 nodes).

**Phase 3 — tree attention.** Verification runs the target once on all tree nodes simultaneously. Each node attends only to its ancestors in the tree — implemented as a custom `attention_mask` (upper-triangular within the tree, with the appropriate parent-child structure) plus custom `attention_pos_id` so that each node's position encoding matches its depth in the causal prefix. TRT-LLM's attention plugin accepts `attention_mask` and `position_ids` overrides for exactly this case.

**Phase 4 — accept + KV compaction.** Walk the tree from root, accept the longest matching prefix (same stochastic acceptance rule as standard spec-decode), keep only the accepted nodes' KV entries, discard rejected branches. Compacting the tree KV → linear KV is the fiddly part: it's an in-place gather over the KV buffer, keyed by the accepted token's tree path.

**Phase 5 — register as a drafter.** Wire into TRT-LLM's spec-decode scheduling loop: draft N trees → verify batch → accept → continue. At batch=1 the payoff is large because the target forward is weight-read-bound — verifying K tokens costs roughly the same HBM as decoding 1, so you get K accepted tokens for ~1 token's bandwidth cost (at good acceptance rates).

**Gotcha:** EAGLE's feature dependence means `draft(t+1)` cannot start until `verify(t)` completes (it needs the target's hidden states from t). Unlike token-level external draft (1B model), you cannot overlap draft and verify across steps. On edge hardware with low parallelism this reduces the wall-clock benefit.

---

## Q3. How do you optimize KV cache on memory-constrained devices?

**Context:** my Orin work targeted 8 GB shared LPDDR5 with Gemma 4 4B — every byte matters.

**Priority order:**

**1. Quantize KV to INT8.** Per-token scaling (scale = `max(|K_row|) / 127`), dequantize before attention softmax. ~50% memory reduction vs FP16, and FP16-ULP-bounded drift on typical activations. I shipped it as default for most models.

*Gemma 4 exception:* Gemma 4's V activations have ~10× RMS outliers in certain layers. Per-row INT8 collapses these into clipped values, causing visible quality degradation on long-context summarization. I force FP16 KV for Gemma 4 and document it. Lesson: outlier-aware schemes (keeping a few per-channel FP16 slots) generalize the INT8 approach — but validate per-architecture.

**2. Exploit GQA/MQA.** Each KV head reduction cuts KV memory linearly. Gemma 4 4B uses 4 KV heads (vs 8 Q heads) → 2× KV reduction at the architecture level, before any quantization. GQA is the single highest-leverage architectural knob for KV memory.

**3. Account for KV-sharing layers.** Gemma 4 has 34 transformer layers, of which only 15 **own** their own KV (the trailing 19 layers reuse a prior layer's K/V via the `kv_shared_layers` mapping). Allocate the KV pool for 15 layers, not 34 — a 57% reduction in pool size vs naive accounting.

**4. Cap sliding-window KV.** Gemma 4's local-attention layers only need the last W=1024 tokens, not the full context. Implement the KV buffer as a ring of W entries per local layer. Local layer KV is O(1) regardless of context length.

**5. Pre-allocate the pool + OOM guard.** On edge, never let KV grow unbounded — allocate the full pool at startup with a hard limit, refuse new requests if the pool is full rather than risking mid-generation OOM. Account for:

```text
pool_bytes = num_own_layers × num_kv_heads × max_ctx_tokens × head_dim × dtype_bytes
           + overhead (block tables, metadata)
```

**6. Persist prefixes (prefix caching).** Hydrate the system-prompt / persistent context KV from disk on warm start. This cuts TTFT on turns 2–N and avoids re-prefilling context you already computed. My measured win: 877 ms cold TTFT → 444 ms warm TTFT on a 1261-token context.

**7. Layout for coalesced access.** Store KV in `[layer, head, token, dim]` order so that a single warp reads one head's token slice contiguously. A `cache_head_dim` stride field handles ragged head dims (Gemma 4's 256-d sliding K stored in 512-wide slots for alignment).

**8. Eviction (last resort).** StreamingLLM-style sink+recent eviction (keep first S tokens + last W tokens) or H2O (heavy hitters) for extreme context. These degrade output quality — document the tradeoff and measure it.

---

## Q4. How would you write a fused attention kernel?

**Core idea (FlashAttention tiling):** never materialize the full N×N attention score matrix. Tile Q; loop over K/V tiles; for each tile: compute `S = Q·Kᵀ` in registers or SRAM, apply online softmax, accumulate `O += P·V`, write O once at the end.

**Online softmax:** maintain per-query running max `m` and running denominator `l`. When a new K/V tile produces a new max `m_new > m_old`, rescale the existing O accumulator and `l` by `exp(m_old - m_new)` before adding the new tile's contribution. This keeps numerical correctness without materializing S.

**Engineering details that matter on real hardware:**

**Tensor cores (MMA):** use `m16n8k16` or `m16n8k32` MMA atoms (fp16/bf16 input, fp32 accumulate) for both `Q·Kᵀ` and `P·V`. Keep the softmax intermediates and O accumulator in fp32.

*Precision trap I hit on Gemma 4:* fp16 accumulation in the `P·V` matmul breaks on V-outliers — the accumulated sum clips. fp16-inputs/fp32-accumulate is fine (measured rel-RMS 3e-4 even at 20× outlier amplitudes). Rule: always accumulate in fp32; only the MMA inputs can be fp16/bf16.

**Double-buffer K/V tiles:** issue `cp.async` loads for tile `t+1` while tile `t`'s MMA is executing. This overlaps global-memory latency (HBM access) with compute. Measured impact on D=512, 4K context on Orin: 1289ms → 348ms — by far the single biggest latency win in my kernel.

**Skip fully-masked tiles:** for sliding-window attention, the earliest relevant K tile starts at `kt_start = ((q_pos + 1 - W) / KT) * KT`. Don't even issue a load for tiles that are entirely outside the window — they contribute zero to the output and you save both HBM bandwidth and MMA cycles.

**Occupancy:** keep the O accumulator in **registers**, not shared memory. Shared-memory O limits occupancy to ~1 active block per SM because the shared allocation per block is too large. The cost is that the online-softmax rescale must operate directly on MMA register fragments — you need to know the lane-to-output-row mapping for your specific MMA atom (`lane gid` owns rows `{gid, gid+8}`, columns `{2t, 2t+1}`).

**Roofline first:** on Orin's 8-SM iGPU I measured my attention kernel as bandwidth-bound (arithmetic intensity ≈ 16–32 FLOP/byte, far below the ~190 FLOP/byte ridge point). The real win was **queries-per-block**: launching 16 queries that share each loaded K/V tile = 16× amortization of DRAM traffic, not faster MMA. Don't reach for tensor cores when the bottleneck is bytes.

---

## Q5. Why does FlashAttention reduce HBM traffic?

**What naive attention does:** write `S = QKᵀ` (N×N fp16 matrix) to HBM → read it back for softmax → write `P` (N×N) → read `P` for `P·V`. At sequence length N=4096, D=128, that's ~4 GB of HBM traffic just for the attention matrix on one layer — and softmax is purely memory-bound over it.

**What FlashAttention does:** tile the computation so that `S` and `P` tiles never leave SRAM/registers. The kernel reads Q, K, V essentially once from HBM and writes O once. HBM traffic drops from O(N² · d) to O(N · d).

**The tradeoff:** more FLOPs. The online-softmax rescale is extra arithmetic; the backward pass recomputes S tiles instead of stashing them. But the key insight is that attention is **memory-bound at typical N and D**, so trading abundant FLOP for scarce bandwidth is always correct. The bottleneck moves off HBM bandwidth.

**When FlashAttention stops helping:** very short sequences (N < 512) where the N×N matrix is small enough to fit in cache anyway, or extreme `D` (head_dim=256 or larger) where the Q·Kᵀ arithmetic intensity finally pushes toward compute-bound. Gemma 4's head_dim=256 is right at the edge — at N=4096 it's still bandwidth-bound, but at N=512 with D=256 the tile compute starts to dominate.

**Backward pass:** extends the same idea. Instead of storing P (N×N activations) for the backward, Flash recomputes S in SRAM during the backward pass. Memory for backward drops from O(N²) to O(N). This is why FlashAttention dramatically improves long-context training memory efficiency too.

---

## Q6. How would you schedule speculative decoding on Jetson?

**The opportunity:** Jetson decode at batch=1 is weight-read-bound — the GPU idles on LPDDR5 bandwidth while streaming weights. Verifying K draft tokens in one target forward costs roughly the same HBM as decoding 1 token, because the weight-read bottleneck is the same either way. If K tokens are accepted on average, you get K× throughput for ~1× bandwidth cost.

**Draft model selection:** a tiny draft that shares the target's tokenizer. A 1–2-layer EAGLE feature-draft or Medusa heads are better than a separate 1B model — a full second model competes for the same LPDDR5 bandwidth pool, eroding the gain. The shared LPDDR5 (CPU + draft + target) is the key constraint on Jetson that doesn't exist in datacenter (NVLink-isolated HBM).

**Economics check:** `net_win = accept_len × target_cost - (draft_cost + verify_cost)`. Net win only if `accept_len > 1 + draft_cost / target_cost`. Measure acceptance rate first on your actual task distribution; tune tree depth and width to the measured value. For code/technical text, acceptance is typically 3–4; for reasoning/math it's lower.

**The inner loop:**

```text
while generating:
    draft K tokens (chain or tree) using tiny draft head
    one target forward with tree mask (verify all K in parallel)
    accept longest matching prefix
    repeat
```

**CUDA graphs:** capture the target decode step as a CUDA graph to eliminate kernel-launch overhead (~100–200 µs on Orin per step). The problem: accepted length varies each step, so the graph has variable-length KV updates. Work around it with a padded fixed-length tree + mask-out (accepted path gates the KV write), or maintain separate captured graphs per accept-count. I guard graph capture off for Gemma 4's per-token PLE (Per-Layer Embeddings) because PLE is position-dependent and the dynamic position breaks static graph capture.

**Power/thermal:** speculative decode increases GPU utilization — the draft + verify steps both use the GPU, versus decode-only which leaves the GPU memory-idle between HBM fetches. On Jetson's 15–25W budget this can cause thermal throttling. Always report tok/s at sustained locked clock (`jetson_clocks --store; jetson_clocks`), not peak burst.

**Realistic numbers:** ~1.5–2.5× decode throughput on a 2–4B memory-bound model at good acceptance rates. For Gemma 4 4B + E2B drafter on Orin: measured ~90 tok/s vs ~55 tok/s baseline.

---

## Q7. How would you reduce TTFT on Orin Nano?

**TTFT = model load time + prompt prefill time.** Both are levers; prefill dominates for non-trivial prompts.

**Prefill throughput (primary lever):**

The key failure I found and fixed: a **per-token fallback** for prompts over 1024 tokens. The system was calling `decode_one_token()` in a loop instead of `prefill_batch()` for long prompts — that's O(N) sequential forward passes instead of one batched pass. Fixed by chunking any prompt into `scratch_budget`-sized pieces and passing each as a batch. Measured result: 152 → 620 tok/s on a 1261-token Gemma 4 prompt.

Other prefill wins:
- **Tensor-core GEMMs:** INT8 MMQ for quantized projections (Q, K, V, out, gate, up, down), fp16/fp32 for attention. This moves the GEMM from scalar CUDA cores to tensor cores.
- **Tensor-core attention:** replace the scalar `QKᵀ` loop with cuBLAS batched GEMM (fp16-in, fp32-accumulate). At prefill batch=1 but N>512, this dominates.
- **Batched RoPE+KV-store:** apply RoPE to all prompt tokens in one kernel call and store all K/V in one pass. Avoid per-token RoPE in a Python loop.
- **Skip fully-masked sliding-window tiles** (see Q4).

**Prefix caching (biggest single win for chat):**

Hydrate the system-prompt / conversation KV from disk instead of re-prefilling on every turn. My measurement: 877 ms cold TTFT → 444 ms warm TTFT on a 1261-token context. Implementation: after prefill, serialize KV to a memory-mapped file; on warm start, mmap + `cudaHostRegister` to feed GPU directly from the page cache (zero-copy if pages are hot).

**Model load time:**

`mmap` the GGUF file and `cudaHostRegister` the weight pages → GPU can read directly from the mapped file (zero-copy DMA). Cold load is NVMe-bound; warm load (process restarts with hot page cache) is ~1.3 s for Gemma 4 4B INT4. Keep the serving process warm — don't restart between requests.

**Clocks:**

Lock MAXN_SUPER + `jetson_clocks` to avoid DVFS ramp-up on the first inference burst. Without clock locking, the first prefill pays a ~200ms DVFS ramp that llama-bench's warm-up pass discards — making benchmarks look better than production.

**Prompt compression:**

Bake persona/system/tool-definition content into LoRA weights (adapter) so fewer tokens hit the prefill path. A 500-token system prompt costs ~50ms on Orin; encoded in a LoRA adapter it costs 0 prefill tokens.

**Always measure the actual cold path** — `llama-bench pp` discards a warmup step, but your users pay a real cold start. Profile with `nsys profile --trace cuda,osrt` from a cold process.

---

## Q8. How would you integrate a new model architecture into TensorRT-LLM?

**Context:** this is exactly the Gemma 4 → TensorRT-Edge-LLM port I completed, including divergent features: per-layer-type head dims, dual RoPE, QK+V norm, unit attention scaling, Per-Layer Embeddings, KV-sharing, GeGLU, soft-capped logits.

**Phase 1 — Recognition + config parsing.**

Map `config.json` `model_type` → a parsed config object. Field checklist: head dims (may differ per layer type), RoPE params (`rope_theta`, `rope_scaling`, whether partial or full), layer type annotations (`attention_type` list for interleaved local/global), norm types (RMSNorm vs LayerNorm, placement), KV-sharing (`kv_shared_layers` mapping), soft-cap (`attn_logit_softcapping`). Register `model_type → model_class`. Fail loud on unrecognized features — a checkpoint that silently mis-builds as Llama will run, produce garbage, and take hours to debug.

**Phase 2 — Python model definition.**

Compose the model from the framework's modules: `QuantizedLinear`, the attention plugin, RoPE op, RMSNorm. Only write custom code where the arch genuinely diverges.

*Gemma 4 divergences and their solutions:*

| Feature | Solution |
|---------|---------|
| Per-layer-type head dims (local: 256, global: 512) | Parameterize `head_dim` in the attention plugin call; pass `layer_type` index |
| Dual RoPE (two separate cos/sin tables) | Two model inputs; precompute both tables at runtime init |
| QK-norm + V-norm | Insert RMSNorm on Q and K before attention dot-product; RMSNorm on V after projection |
| Unit attention scaling | Pre-scale Q by `√head_dim` before RoPE to cancel the plugin's built-in `1/√head_dim`; verify RoPE commutes with this scalar (it does — RoPE is a rotation, scaling is a scalar multiply) |
| Per-Layer Embeddings | Second embedding pathway: an extra integer input (PLE token) + in-graph embedding lookup + add to the residual stream |
| KV-sharing (trailing layers reuse prior KV) | Pass the physical KV buffer address of the source layer instead of allocating new KV; implement via `kv_shared_layers` aliasing in the pool |
| GeGLU | Gate proj + up proj double-wide linear, then `gate ⊙ gelu(up)` in a single fused kernel |
| Soft-cap | `logits = tanh(logits / cap) * cap` applied after the lm_head |

**Phase 3 — Export contract.**

For ONNX→TRT stacks: define the forward signature — inputs, dtypes, dynamic axes, names. New arch features become new graph inputs that the runtime must supply (I added a second RoPE table input, a PLE integer input, and `layer_type` as a constant in the graph). Export via the dynamo exporter with the custom TRT op translation table; structurally validate (instantiate + export → check the ONNX graph with `onnx.checker.check_model`) before writing a single line of runtime C++.

**Phase 4 — Plugins + runtime.**

Map new ops to TRT plugins: dual-RoPE precompute (extend the RoPE fuser), PLE lookup (new embedding plugin), KV-share aliasing (extend the cache manager). Each new plugin is a risk — unit-test each one in isolation against a reference (HuggingFace or numpy).

**Phase 5 — Weight loading.**

Map HuggingFace checkpoint key names → TRT-LLM parameter names. Watch for: tied embeddings (lm_head shares `embed_tokens.weight`), per-layer-type weight shapes (local head != global head), dropped weights on KV-sharing layers (those layers have no KV projections in the checkpoint).

**Phase 6 — Validate.**

Greedy-identical check: compare token-by-token against HuggingFace on 5 factual prompts with `temperature=0`. Per-layer tensor comparison to isolate regressions (a `model.forward(return_all_hidden_states=True)` comparison is the fastest debug tool). Long-context test at max supported length. Then perf: prefill throughput, decode tok/s, KV memory accounting.

**De-risk strategy:** I got greedy-identical to `llama.cpp` in a simpler GGUF runtime first. That validated the math independently of TRT mechanics. The TRT port's risk then was mechanical (graph construction, plugin wiring) not mathematical — much easier to debug.

**One concern per PR:** recognition → modeling+export → runtime → perf. Don't mix weight-loading bugs with attention-plugin bugs — you won't be able to isolate them.

---

## Quick-fire calibration questions

These test your ability to give a precise 30-second answer — the interview equivalent of checking your units.

**Q: What is the arithmetic intensity of a decode-phase GEMM, and what does it imply?**

A: AI = `2·M·K·N / (M·K + M·N + K·N) bytes` at batch=1 (M=1): `2·K·N / (K + N + K·N) ≈ 2 / (1/K + 1/N)`. For K=N=4096, AI ≈ 2 FLOP/byte. The H100's ridge point is ~300 FLOP/byte — decode GEMM is ~150× below the ridge. It's entirely memory-bandwidth-bound. Implication: faster math units do not help; more HBM bandwidth or weight quantization (fewer bytes to stream) does.

**Q: What's the difference between tensor parallelism and pipeline parallelism?**

A: TP splits each layer's weight matrices across devices (column-parallel + row-parallel for MLP; Q/K/V heads across devices for attention) — every layer runs on all devices, communicating with `AllReduce` after each layer. Latency-friendly, bandwidth-expensive. PP splits the model depth — different layers live on different devices, connected by point-to-point `send`/`recv`. Lower communication volume but adds pipeline bubbles (devices idle while waiting for their stage's input). For large-batch serving, PP reduces the AllReduce pressure per step; for latency-critical single-request paths, TP typically wins because it has no pipeline idle.

**Q: Why is GQA KV memory smaller than MHA?**

A: MHA stores one K and V head per query head. GQA groups G query heads to share one K/V head. Memory: MHA → `num_heads × 2 × seq_len × head_dim`; GQA → `num_kv_heads × 2 × seq_len × head_dim` where `num_kv_heads = num_heads / G`. For Gemma 4 4B (G=2): half the KV memory vs MHA. Throughput: the KV HBM read per step is halved → 2× decode throughput ceiling from bandwidth alone.

**Q: What is the difference between Q8_0 and Q4_K_M in GGUF?**

A: Q8_0 is 8-bit integer per-block quantization with one FP32 scale per 32-element block. ~8.5 bits per weight effective. High quality, moderate compression. Q4_K_M is a k-quant: 4-bit (M = mixed, some critical layers use 6-bit), grouped scales and mins per 256-element block using a 6-bit sub-quantization for the scales. ~4.5 bits per weight effective. Quality loss is surprisingly low on most models because of the scale sub-quantization. Q4_K_M is the standard recommendation for edge deployment — it fits Gemma 4 4B in 2.2 GB vs 8.2 GB BF16.

---

*Up: [MLSys Engineer](README.md) | Back to: [Interview Preparation](../README.md)*
