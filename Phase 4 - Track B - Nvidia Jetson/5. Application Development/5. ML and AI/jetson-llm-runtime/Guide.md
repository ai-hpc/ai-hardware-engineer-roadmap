# Experimental native LLM runtime: audit correctness before speed

<div class="course-identity auto-course" style="--course-accent: #65a30d; --course-accent-rgb: 101, 163, 13;" markdown="1">
<div class="course-identity__icon">JLRM</div>
<div markdown="1">
<p class="course-identity__eyebrow">Deep Dive · Jetson Track</p>
<p class="course-identity__title">Specialized course identity for Jetson LLM Runtime — Memory-First Inference Engine.</p>
<p class="course-identity__meta">Artifact: Jetson integration demo · Measure: latency, memory, power, logs</p>
</div>
</div>

**Reviewed 18 October 2026.** This course's [native C++/CUDA project](../../../../Projects/jetson-llm-runtime/README.md) is an experimental implementation for studying model loading, memory accounting, tokenization and kernel validation. Source files and test names do not by themselves establish coherent generation, complete model support or a performance improvement.

Read the project's current [testing/status record](../../../../Projects/jetson-llm-runtime/TESTING.md) for checks actually executed. This lesson does not claim a Jetson GPU run, supported production model matrix, or superiority over an established runtime.

Start with the [native CPU lab](../../../../Projects/jetson-llm-runtime/labs/README.md): build the actual C++ sampler, GGUF parser and configuration reader, then inspect small generated binary fixtures. The tokenizer deliberately supports only the `course.greedy.bytes` teaching profile; ordinary pretrained tokenizer profiles are rejected. Passing those component tests does not establish pretrained text generation.

## 1. What a runtime must preserve

A working decoder joins several contracts:

```text
bytes and metadata → typed tensors → model equations → logits
text → exact tokenizer/template → token IDs → positions/masks
logits → sampling → token IDs → exact detokenizer → text
```

Every arrow can fail independently. Matching a generated sentence on one prompt is weak evidence: it may conceal a wrong vocabulary, transposed weight, cache position or attention grouping.

The [CPU ONNX lab](../cpu-export-quantization/Guide.md) offers an executable example of reference output comparison, shape checks and deliberate failures. Apply that discipline to each native component.

## 2. Tokenization is part of the model

A greedy longest-string-match tokenizer is not automatically equivalent to BPE, SentencePiece or another model tokenizer. Merge ranks, normalization, byte fallback, whitespace, added tokens, BOS/EOS rules and chat templates matter. The model was trained on particular IDs, not just visually similar text.

Create golden token-ID fixtures from the model's pinned reference tokenizer for empty text, repeated spaces, line breaks, non-ASCII text, literal special-token strings and a complete conversation. Compare both encoding and decoding. Preserve unsupported input as an explicit error rather than silently dropping unknown bytes.

**Worked failure:** vocabulary contains `a`, `b`, `c`, `ab`, `bc`. Greedy longest match on `abc` can choose `ab,c`; BPE merge priorities can choose `a,bc`. Both decode to the same string but feed different embeddings to the transformer. Round-trip text equality alone cannot validate tokenization.

## 3. Parse a model as structured binary data

A GGUF loader must validate magic/version, bounded counts and string lengths, tensor shapes/types, alignment, byte spans, overflow and truncated reads. Do not guess the width of an unknown metadata type and continue: one wrong skip shifts all following fields.

Distinguish a file-level quantization label such as Q4_K_M from each tensor's actual encoding. A mixed quantization can contain several tensor types. A loader supporting one block decoder cannot claim support for the entire file based only on its filename.

Use tiny generated fixtures before large files: a valid single tensor, one truncated metadata array, an overflowing dimension product, an unknown type, a tensor span beyond EOF and an incorrect alignment. Require a clear failure result and no invalid memory access. See the upstream [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md).

## 4. Compare kernels with independent references

For GEMV, use a non-square matrix with positive/negative values and compare against a CPU double-precision sum. For quantized GEMV, validate the block dequantizer separately before testing multiplication. For attention, use a small example with multiple query heads sharing fewer KV heads, several cache lengths, a causal mask and large logits that require stable softmax.

Test prompt prefill followed by incremental decode against full-sequence recomputation. Correct single-token attention does not prove that cache offsets, RoPE positions or graph replay update properly across tokens.

Declare absolute and relative tolerances per dtype and operation. Use finite checks and compare near-zero values sensibly. Floating-point reduction order changes are not necessarily bugs; a large change hidden by cosine similarity may still change the selected token.

## 5. Query hardware rather than invent constants

Orin Nano has no DLA. Its memory capacity and bandwidth are shared resources; a power mode changes clocks and limits, not the physical number of GPU units. Read the actual device properties for SM count, shared-memory capacity and supported features, and verify the exact module specifications.

A roofline ridge point needs consistent units. If a hypothetical peak is 67 TOPS and bandwidth is 102 GB/s, the ratio is about **657 operations/byte**, not 0.657. Sparse INT8 TOPS cannot be used as the dense FP16 peak without changing the comparison. Use the [memory lesson](../../../1.%20Nvidia%20Jetson%20Platform/Orin-Nano-Memory-Architecture/Guide.md) and [current compatibility record](../../../Compatibility.md).

Pinned or mapped host memory is not an extra physical memory bank. CPU/GPU accessibility does not establish cache coherence behavior, registration eligibility, lifetime, or performance. Check the allocation contract and synchronize ownership before reuse.

## 6. Treat serving as another correctness boundary

A server needs bounded input size, valid request parsing, supported parameters, a defined error response, cancellation, request ownership and cache isolation. A route named `/v1/chat/completions` does not establish complete compatibility with every client's protocol.

Use two distinct prompts in alternating requests and verify their caches remain separate. Cancel a long request, then submit a short one. Test malformed JSON and out-of-range generation limits. Report unsupported features instead of accepting them and silently ignoring their values.

## 7. Completion artifact

Choose one native component. Document its input/output contract, add an independent reference case and a negative case, and run the applicable tests from [TESTING.md](../../../../Projects/jetson-llm-runtime/TESTING.md). Then explain what the result establishes and which GPU/model checks remain pending.

**Diagnostic answers:** a passing binary parser test does not prove valid inference; readable output does not prove tokenizer equivalence; a memory budget is not an allocation guarantee; and a faster wrong kernel is not an optimization. Continue with [LLM memory and measurement](../llm-optimization-jetson/Guide.md).
