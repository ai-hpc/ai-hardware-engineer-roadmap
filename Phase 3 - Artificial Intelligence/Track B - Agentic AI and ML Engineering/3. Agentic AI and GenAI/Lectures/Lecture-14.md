# Lecture 14 - Efficient Local RAG Stack: Qwen3.5-4B INT4 and Granite Embeddings

**Course:** [AI Agent Development 2026](../Guide.md) | **Previous:** [Lecture 13](Lecture-13.md) | **Next:** [Lecture 15](Lecture-15.md)

---

**Efficient local RAG** is not about using the largest model you can fit.

It is about spending memory and compute **where they improve answer quality**.

A strong edge RAG stack looks like:

```text
User query
  -> Granite embedding model
  -> Qdrant or pgvector
  -> top-k retrieval
  -> optional reranker
  -> compact retrieved context
  -> Qwen3.5-4B INT4 generator
  -> grounded answer
```

This architecture is useful for:

- Jetson Orin
- local/private AI
- edge agents
- coding assistants
- multilingual RAG
- low-power inference
- small office knowledge bases
- factory or robotics documentation assistants

The central idea:

```text
retrieval precision + small fast generator
beats weak retrieval + huge generator
```

Most RAG failures are **not model-size failures**.

They are **retrieval, chunking, context, and memory-budget failures**.

---

## Learning objectives

By the end of this lecture, you should be able to:

1. Design a local RAG stack around a 4B-class generator and compact embedding model.
2. Explain why Granite 97M Multilingual R2 is attractive for edge retrieval.
3. Choose between Qdrant and pgvector for local vector search.
4. Estimate memory pressure for INT4 generator deployment.
5. Choose chunk sizes for code, docs, PDFs, and manuals.
6. Explain why reranking matters more than dumping many chunks into the prompt.
7. Compare llama.cpp, vLLM, and TensorRT-LLM for local/edge RAG.
8. Use prompt constraints, KV-cache optimization, prefix caching, and speculative decoding to improve small-model RAG.
9. Avoid common efficient-RAG failure modes.

---

## 1. Target architecture

The recommended stack:

```text
generator:
  Qwen/Qwen3.5-4B

generator quantization:
  INT4 class quantization
  AWQ / GPTQ / GGUF Q4_K_M depending on runtime

embedding:
  ibm-granite/granite-embedding-97m-multilingual-r2

vector database:
  Qdrant for edge-first service
  pgvector if PostgreSQL integration is already required

retrieval:
  top_k = 8
  rerank to 3
  compress before generation

runtime:
  llama.cpp for embedded/low-RAM
  vLLM for server and batching
  TensorRT-LLM for maximum NVIDIA optimization work
```

This is not the only valid stack.

Treat it as a candidate architecture to compare with a simpler lexical baseline and the [executed local RAG lab](../Local-RAG-Workflow-Lab/Guide.md). No model or Jetson runtime was executed for this design lecture.

The goal is:

```text
good retrieval quality
  + low VRAM
  + short prompts
  + fast decode
  + private/local operation
```

---

## 2. Generator: Qwen3.5-4B

Qwen3.5-4B is a **4B-class Qwen model** available on Hugging Face.

The model card includes examples for Transformers, vLLM, SGLang, and Docker model runner usage.

For this lecture, the reason it is interesting is the size/capability tradeoff:

- small enough for local and edge experiments
- a candidate whose quality must be measured on the intended tasks
- useful for coding and multilingual tasks
- suitable for grounded answer generation when retrieval is good

Important caveat:

```text
Always verify the exact model variant, chat template, modality mode,
license, quantization artifact, and serving backend before deployment.
```

The Hugging Face card for `Qwen/Qwen3.5-4B` is tagged around image-text-to-text usage, while many local RAG stacks use text-only chat/completion paths.

That means you should validate:

- tokenizer behavior
- chat template
- thinking mode or reasoning controls
- vLLM support
- llama.cpp/GGUF support if using GGUF
- memory use at your target context length
- answer quality on your documents

Do **not assume all Qwen3.5-4B variants behave identically**.

---

## 3. Quantization and memory: calculate from the actual architecture

“INT4” is not a portable artifact format. AWQ, GPTQ and GGUF variants have different layouts, scales, calibration methods and runtime kernels. Verify support for this exact hybrid model, hardware, quantization artifact and runtime revision. A format supported for another Qwen family is not sufficient evidence.

The pinned [Qwen3.5-4B config](https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/config.json) has **32 text layers: 8 full-attention and 24 linear-attention layers**. Full attention has 4 KV heads and head dimension 256. Linear attention uses 32 value heads with 128-by-128 recurrent matrices; the configured state dtype is float32. This is not a 32-layer full-attention transformer.

For a batch of one with BF16 full-attention KV, excluding padding and allocator overhead:

```text
KV bytes/token = 2(K,V) × 8 layers × 4 KV heads × 256 elements × 2 bytes
               = 32,768 bytes = 32 KiB
```

| Token positions | Full-attention KV only |
|---:|---:|
| 4,096 | 128 MiB |
| 8,192 | 256 MiB |
| 16,384 | 512 MiB |

An illustrative recurrent-matrix term is `24 × 32 × 128 × 128 × 4 = 48 MiB` per sequence. Add convolution state, runtime-dependent layout/padding, possible speculative rollback copies, workspaces, multimodal components and allocator reserve. Verify the actual state implementation: this arithmetic is not a measured process-memory result.

Four billion uniformly packed 4-bit parameters would take 2 GB decimal, or about 1.86 GiB. Real artifacts include scales/metadata and often higher-precision tensors; the model's name is a rounded parameter count. Read the artifact's tensor table and file size. Weight-only quantization does not automatically quantize KV or recurrent state.

For total memory, account for loaded weights, full-attention KV, recurrent/conv state, activation/workspace peaks, index, embedding model, OS and other processes. On Jetson these compete in unified physical RAM. The previous unexplained “4K requires 4–5 GB / 16K requires 8–10 GB” table was not a measurement and should not be used as evidence.

**Worked exercise:** doubling context from 4K to 8K adds 128 MiB of this BF16 KV for batch one; doubling simultaneous sequences doubles both KV and per-sequence recurrent state. It does not double shared weights. Report the exclusions alongside the estimate.

## 4. Embedding model: Granite 97M Multilingual R2

Use:

```text
ibm-granite/granite-embedding-97m-multilingual-r2
```

IBM's model card describes it as a **97M-parameter dense embedding model** with:

- 384-dimensional embeddings
- up to 32,768-token context
- multilingual support
- code retrieval support
- Apache-2.0 license
- ONNX and OpenVINO deployment paths
- vLLM embedding serving support
- GGUF conversion option for llama.cpp-style embedding

Why it is a strong edge fit:

- much smaller than the 311M Granite multilingual variant
- lower memory pressure
- lower latency
- easier batching
- good multilingual retrieval quality for its size
- practical for Jetson and CPU-side embedding paths

The 311M version may improve quality in larger server deployments.

The [pinned Granite revision](https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2/tree/835ad14087e140460703cf0fae09f97d469d65c2) specifies 384 dimensions and a 32,768-position encoder. Its Sentence Transformers pooling configuration uses the CLS token, not mean pooling. Follow the published model pipeline, including normalization, rather than inventing a pooling method. A maximum sequence length is not a recommended chunk size or a latency guarantee.

Compare the 97M and 311M candidates on the same held-out queries and actual target CPU/GPU. Size alone does not establish lower end-to-end latency or better retrieval. OpenVINO GPU acceleration targets supported Intel devices; it is not a Jetson CUDA execution provider. ONNX Runtime likewise needs an installed provider compatible with the target stack.

---

## 5. Retrieval matters more than generator size

A small generator can answer well **if the prompt contains the right evidence**.

A large generator can **still fail if retrieval is poor**.

Bad retrieval causes:

- hallucinated answers
- overconfident missing information
- irrelevant citations
- excessive prompt length
- slow decode
- context window waste

The working rule:

```text
better top-3 evidence
  > bigger model reading 20 weak chunks
```

Good retrieval pipeline:

```text
authorize query scope
  -> embed query
  -> vector search top 8 with authorization/version filters
  -> rerank top 8
  -> keep top 3
  -> optionally compress
  -> generate answer
```

Do **not dump all retrieved chunks** into the generator.

Small models need **high signal**.

---

## 6. Chunking strategy

Chunking often matters **more than model selection**.

Recommended starting ranges:

| Content type | Chunk size |
|---|---:|
| code | 256-512 tokens |
| documentation | 512-1024 tokens |
| PDFs/manuals | 768-1536 tokens |

Overlap:

```text
10-20%
```

Common starting point:

```text
chunk_size = 512 tokens
overlap = 64 tokens
```

Why not giant chunks?

```text
giant chunks reduce retrieval precision
```

If every chunk contains too many topics, vector search **cannot identify the exact relevant passage**.

Bad chunking symptoms:

- retrieved chunks are broadly related but not answer-bearing
- answer requires many chunks
- reranker struggles to choose
- model sees too much irrelevant context
- citations point to generic sections

Use structure-aware chunking:

- preserve headings
- preserve code blocks
- preserve function/class boundaries
- include file path metadata
- include section title metadata
- include document version metadata

---

## 7. Vector database choice

### Qdrant

Use Qdrant when you want:

- edge-friendly vector search
- Rust implementation
- HNSW dense vector index
- payload metadata filtering
- standalone service deployment
- hybrid search options
- clean API for local agents

Qdrant is usually the better Jetson/local default when you do not already need PostgreSQL.

### pgvector

Use pgvector when:

- PostgreSQL is already part of the product
- SQL joins and relational metadata matter
- you want one operational database
- enterprise app integration matters more than raw vector-specialized deployment

The choice is not ideological.

Use:

```text
Qdrant:
  edge-first vector service

pgvector:
  SQL-first application integration
```

---

## 8. Reranking

Reranking is often the **highest-leverage quality improvement**.

Vector search gets **candidates**.

The reranker picks the **best evidence**.

Recommended pattern:

```text
retrieve top 8
rerank top 8
keep top 3
```

Small reranker candidates:

| Reranker | Good fit |
|---|---|
| bge-reranker-base | strong general quality |
| Granite reranker | IBM/Granite ecosystem |
| Jina reranker | multilingual workflows |

If latency is tight:

- rerank only top 8 or top 10
- run reranker on CPU if GPU is reserved for generator
- cache rerank results for repeated queries
- skip reranker for exact metadata hits

If quality matters:

```text
rerank before increasing generator size
```

---

## 9. Context management for small models

Small models are **sensitive to noisy prompts**.

Recommended final prompt structure:

```text
system instructions
  -> compact retrieved context
  -> user question
  -> answer format requirements
```

Target retrieved context:

```text
~2K tokens for many edge RAG tasks
```

Instead of:

```text
retrieve 20 chunks
send everything
hope the model finds the answer
```

Do:

```text
retrieve 8
rerank to 3
compress
answer with citations
```

Compression can be:

- extract only answer-bearing paragraphs
- remove boilerplate
- preserve headings and citations
- keep code snippets intact
- deduplicate repeated passages

Small models **reward discipline**.

---

## 10. Prompt design

Use **strict grounded prompts**.

Example:

```text
You are a retrieval-grounded assistant.
Answer only from the retrieved context.
If the context does not contain enough evidence, say "insufficient information."
Include citations using the provided source IDs.
Do not use outside knowledge unless explicitly asked.
```

Why this helps:

- reduces hallucination
- requests abstention, whose reliability must still be tested
- improves citation behavior
- prevents over-answering
- makes failures easier to detect

For code RAG:

```text
Use only the provided repository snippets.
If a function or file is not present in context, say which file is missing.
Do not invent APIs.
```

For multilingual RAG:

```text
Answer in the user's language unless the task requests otherwise.
Preserve technical identifiers exactly.
```

---

## 11. Runtime choices

### llama.cpp

Use when:

- Jetson or embedded deployment
- low RAM
- GGUF quantization
- CPU/GPU mixed execution
- simple local server
- offline/private deployment

Best for:

```text
Qwen3.5-4B Q4_K_M
4K-8K context
single-user local RAG
```

### vLLM

Use when:

- server deployment
- continuous batching
- multiple users
- OpenAI-compatible API
- embedding endpoint support
- model serving at higher concurrency

Best for:

```text
Qwen3.5-4B AWQ/GPTQ
Granite embedding endpoint
multi-user local server
```

### TensorRT-LLM

Use when:

- NVIDIA-specific maximum performance
- production CUDA optimization
- Tensor Core paths matter
- static-ish deployment configuration
- kernel tuning is worth the complexity

Best for:

```text
Supported server GPU/model/runtime combinations
Check exact release support; do not infer Jetson Orin support
latency-sensitive production inference
```

The runtime choice **changes the whole system**.

**Benchmark before committing.**

---

## 12. Jetson-oriented deployment

Good embedded default:

```text
generator:
  Qwen3.5-4B GGUF Q4_K_M

embedding:
  Granite 97M Multilingual R2
  Transformers or a verified ONNX execution provider for the exact hardware

vector DB:
  Qdrant

context:
  4K-8K

retrieval:
  top_k = 8
  rerank = 3

runtime:
  llama.cpp or carefully tested vLLM path
```

Compute the component estimate in section 3, then record peak resident/device memory from the actual selected runtime. No measured 5–7 GB memory claim is supplied by this lecture.

But on Jetson, also consider unified memory pressure:

- model weights
- KV cache
- embedding model
- vector index
- OS and desktop services
- Python runtime
- Qdrant memory
- buffers and temporary tensors

For Jetson, do **not run every component on GPU**.

Often:

```text
GPU:
  generator

CPU / optimized runtime:
  embedding
  vector search
  reranker if latency allows
```

---

## 13. Server-oriented deployment

Good small GPU server stack:

```text
generator:
  Qwen3.5-4B AWQ or GPTQ

embedding:
  Granite 311M if quality matters and memory allows
  Granite 97M if latency/cost matters

runtime:
  vLLM

vector DB:
  Qdrant

optimization:
  FlashInfer backend where applicable
  continuous batching
  prefix caching
  KV-cache optimization
```

Good when:

- many users
- concurrent local agents
- OpenAI-compatible endpoint desired
- larger context windows
- multiple models served behind routing

The server stack should measure:

- throughput
- p95 latency
- TTFT
- ITL
- GPU memory
- retrieval latency
- reranker latency
- prompt token count
- answer correctness

---

## 14. Advanced optimization

### KV-cache quantization

KV cache can **dominate long-context decode memory**.

Options:

- INT8 KV
- FP8 KV on supported hardware/backend
- paged KV cache

Use when:

- context length grows
- concurrency matters
- decode is memory-bound

See [MLSys Deep Dives Lecture 02](../../../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20G%20-%20ML%20Systems%20Engineering/MLSys%20Deep%20Dives/Lecture-02.md) for FP8 KV tradeoffs. Paging changes allocation, not bytes per scalar by itself; recurrent state needs its own dtype/quality analysis.

### Prefix caching

Agents often reuse:

- system prompt
- tool instructions
- safety policy
- RAG answer format

Cache stable prefixes when runtime supports it.

This avoids **recomputing the same prompt prefix** repeatedly.

### Speculative decoding

Use:

```text
small draft model
  -> larger verifier model
```

For a 4B verifier, a 0.5B-1B draft can improve throughput if acceptance rate is high.

Speculative decoding is **not free**.

Measure:

- acceptance rate
- extra memory
- added complexity
- latency distribution

---

## 15. Evaluation plan

Efficient RAG needs **both retrieval and generation evals**.

Measure retrieval:

- recall@k
- MRR
- nDCG
- exact source hit rate
- multilingual retrieval accuracy
- code symbol retrieval accuracy

Measure answer quality:

- groundedness
- citation correctness
- abstention when context is insufficient
- multilingual answer quality
- code correctness
- hallucination rate

Measure systems performance:

- query embedding latency
- vector search latency
- rerank latency
- prompt assembly time
- TTFT
- ITL
- total response latency
- VRAM
- RAM
- watts if on Jetson

The decision rule:

```text
optimize retrieval before increasing model size
```

---

## 16. Biggest mistakes

Avoid:

- giant chunks
- too many retrieved chunks
- no reranking
- huge prompts
- FP16 everywhere
- no caching
- no metadata filtering
- no citation checks
- no abstention behavior
- evaluating only final answers
- ignoring retrieval metrics
- running embedding, vector DB, reranker, and generator all on the GPU without measuring pressure

The **best engineering insight**:

```text
efficient RAG is mostly memory bandwidth and context-quality engineering,
not raw parameter count
```

The winning systems optimize:

- retrieval precision
- KV cache
- prompt size
- chunk quality
- batching
- token efficiency
- cache reuse
- metadata filtering

---

## Mini-lab: design a Jetson RAG stack

Design a local RAG system for technical documentation on Jetson Orin 16GB.

Fill this out:

```text
Generator:
Quantization:
Context length:
Embedding model:
Embedding runtime:
Vector DB:
Chunk size:
Overlap:
Top-K:
Rerank strategy:
Prompt budget:
Backend:
Calculated components and exclusions:
Measured peak unified RAM/device allocation:
Latency target:
Evaluation set:
Failure threshold:
```

Then write a decision:

```text
Use Qwen3.5-4B INT4 because:
Use Granite 97M because:
Use Qdrant because:
Need reranking because:
Do not increase context beyond:
First optimization to try:
First metric to monitor:
```

---

## Key takeaways

- Efficient local RAG is retrieval-quality engineering plus memory discipline.
- Qwen3.5-4B INT4 is a plausible small generator target, but exact variant/backend behavior must be validated.
- Granite 97M Multilingual R2 is a strong edge embedding model because it is compact, multilingual, and retrieval-oriented.
- Qdrant is usually a good edge-first vector database; pgvector is better when PostgreSQL integration dominates.
- Chunking and reranking often improve quality more than increasing generator size.
- Small models need compact, high-signal retrieved context and strict grounded prompts.
- llama.cpp is strong for embedded GGUF deployment; vLLM is strong for server batching; TensorRT-LLM is for deeper NVIDIA optimization.
- KV-cache optimization, prefix caching, and speculative decoding can improve local agent throughput.
- Measure retrieval, answer quality, latency, VRAM, RAM, and power before declaring the stack "efficient."

---

## References

- Qwen/Qwen3.5-4B model card: [https://huggingface.co/Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)
- Granite 97M Multilingual R2 model card: [https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2](https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2)
- IBM Granite Embedding docs: [https://www.ibm.com/us-en/granite/docs/models/embedding](https://www.ibm.com/us-en/granite/docs/models/embedding)
- Qdrant indexing documentation: [https://qdrant.tech/documentation/manage-data/indexing/](https://qdrant.tech/documentation/manage-data/indexing/)
- llama.cpp: [https://github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
- vLLM documentation: [https://docs.vllm.ai](https://docs.vllm.ai)
- TensorRT-LLM: [https://github.com/NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- FP8 KV-Cache in vLLM → [MLSys Deep Dives · Lecture 02](../../../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20G%20-%20ML%20Systems%20Engineering/MLSys%20Deep%20Dives/Lecture-02.md)
- MLSys 2026 Kernel Contest → [MLSys Deep Dives · Lecture 06](../../../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20G%20-%20ML%20Systems%20Engineering/MLSys%20Deep%20Dives/Lecture-06.md)

---

*Next: [Lecture 15](Lecture-15.md)*
