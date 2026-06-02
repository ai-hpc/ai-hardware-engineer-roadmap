# Lecture 06 - LLM From Scratch: Model Mechanics for Agent and GPU Engineers

**Course:** [AI Agent Development 2026](../Guide.md) | **Previous:** [Lecture 05](Lecture-05.md) | **Next:** [Lecture 07](Lecture-07.md)

---

**Agent engineers** usually work above the model:

```text
prompt
  -> tool calls
  -> memory
  -> Gateway
  -> runtime
  -> UI
```

**GPU and systems engineers** need to understand what happens below the API:

```text
tokens
  -> embeddings
  -> attention
  -> MLP
  -> logits
  -> sampling
```

The `angelos-p/llm-from-scratch` repository is useful because it strips the problem down to a **workshop-sized GPT**. The project walks through writing a tokenizer, transformer model, training loop, and generation code, then trains a small Shakespeare-style model on a laptop-class machine.

The key lesson for this roadmap:

```text
If you understand the model loop, agent-runtime bottlenecks stop looking mysterious.
```

---

## Learning objectives

By the end of this lecture, you should be able to:

1. Explain what a small GPT training pipeline contains.
2. Understand why tokenization changes the shape of the whole workload.
3. Map transformer blocks to GPU kernels and memory movement.
4. Distinguish training-time cost from inference-time cost.
5. Explain prefill, decode, KV cache, logits, and sampling in practical terms.
6. Connect model internals to agent system behavior: latency, context growth, streaming, and batching.
7. Use an from-scratch LLM workshop as a bridge from agent engineering to GPU/kernel engineering.

---

## 1. Why this belongs in an agent course

Most **agent failures** are not caused by attention math.

They are caused by **harness, tool, memory, policy, and product issues**.

Still, model mechanics matter because agents create **unusual inference workloads**:

- long context windows
- many short turns
- tool-call interruptions
- streaming tokens
- retries and repair loops
- multiple subagents
- background cron runs
- local/edge deployment

If you do not understand the model under the API, you will **misdiagnose performance**.

Examples:

| Symptom | Model-level explanation |
|---|---|
| first token is slow | prefill over the full prompt/context is expensive |
| later tokens stream steadily | decode reuses KV cache and generates one token at a time |
| long sessions get slower | attention and KV memory grow with context |
| batch serving helps throughput | multiple requests share GPU work more efficiently |
| tool-heavy agents feel bursty | execution alternates between CPU/IO tools and GPU inference |

Agent systems are **runtime systems**, but their runtime behavior is shaped by **transformer inference**.

---

## 2. What the reference repo builds

The reference project is a hands-on workshop titled **Train Your Own LLM From Scratch**.

It targets a **small GPT-style model**, not a production-scale frontier model.

The project has learners write:

- character-level tokenizer
- transformer model architecture
- training loop
- text generation and sampling
- experiments on real data

The repository describes three workshop model sizes:

| Config | Approx params | Layers | Heads | Embedding dim | Example train time |
|---|---:|---:|---:|---:|---|
| Tiny | ~0.5M | 2 | 2 | 128 | minutes |
| Small | ~4M | 4 | 4 | 256 | tens of minutes |
| Medium | ~10M | 6 | 6 | 384 | under an hour on an M3 Pro-class machine |

This scale is intentionally small.

That is the point.

You can see **every component** without distributed training, tokenizer complexity, or cluster infrastructure hiding the basics.

---

## 3. Pipeline overview

A minimal GPT pipeline:

```text
raw text
  -> tokenizer
  -> token ids
  -> token embedding + position embedding
  -> repeated transformer blocks
  -> final layer norm
  -> linear projection to logits
  -> loss during training
  -> sampling during inference
```

Training path:

```text
token batch
  -> forward pass
  -> logits
  -> cross-entropy loss
  -> backward pass
  -> optimizer step
```

Inference path:

```text
prompt tokens
  -> forward pass
  -> next-token logits
  -> sample/select token
  -> append token
  -> repeat
```

The same model is used in both paths.

The workload is different.

Training does forward and backward over batches.

Inference usually does prefill once and decode repeatedly.

---

## 4. Tokenization is not a detail

The workshop uses **character-level tokenization** for Shakespeare.

Why?

Because the dataset is small.

A GPT-2-style **BPE vocabulary** has roughly 50k tokens. On a tiny dataset, many token patterns are **too rare** for a small model to learn useful structure.

Character-level tokenization gives a tiny vocabulary:

```text
vocab_size ≈ tens of characters
```

Tradeoff:

| Tokenizer | Benefit | Cost |
|---|---|---|
| Character-level | simple, works on small data, easy to inspect | longer sequences |
| BPE/subword | shorter sequences, production-like | needs larger data and more machinery |

Hardware implication:

```text
tokenizer choice changes sequence length,
sequence length changes attention cost,
attention cost changes memory and latency.
```

For agent systems, tokenization affects:

- prompt size
- context-window usage
- retrieval chunk size
- cost accounting
- KV-cache memory
- latency

---

## 5. Transformer block anatomy

A basic GPT block:

```text
x
  -> LayerNorm
  -> self-attention
  -> residual add
  -> LayerNorm
  -> MLP / feed-forward
  -> residual add
```

Key components:

| Component | Job |
|---|---|
| token embedding | maps token IDs to vectors |
| position embedding | tells model where tokens are in sequence |
| Q/K/V projections | create query, key, value vectors for attention |
| attention scores | decide which earlier tokens matter |
| softmax | turns scores into weights |
| attention output | mixes value vectors according to weights |
| MLP | per-token nonlinear transformation |
| residuals | preserve information and stabilize optimization |
| layer norm | stabilizes activations |

GPU view:

```text
linear layers = matrix multiplies
attention = matmul + softmax + matmul
MLP = large matrix multiplies + activation
```

This is where CUDA/TensorRT/kernel engineers enter.

---

## 6. Training loop mechanics

Minimal training loop:

```text
for each step:
  sample batch
  forward model
  compute cross-entropy loss
  zero gradients
  backward loss
  clip gradients if needed
  optimizer step
  update learning rate schedule
  periodically evaluate/generate sample text
```

Important pieces:

| Piece | Why it matters |
|---|---|
| batch size | affects throughput and memory |
| block size | sequence length per sample |
| loss | tells model how wrong next-token prediction was |
| AdamW | common optimizer for transformer training |
| gradient clipping | prevents unstable updates |
| learning-rate schedule | avoids bad convergence |

Training is not just inference repeated.

**Backward pass and optimizer state** dominate memory.

For a small workshop model, that is manageable.

For production-scale models, it becomes a **distributed systems problem**.

---

## 7. Inference and sampling

Generation loop:

```text
prompt tokens
  -> model
  -> logits for next token
  -> adjust logits with temperature/top-k
  -> sample next token
  -> append token
  -> repeat
```

Important concepts:

| Concept | Meaning |
|---|---|
| logits | raw scores for each possible next token |
| temperature | controls randomness |
| top-k | restricts sampling to the k strongest candidates |
| autoregressive decoding | generated token becomes input for the next step |

Agent implication:

Every assistant response is a **decode loop**.

**Streaming** is just exposing that loop token-by-token or chunk-by-chunk.

Tool use interrupts the loop:

```text
model emits tool call
  -> runtime executes tool
  -> tool result enters context
  -> model continues
```

That is why agent latency is partly **model latency** and partly **harness/tool latency**.

---

## 8. Prefill and decode

Inference has two important phases:

### Prefill

The model processes the existing prompt/context:

```text
system prompt + history + retrieved context + user message
```

This is usually **compute-heavy** and grows with context length.

### Decode

The model generates one new token at a time while **reusing KV cache**.

This is often **memory-bandwidth-sensitive**.

Agent runtime connection:

| Agent behavior | Model-level effect |
|---|---|
| huge system prompt | larger prefill |
| long session history | larger prefill and KV cache |
| many retrieved docs | larger prefill |
| verbose tool outputs | context bloat |
| concise context compaction | lower prefill cost |
| streaming response | exposes decode phase |

This directly connects to previous lectures on context hygiene, TokenJuice, system prompts, and agent skills.

---

## 9. GPU/kernel-level view

A small from-scratch model helps you map **Python code to GPU work**.

Common hot paths:

| Model operation | Kernel-level concern |
|---|---|
| embedding lookup | memory access pattern |
| linear projection | GEMM throughput |
| attention score matmul | sequence-length scaling |
| softmax | numerical stability and memory bandwidth |
| attention value matmul | GEMM plus data layout |
| MLP up/down projection | dense matrix multiply |
| GELU/ReLU | elementwise kernel fusion |
| layer norm | reduction and memory bandwidth |
| logits projection | vocab-size-dependent GEMM |

This is why transformer inference optimization focuses on:

- fused kernels
- FlashAttention-style attention kernels
- KV-cache layout
- quantization
- batching
- memory bandwidth
- tensor parallelism
- graph capture

The workshop code will not implement all of those.

It gives you the mental map needed to understand them.

---

## 10. Why small models are still useful

Do not dismiss a 10M parameter GPT as a toy.

It is a **microscope**.

Small models let you:

- inspect every tensor shape
- see loss curves quickly
- test tokenizer changes
- understand sampling behavior
- profile a full training loop locally
- experiment without cluster cost

What transfers:

- architecture concepts
- training loop structure
- inference loop structure
- tensor shape reasoning
- performance intuition

What does not transfer directly:

- distributed training complexity
- production tokenizer/data pipelines
- large-scale optimizer state management
- serving at high concurrency
- frontier-model behavior

Use small models to understand **mechanisms**.

Use large systems to understand **scaling**.

---

## 11. Connection to agent skills and SDLC

From Lecture 21:

```text
skills require evidence
```

From Lecture 29:

```text
tests and intent are durable assets
```

For model work, evidence changes shape:

| Work item | Evidence |
|---|---|
| tokenizer change | vocab size, sample encoding/decoding, sequence length distribution |
| model change | parameter count, tensor shape checks, loss curve |
| training loop change | stable loss, gradient norms, eval loss |
| generation change | sample outputs, temperature/top-k comparison |
| performance change | tokens/sec, memory use, profiler trace |

A model-focused skill should not accept **"it trains"** as enough.

It should ask:

```text
What changed?
What metric moved?
What got slower?
What got less stable?
What evidence proves the behavior?
```

---

## 12. Mini-lab: trace one token through the model

Use the reference repo or your own minimal GPT code.

Trace:

```text
input character
  -> token id
  -> embedding vector
  -> attention block
  -> logits
  -> sampled next token
```

Record:

- token ID
- tensor shapes at each stage
- parameter count
- sequence length
- one generated sample
- training/eval loss after a short run

Then answer:

```text
Which operation is most expensive?
Which tensor grows with context length?
Where would KV cache matter?
Where would quantization help?
```

---

## 13. Design exercise: from scratch to serving

Take the workshop model and imagine serving it behind an agent runtime.

Design:

```text
HTTP/WebSocket API
request format
streaming token output
batching strategy
KV-cache ownership
context limit
tool-call interruption model
metrics
failure modes
```

Then compare:

```text
training script
  vs
serving runtime
  vs
agent harness
```

This separation is the **central architecture lesson**.

The **training script** creates weights.

The **serving runtime** turns weights into tokens.

The **harness** turns tokens and tools into work.

---

## Key takeaways

- Agent engineers benefit from understanding the model loop underneath the API.
- A from-scratch GPT workshop teaches tokenizer, architecture, training, and generation without hiding the basics.
- Tokenization changes sequence length, which changes attention cost and context behavior.
- Training and inference stress hardware differently.
- Prefill and decode explain much of agent latency.
- GPU optimization maps directly to transformer operations: GEMM, softmax, layer norm, KV cache, and memory layout.
- Small models are useful because they make mechanisms visible.
- The agent stack is layered: model mechanics, serving runtime, harness, tools, and product interface are different concerns.

---

## References

- Train Your Own LLM From Scratch: [https://github.com/angelos-p/llm-from-scratch](https://github.com/angelos-p/llm-from-scratch)
- nanoGPT: [https://github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- Attention Is All You Need: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
- GPT-2 paper: [https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- TinyStories: [https://arxiv.org/abs/2305.07759](https://arxiv.org/abs/2305.07759)
- Lecture 05 - LLM Fundamentals: [Lecture-05.md](Lecture-05.md)
- Lecture 28 - Runtime Strategy: [Lecture-28.md](Lecture-28.md)

---

*Next: [Lecture 07](Lecture-07.md)*
