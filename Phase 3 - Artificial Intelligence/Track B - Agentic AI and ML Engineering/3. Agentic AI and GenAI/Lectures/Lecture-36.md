# Lecture 36 - FP8 KV-Cache in vLLM: Long-Context Serving for Agent Workloads

**Course:** [Agentic AI & GenAI](../Guide.md) | **Previous:** [Lecture 35](Lecture-35.md) | **Next:** [Lecture 37](Lecture-37.md)

---

**Long-context agents** are memory systems.

They keep:

- system prompts
- session history
- retrieved documents
- tool outputs
- planner traces
- multimodal context
- generated reasoning

Underneath the API, the model server stores this context in a **KV cache**.

For standard full-attention decoder models, that cache can **dominate GPU memory** at long contexts, and every decode step must read a large fraction of it.

The core serving problem:

```text
longer context
  -> larger KV cache
  -> more GPU memory
  -> more memory traffic per generated token
  -> higher inter-token latency
  -> lower concurrency
```

vLLM's **FP8 KV-cache** work matters because it attacks that bottleneck directly.

---

## Learning objectives

By the end of this lecture, you should be able to:

1. Explain why long-context serving becomes KV-cache and memory-bandwidth bound.
2. Distinguish TTFT, ITL, prefill, decode, and throughput under load.
3. Understand what `--kv-cache-dtype fp8` changes in vLLM.
4. Explain why Hopper needed a two-level accumulation fix for long-context FP8 accuracy.
5. Understand why small sliding-window layers should often stay BF16.
6. Identify when FP8 KV-cache helps, when it hurts, and when calibration is required.
7. Design a benchmark plan for FP8 KV-cache on agent workloads.
8. Connect KV-cache quantization to OpenClaw-style long-running agents, cron jobs, RAG, and multimodal sessions.

---

## 1. Why KV cache dominates long-context serving

During autoregressive inference, the model generates one token at a time.

For each generated token, attention needs keys and values from the previous context.

That stored state is the KV cache:

```text
K = keys for previous tokens
V = values for previous tokens
```

At short context, **model compute** may dominate.

At long context, decode often becomes **memory-bound**:

```text
generated token
  -> read KV cache
  -> compute attention
  -> produce next token
```

If the cache is large, every token generation pays for moving a lot of data.

This is why long-context agents can become slow even when the model's **raw FLOPs** look sufficient.

---

## 2. Prefill vs decode

Long-context serving has two different phases.

### Prefill

The model processes the entire input prompt:

```text
system prompt + history + retrieved context + user message
```

This is **time-to-first-token** work.

Metric:

```text
TTFT = time to first token
```

For full attention, prefill cost can grow **roughly quadratically** with input length.

### Decode

The model emits new tokens one at a time.

Metric:

```text
ITL = inter-token latency
```

For long contexts, ITL tends to grow **roughly linearly** with input length because each new token attends over the cache.

Serving engineers must track both:

```text
TTFT: how long until the user sees the first token?
ITL: how fast do later tokens stream?
```

An optimization can **improve one and hurt the other**.

---

## 3. What FP8 KV-cache does

vLLM exposes:

```bash
vllm serve meta-llama/Llama-3.1-8B --kv-cache-dtype fp8
```

This stores KV cache in **FP8** and runs attention's QK and ScoreV matrix multiplications in FP8 on the supported paths described by vLLM.

The simplest mental model:

```text
BF16 KV cache
  -> larger cache
  -> more memory traffic

FP8 KV cache
  -> roughly half cache storage
  -> less memory traffic
  -> potentially lower decode latency and higher concurrency
```

But the real result depends on:

- GPU architecture
- attention backend
- head dimension
- context length
- prefill/decode mix
- sliding-window layers
- quantization scales
- model sensitivity

FP8 is a **strong default candidate** for long-context decode-heavy workloads.

It is **not a universal win**.

---

## 4. The accuracy problem vLLM found

The vLLM team found a serious **Hopper/FlashAttention-3 issue** under stress testing.

On a 128k needle-in-a-haystack task:

```text
BF16 baseline accuracy: 91%
FP8 before fix:         13%
FP8 after fix:          89%
```

The issue was not "FP8 is bad" in the abstract.

It was **accumulation precision** during long-context attention.

In long-context inference:

```text
Softmax(AttentionScore) * V
```

has a contraction dimension corresponding to context length.

At very large context lengths, imprecise intermediate accumulation caused severe numerical errors.

vLLM and FlashAttention added a **two-level accumulation** strategy to restore accuracy.

Tradeoff:

```text
accuracy restored
  -> more register pressure
  -> possible prefill slowdown, especially head_dim > 128
```

This is the core hardware lesson:

```text
low precision is a systems contract,
not just a dtype flag
```

**Kernel details matter.**

---

## 5. The performance metric: ITL slope

For concurrency 1, vLLM models ITL as:

```text
ITL = slope * input_len + intercept
```

Interpretation:

| Term | Meaning |
|---|---|
| slope | extra decode latency added by each cached token |
| intercept | fixed per-token overhead independent of input length |

FP8 is attractive when it **lowers the slope** enough to overcome any intercept overhead.

That produces a **break-even context length**:

```text
below break-even: BF16 may be faster
above break-even: FP8 decode is faster
```

This is a better way to reason than simply saying:

```text
FP8 is faster
```

You should ask:

```text
At what context length?
For which model?
On which backend?
With what head dimension?
```

---

## 6. Llama-class result

For Llama-3.1-8B on H100 with the improved FP8 path, vLLM reports:

```text
BF16 ITL slope: 4.37e-05 ms/token
FP8 ITL slope:  2.37e-05 ms/token
FP8 slope:      54% of BF16
break-even:     ~7k tokens
```

Under load:

```text
150 requests
concurrency 8
~20k input tokens
~2k output tokens
```

vLLM reports for Llama-3.1-8B:

```text
BF16 output throughput: 450.3 tok/s
FP8 output throughput:  517.5 tok/s
gain:                  14.9%
```

The serving takeaway:

```text
single-request ITL slope improvements can translate into real throughput gains,
but the end-to-end gain is smaller than the raw slope reduction
```

That is normal because real serving also includes **scheduling, prefill, batching, and non-attention work**.

---

## 7. Hybrid attention and sliding-window layers

Hybrid models may include both:

- global attention layers
- sliding-window attention layers

**Sliding-window layers** attend only over a bounded recent window.

Example:

```text
window size = 128
```

For these layers, KV-cache size **does not grow** with full context length.

Quantizing small bounded windows may **add overhead** without enough memory-traffic savings.

vLLM added:

```bash
--kv-cache-dtype-skip-layers sliding_window
```

Example:

```bash
vllm serve gpt-oss-20b \
  --kv-cache-dtype fp8 \
  --kv-cache-dtype-skip-layers sliding_window
```

For gpt-oss-20b on H100, vLLM reports:

```text
BF16 ITL slope:        8.94e-06 ms/token
full FP8 slope:        7.14e-06 ms/token  (80% of BF16)
FP8 skip-SW slope:     6.34e-06 ms/token  (71% of BF16)
break-even skip-SW:    ~7.7k tokens
```

Design rule:

```text
Quantize the layers where KV-cache memory traffic dominates.
Do not quantize small bounded windows just because a global dtype flag exists.
```

---

## 8. Head dimension 256 caveat

**Large head dimensions** change the tradeoff.

For a model with `head_dim = 256`, vLLM reports that FP8 **improves decode ITL** but can **worsen TTFT** because two-level accumulation increases register pressure.

Example from gemma-4-E2B on H100:

```text
BF16 ITL slope: 5.30e-05 ms/token
FP8 ITL slope:  3.60e-05 ms/token
FP8 slope:      68% of BF16

BF16 TTFT quadratic coefficient: 6.93e-07
FP8 TTFT quadratic coefficient:  1.12e-06
```

Interpretation:

```text
decode improves
prefill slows down
```

If your workload is **decode-heavy**, FP8 may still help.

If your workload is **prefill-heavy**, especially with very long prompts and short outputs, BF16 may be better.

Agent implication:

| Workload | Risk |
|---|---|
| long RAG prompt, short answer | TTFT dominates |
| long reasoning output | decode dominates |
| chat with repeated long history | both matter |

**Do not optimize blindly.**

Measure the **phase that dominates** your workload.

---

## 9. Hopper vs Blackwell

vLLM's post distinguishes H100/Hopper and B200/Blackwell paths.

On Hopper with FlashAttention-3:

- two-level accumulation was needed to address long-context FP8 accuracy
- optimized tile sizes improved prefill/decode behavior

On Blackwell B200 with FlashInfer:

- the accumulation issue is described as fixed on that path
- FP8 still reduces decode slope

Examples reported:

```text
Llama-3.1-8B on B200:
BF16 slope: 1.80e-05
FP8 slope:  9.72e-06
break-even: ~4k tokens

gpt-oss-20b on B200:
BF16 slope: 3.56e-06
FP8 slope:  2.06e-06
break-even: ~13k tokens
```

**Hardware generation and attention backend** are part of the configuration.

Do not assume H100 results **transfer exactly** to B200, or vice versa.

---

## 10. Accuracy results

The vLLM team tested:

- reasoning benchmarks such as AIME25, GPQA:Diamond, MATH500, LiveCodeBench-v6
- long-context MRCR evaluations up to 1M tokens
- decoder-only and MoE models
- BF16 and FP8 weight/activation settings
- Hopper and Blackwell paths

High-level findings:

- Qwen3-30B-A3B-Thinking-2507 reasoning changed by about 1-2 points with FP8 KV-cache plus FP8 attention.
- Qwen3.5-27B reasoning showed sub-point differences in the reported aggregate scores.
- Llama-3.3-70B-Instruct recovered about 97-98% of baseline AUC at 128k MRCR.
- Qwen3-30B-A3B-Instruct-2507 recovered roughly 94-98% AUC at 256k depending on model setting.
- Qwen3.5-27B matched aggregate AUC up to 1M in the reported setup.

The post intentionally uses simple per-tensor **uncalibrated scale** `1.0` as a reproducible lower bound.

That matters because:

```text
if uncalibrated works, deployment is simple
if uncalibrated shows systematic degradation, calibration is the next step
```

---

## 11. When to calibrate

Start simple:

```text
--kv-cache-dtype fp8
```

Then evaluate your workload.

Calibrate if you see:

- systematic downward accuracy shift
- model-specific degradation
- non-standard attention backend behavior
- task-specific sensitivity
- long-context retrieval failures

vLLM specifically notes **Kimi-K2.5 with FlashMLA** as an example where uncalibrated FP8 showed consistent negative shift, making calibration worth considering.

**Calibration is not free.**

It adds:

- dataset selection work
- evaluation work
- deployment complexity
- possible per-head/per-tensor scale management

Use it when the **accuracy evidence** justifies it.

---

## 12. When to avoid FP8 KV-cache

Stay with BF16, or at least be cautious, when:

| Condition | Reason |
|---|---|
| contexts are short, roughly below 7k tokens | FP8 overhead may not amortize |
| `head_dim = 256` and prefill matters | two-level accumulation can slow TTFT |
| uncalibrated accuracy drops below your threshold | calibration or BF16 may be needed |
| many small sliding-window layers | FP8 overhead may not pay off |
| backend/model path is not well validated | hidden accuracy/performance regressions possible |

Decision rule:

```text
FP8 KV-cache is a default candidate for long-context decode-heavy serving.
It is not a default truth.
```

---

## 13. OpenClaw and agent workload mapping

OpenClaw-style agents create several long-context serving patterns.

| Agent pattern | Serving shape |
|---|---|
| long chat session | growing KV cache and repeated decode |
| RAG over large docs | heavy prefill, often shorter decode |
| code agent with logs | long prompt and medium decode |
| reasoning agent | short/medium prefill, long decode |
| cron summarizer | batch-like prefill plus summary decode |
| multimodal perception handoff | large context compression plus downstream decode |

FP8 KV-cache is most attractive when:

```text
many concurrent sessions
long contexts
meaningful output lengths
decode is memory-bound
accuracy passes workload evals
```

It is less compelling when:

```text
short prompts
short outputs
prefill dominates
model has sensitive attention backend
hybrid small-window layers dominate
```

---

## 14. Benchmark plan for your agent server

Do not rely only on **public benchmark numbers**.

Run your **own matrix**:

```text
models:
  - primary chat model
  - reasoning model
  - code model
  - multimodal/context model if served through vLLM

configs:
  - BF16 KV cache
  - FP8 KV cache
  - FP8 skip sliding-window layers where relevant
  - calibrated FP8 if uncalibrated drops

workloads:
  - short chat
  - long chat
  - RAG long prefill / short decode
  - code agent logs / medium decode
  - reasoning long decode
  - concurrent sessions
```

Measure:

```text
TTFT
ITL
output tokens/sec
requests/sec
GPU memory
max concurrency before OOM
accuracy / task pass rate
long-context retrieval correctness
cost per task
```

The winner is **workload-dependent**.

---

## 15. Practical commands

Basic:

```bash
vllm serve meta-llama/Llama-3.1-8B \
  --kv-cache-dtype fp8
```

Hybrid attention model with small sliding-window layers:

```bash
vllm serve gpt-oss-20b \
  --kv-cache-dtype fp8 \
  --kv-cache-dtype-skip-layers sliding_window
```

Benchmarking shape:

```bash
vllm bench serve \
  --model <model> \
  --num-prompts 150 \
  --request-rate inf
```

Treat commands as **starting points**.

Pin **vLLM version, GPU backend, model revision, and benchmark dataset** before comparing results.

---

## 16. Hardware engineer view

FP8 KV-cache is a **hardware/software co-design** example.

The optimization involves:

- lower precision storage
- tensor core behavior
- attention kernel design
- register pressure
- tiling
- memory bandwidth
- quantization scales
- model architecture
- serving scheduler behavior

The key insight:

```text
Quantization is not just a model-compression trick.
It changes the memory traffic and kernel behavior of the serving system.
```

For GPU engineers, the interesting questions are:

```text
Where is the bottleneck: memory bandwidth, compute, registers, or scheduler?
Does the dtype reduce traffic enough to pay for conversion overhead?
Does the attention backend preserve accuracy at long context?
Does the workload benefit from higher concurrency or lower ITL?
```

---

## Mini-lab: FP8 KV-cache deployment decision

Pick one vLLM-served model.

Run three configurations:

```text
BF16
FP8
FP8 with skip sliding-window layers, if relevant
```

Use two workloads:

```text
long prefill / short output
medium prefill / long output
```

Record:

```text
TTFT
median ITL
output tok/s
GPU memory
max concurrency before OOM
task accuracy
```

Then write a deployment decision:

```text
Use FP8 KV-cache for:
Avoid FP8 KV-cache for:
Need calibration for:
Need further testing for:
```

---

## Key takeaways

- Long-context agent serving is often KV-cache and memory-bandwidth bound.
- FP8 KV-cache can roughly halve cache storage and reduce decode memory traffic.
- vLLM's FP8 path is now a strong starting point for many long-context decode-heavy deployments.
- Hopper required a two-level accumulation fix to recover long-context FP8 accuracy in FlashAttention-3.
- Sliding-window layers with small windows often should stay BF16 using `--kv-cache-dtype-skip-layers sliding_window`.
- `head_dim = 256` can make prefill slower under two-level accumulation even when decode improves.
- Use uncalibrated FP8 first for simplicity, but calibrate if workload accuracy shows systematic degradation.
- Always benchmark TTFT, ITL, throughput, memory, concurrency, and task accuracy on your real agent workload.

---

## References

- vLLM Blog, "The State of FP8 KV-Cache and Attention Quantization in vLLM": [https://vllm.ai/blog/fp8-kvcache](https://vllm.ai/blog/fp8-kvcache)
- vLLM documentation: [https://docs.vllm.ai](https://docs.vllm.ai)
- FlashAttention: [https://github.com/Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)
- FlashInfer: [https://github.com/flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer)
- LLM Compressor: [https://github.com/vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor)
- Lecture 32 - LLM From Scratch: [Lecture-32.md](Lecture-32.md)
- Lecture 34 - Nemotron 3 Nano Omni: [Lecture-34.md](Lecture-34.md)

---

*Next: [Lecture 37 - TraceLens](Lecture-37.md)*
