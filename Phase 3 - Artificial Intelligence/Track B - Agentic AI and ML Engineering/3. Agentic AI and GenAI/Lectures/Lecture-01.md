# Lecture 01 — LLM Fundamentals for Agents

**Track B · Agentic AI & GenAI** | [← Index](README.md) | [Next →](Lecture-02.md)

---

## Learning Objectives

By the end of this lecture you will be able to:

- Explain how transformer inference works at a high level (prefill vs. decode)
- Calculate token counts and context window costs
- Choose the right model for a given agent task
- Understand why latency, throughput, and TTFT matter for agentic loops

---

## 1. How Transformers Generate Text

An LLM does one thing: given a sequence of tokens, **predict the next token**. An **agent** is just a loop that keeps calling this function.

```
Input tokens → [Transformer] → Logits → Sample → Output token
                                                        ↓
                                              Append to context
                                                        ↓
                                              Repeat until stop
```

**Two phases of inference:**

| Phase | What happens | Compute bound |
|-------|-------------|---------------|
| **Prefill** | Process all input tokens in parallel (matrix multiply) | Compute (FLOP-bound) |
| **Decode** | Generate one token at a time (autoregressive) | Memory bandwidth |

> **Hardware implication:** Prefill saturates GPU compute. Decode is bottlenecked by how fast you can stream weights from HBM. This is why inference accelerators (Groq, Etched) focus on memory bandwidth, not just FLOPS.

---

## 2. Tokens and Context Windows

**Tokens ≠ words**. Rule of thumb: **1 token ≈ 0.75 English words** (4 characters).

```python
import os
import anthropic

client = anthropic.Anthropic()

# Count tokens before sending
response = client.messages.count_tokens(
    model=os.environ.get("ANTHROPIC_MODEL", "your-model-id"),
    messages=[{"role": "user", "content": "Hello, how are you?"}]
)
print(response.input_tokens)  # → 10
```

**Context windows change quickly. Think in categories instead of memorizing one snapshot:**

| Category | Typical use | Engineering concern |
|----------|-------------|---------------------|
| Small/fast chat model | routing, classification, short summaries | low latency and low cost |
| Balanced agent model | tool use, JSON extraction, code review | reliable structure and good reasoning |
| Long-context model | repo analysis, large documents, multi-file tasks | context cost, retrieval quality, memory pressure |
| Local/open-weight model | edge inference, privacy, offline demos | VRAM, quantization, throughput |
| Embedding model | RAG indexing and retrieval | vector dimension, recall, index cost |

Always **verify current context limits** in the provider documentation before designing a production agent around a specific window size.

**Why context size matters for agents:**
- Multi-step reasoning accumulates tokens fast
- Tool call results land in context
- Long documents fed to RAG agent must fit

---

## 3. Inference Parameters

```python
import os

response = client.messages.create(
    model=os.environ.get("ANTHROPIC_MODEL", "your-model-id"),
    max_tokens=1024,
    temperature=0.0,    # 0 = deterministic (good for agents/tools)
                        # 1 = creative (good for writing)
    top_p=1.0,
    messages=[{"role": "user", "content": "What is 2+2?"}]
)
```

| Parameter | Effect | Agent recommendation |
|-----------|--------|---------------------|
| `temperature` | Randomness of sampling | 0.0–0.3 for tool use / reasoning |
| `top_p` | Nucleus sampling cutoff | Leave at 1.0 (let temperature do the work) |
| `max_tokens` | Hard output limit | Set generously — truncation breaks JSON |

> **Pro tip:** For tool-use agents, always use `temperature=0` or close to it. Randomness in function call generation causes JSON parse errors and unpredictable behavior.

---

## 4. The Anatomy of an API Call

```python
import os
import anthropic

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

response = client.messages.create(
    model=os.environ.get("ANTHROPIC_MODEL", "your-model-id"),
    max_tokens=2048,
    system="You are a helpful assistant.",       # system prompt
    messages=[
        {"role": "user",    "content": "Tell me about CUDA."},
        {"role": "assistant","content": "CUDA is..."},  # prior turn
        {"role": "user",    "content": "How does it compare to ROCm?"},
    ]
)

print(response.content[0].text)
print(f"Input tokens:  {response.usage.input_tokens}")
print(f"Output tokens: {response.usage.output_tokens}")
print(f"Stop reason:   {response.stop_reason}")  # end_turn | tool_use | max_tokens
```

**`stop_reason` values for agents:**

| Value | Meaning |
|-------|---------|
| `end_turn` | Model finished naturally |
| `tool_use` | Model wants to call a tool — your loop must handle this |
| `max_tokens` | Hit the limit — increase or handle gracefully |

---

## 5. Model Selection for Agent Tasks

Not every task needs the most powerful model. **Cost and latency** add up in **multi-step loops**.

```python
# Router pattern: use fast/cheap model for simple steps
def route_model(task_type: str) -> str:
    fast_model = "provider-fast-model"
    balanced_model = "provider-balanced-agent-model"
    reasoning_model = "provider-reasoning-model"

    routing = {
        "classification": fast_model,
        "summarization": fast_model,
        "tool_use": balanced_model,
        "complex_reasoning": reasoning_model,
        "coding": balanced_model,
    }
    return routing.get(task_type, balanced_model)
```

| Task | Recommended model class | Why |
|------|-------------------------|-----|
| Simple Q&A, routing | fast model | low latency and cost |
| Tool use, JSON extraction | balanced agent model | reliable structured output |
| Complex reasoning, long context | reasoning or long-context model | stronger planning and larger working set |
| Embeddings | embedding model | specialized vector representation |

---

## 6. Streaming for Responsive Agents

In agentic UIs, **streaming** dramatically improves **perceived responsiveness**.

```python
import os
import anthropic

client = anthropic.Anthropic()

with client.messages.stream(
    model=os.environ.get("ANTHROPIC_MODEL", "your-model-id"),
    max_tokens=1024,
    messages=[{"role": "user", "content": "Explain transformer attention."}]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)

# Access final message with usage stats
final = stream.get_final_message()
print(f"\nTokens used: {final.usage.input_tokens} in, {final.usage.output_tokens} out")
```

---

## 7. Cost Estimation

```python
# Rough cost calculator.
# Do not hardcode provider prices in production. Load this from a config file
# maintained from the provider pricing page.
PRICING = {
    "provider-fast-model": {"input": 0.15, "output": 0.60},       # example only, per 1M tokens
    "provider-balanced-agent-model": {"input": 3.00, "output": 15.00},
    "provider-reasoning-model": {"input": 15.00, "output": 75.00},
}

def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    p = PRICING[model]
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000

# A 10-step agent loop with a balanced agent model
steps = 10
per_step_input  = 2000   # context grows each step
per_step_output = 500
total = sum(
    estimate_cost("provider-balanced-agent-model", per_step_input * i, per_step_output)
    for i in range(1, steps + 1)
)
print(f"Estimated loop cost: ${total:.4f}")
```

> **Key insight:** In a 10-step agent loop, context grows linearly — step 10 has 10× the input tokens of step 1. This is why context management (summarization, pruning) is critical in production.

---

## Key Takeaways

1. LLM inference = prefill (compute-bound) + decode (memory-bandwidth-bound)
2. Use `temperature=0` for tool-use agents; reserve higher values for creative tasks
3. Check `stop_reason` — `tool_use` means your loop must call the tool and continue
4. Route tasks to cheaper models where possible; the cost compounds in multi-step loops
5. Context grows each step — plan for summarization or windowing in long-running agents

---

## Exercises

1. Write a script that counts tokens for a 10-page PDF before sending it to the API.
2. Build a simple cost logger that wraps `client.messages.create` and prints cumulative cost.
3. Implement a model router that uses a fast model for tasks under 200 input tokens and a balanced agent model otherwise.

---

**Next:** [Lecture 02 — Prompt Engineering & Structured Output](Lecture-02.md)
