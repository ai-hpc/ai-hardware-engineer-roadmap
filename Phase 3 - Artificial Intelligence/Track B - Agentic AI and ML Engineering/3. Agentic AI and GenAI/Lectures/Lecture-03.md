# Lecture 03 — Tool Use & Function Calling

**Track B · Agentic AI & GenAI** | [← Lecture 02](Lecture-02.md) | [Next →](Lecture-04.md)

---

## Learning Objectives

- Define tools with precise schemas that minimize hallucination
- Implement the tool-use loop correctly (the core of every agent)
- Handle tool errors gracefully
- Use parallel tool calls to reduce latency
- Understand safety boundaries for dangerous tools

---

## 1. What Is Tool Use?

**Tool use** lets the LLM call external functions — search the web, run code, query a database, control a browser. The model doesn't execute code; it outputs a **structured JSON call** that your application executes.

```
User message
     ↓
  [LLM] → stop_reason: "tool_use" → tool call JSON
     ↓
Your code executes the tool
     ↓
Tool result sent back to LLM as "tool" role message
     ↓
  [LLM] → stop_reason: "end_turn" → final answer
```

---

## 2. Defining Tools

**Tool schemas** are the most important thing to get right. Vague descriptions cause wrong calls; wrong input types cause parse errors.

```python
import anthropic

tools = [
    {
        "name": "get_gpu_specs",
        "description": (
            "Retrieve detailed specifications for an NVIDIA or AMD GPU by model name. "
            "Returns memory, bandwidth, compute, and TDP. "
            "Use this when the user asks about specific GPU hardware capabilities."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "model": {
                    "type": "string",
                    "description": "GPU model name, e.g. 'H100 SXM5', 'A100 PCIe 80GB', 'RX 7900 XTX'"
                },
                "metric": {
                    "type": "string",
                    "enum": ["memory", "bandwidth", "compute", "tdp", "all"],
                    "description": "Which specification to retrieve. Use 'all' if unsure."
                }
            },
            "required": ["model"]
        }
    },
    {
        "name": "run_cuda_profiler",
        "description": (
            "Run Nsight Systems profiler on a CUDA kernel file and return performance metrics. "
            "Use this to identify bottlenecks: memory bound vs compute bound."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "kernel_path": {
                    "type": "string",
                    "description": "Absolute path to the .cu file"
                },
                "num_iterations": {
                    "type": "integer",
                    "description": "Number of profiling iterations (default: 100)",
                    "default": 100
                }
            },
            "required": ["kernel_path"]
        }
    }
]
```

**Schema writing rules:**

| Rule | Why |
|------|-----|
| Description explains *when* to call, not just *what* it does | Model decides whether to call based on this |
| Use `enum` for constrained string values | Eliminates typos and invalid inputs |
| Mark truly optional fields with `default`, don't put in `required` | Prevents unnecessary calls |
| Keep parameter names short and clear | Model generates parameter names verbatim |

---

## 3. The Tool-Use Loop

```python
import anthropic
import json
from typing import Any

client = anthropic.Anthropic()

# Mock tool implementations
def get_gpu_specs(model: str, metric: str = "all") -> dict:
    specs_db = {
        "H100 SXM5": {"memory": "80GB HBM3", "bandwidth": "3.35 TB/s",
                       "compute": "989 TFLOPS FP16", "tdp": "700W"},
        "A100 PCIe 80GB": {"memory": "80GB HBM2e", "bandwidth": "1.935 TB/s",
                            "compute": "312 TFLOPS FP16", "tdp": "300W"},
    }
    data = specs_db.get(model, {"error": f"GPU '{model}' not found in database"})
    if metric == "all" or "error" in data:
        return data
    return {metric: data.get(metric, "unknown")}

def run_cuda_profiler(kernel_path: str, num_iterations: int = 100) -> dict:
    # In production, actually run nsys/ncu here
    return {
        "kernel": kernel_path,
        "avg_duration_ms": 2.34,
        "memory_throughput_pct": 78.5,
        "compute_throughput_pct": 31.2,
        "bottleneck": "memory_bound",
        "recommendation": "Improve memory access coalescing"
    }

TOOL_MAP = {
    "get_gpu_specs": get_gpu_specs,
    "run_cuda_profiler": run_cuda_profiler,
}

def run_agent(user_message: str) -> str:
    """Core agent loop with tool use."""
    messages = [{"role": "user", "content": user_message}]

    while True:
        response = client.messages.create(
            model="your-agent-model-id",
            max_tokens=2048,
            tools=tools,
            messages=messages
        )

        # Append assistant response to message history
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            # Extract final text response
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text

        elif response.stop_reason == "tool_use":
            # Process all tool calls in this response
            tool_results = []

            for block in response.content:
                if block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input

                    # Execute the tool
                    if tool_name in TOOL_MAP:
                        result = TOOL_MAP[tool_name](**tool_input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result)
                        })
                    else:
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": f"Error: tool '{tool_name}' not found",
                            "is_error": True
                        })

            # Send tool results back to model
            messages.append({"role": "user", "content": tool_results})

        else:
            break  # Unexpected stop reason

    return "Agent completed without response."


# Test it
answer = run_agent(
    "Compare the H100 SXM5 and A100 PCIe 80GB memory bandwidth. "
    "Which is better for memory-bound kernels?"
)
print(answer)
```

---

## 4. Parallel Tool Calls

When multiple independent tools are needed, Claude can call them **simultaneously** — reducing **round-trips**.

```python
# Claude may return multiple tool_use blocks in one response
# Your loop must handle ALL of them before sending results back

for block in response.content:
    if block.type == "tool_use":
        # This may execute multiple times per response
        result = TOOL_MAP[block.name](**block.input)
        tool_results.append({
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": json.dumps(result)
        })

# Send ALL results in one message — critical!
messages.append({"role": "user", "content": tool_results})
```

**With Python `asyncio` for true parallel execution:**

```python
import asyncio
import anthropic

async def execute_tool_async(tool_name: str, tool_input: dict, tool_id: str) -> dict:
    # Run tool in thread pool (for blocking I/O tools)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, lambda: TOOL_MAP[tool_name](**tool_input))
    return {
        "type": "tool_result",
        "tool_use_id": tool_id,
        "content": json.dumps(result)
    }

async def process_tool_calls(tool_blocks: list) -> list:
    tasks = [
        execute_tool_async(b.name, b.input, b.id)
        for b in tool_blocks if b.type == "tool_use"
    ]
    return await asyncio.gather(*tasks)
```

> **Pro tip:** For I/O-heavy tools (HTTP requests, database queries), parallel async execution can cut multi-tool latency by 3–5×.

---

## 5. Error Handling

Agents must **handle tool failures gracefully** — the LLM can recover if you give it good error messages.

```python
def safe_tool_call(tool_name: str, tool_input: dict, tool_id: str) -> dict:
    """Execute a tool with error handling."""
    try:
        result = TOOL_MAP[tool_name](**tool_input)
        return {
            "type": "tool_result",
            "tool_use_id": tool_id,
            "content": json.dumps(result)
        }
    except KeyError as e:
        return {
            "type": "tool_result",
            "tool_use_id": tool_id,
            "content": f"Missing required parameter: {e}",
            "is_error": True
        }
    except Exception as e:
        return {
            "type": "tool_result",
            "tool_use_id": tool_id,
            "content": f"Tool execution failed: {type(e).__name__}: {e}",
            "is_error": True
        }
```

**When `is_error: True` is set**, Claude will typically:
1. Acknowledge the error
2. Try a different approach (different parameters, different tool)
3. Ask the user for clarification if stuck

---

## 6. Tool Safety Patterns

Some tools are **dangerous** (delete files, send emails, execute shell commands). Add **confirmation gates**.

```python
DANGEROUS_TOOLS = {"delete_file", "send_email", "execute_shell", "git_push"}

def run_agent_with_confirmation(user_message: str) -> str:
    messages = [{"role": "user", "content": user_message}]

    while True:
        response = client.messages.create(
            model="your-agent-model-id",
            max_tokens=2048,
            tools=all_tools,
            messages=messages
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            return next(b.text for b in response.content if hasattr(b, "text"))

        elif response.stop_reason == "tool_use":
            tool_results = []

            for block in response.content:
                if block.type != "tool_use":
                    continue

                # Human-in-the-loop for dangerous operations
                if block.name in DANGEROUS_TOOLS:
                    print(f"\n⚠️  Agent wants to call: {block.name}")
                    print(f"   Input: {json.dumps(block.input, indent=2)}")
                    confirm = input("   Allow? (y/n): ").strip().lower()

                    if confirm != "y":
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": "User denied this action.",
                            "is_error": True
                        })
                        continue

                result = safe_tool_call(block.name, block.input, block.id)
                tool_results.append(result)

            messages.append({"role": "user", "content": tool_results})
```

---

## 7. Tool Use with `tool_choice`

Force or restrict which tools the model uses:

```python
# Force the model to use a specific tool
response = client.messages.create(
    model="your-agent-model-id",
    max_tokens=512,
    tools=tools,
    tool_choice={"type": "tool", "name": "get_gpu_specs"},  # must call this
    messages=[{"role": "user", "content": "Tell me about the H100."}]
)

# Allow any tool (default)
tool_choice={"type": "auto"}

# Prevent any tool use
tool_choice={"type": "none"}
```

---

## Key Takeaways

1. The tool loop: call API → check `stop_reason` → execute tools → append results → repeat
2. Send **all** tool results in one message — never split them
3. Use `is_error: True` for failures; the model will try to recover
4. Parallel tool calls are automatic — your code must handle multiple blocks per response
5. Gate dangerous operations with human-in-the-loop confirmation
6. `tool_choice` forces or prevents tool use — useful for structured extraction workflows

---

## Exercises

1. Build a weather + calculator dual-tool agent. Test it with a query that requires both tools in one response.
2. Add retry logic to `safe_tool_call` — retry up to 3 times with exponential backoff on network errors.
3. Implement a `max_tool_calls` safety limit that stops the agent after N tool calls to prevent infinite loops.

---

**Previous:** [Lecture 02](Lecture-02.md) | **Next:** [Lecture 04 — Agent Architecture Patterns](Lecture-04.md)
