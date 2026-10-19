# Lecture 08 — Tool Calls Are a Protocol

**Course:** [AI Agent Development 2026](../Guide.md) | **Previous:** [Lecture 07](Lecture-07.md) | **Next:** [Lecture 09](Lecture-09.md)

**Outcome:** implement a bounded client-tool loop, preserve call identifiers and distinguish a model request from an authorized action. The [provider contract lab](Provider-Contract-Lab.md) executes both Anthropic and OpenAI SDKs against local fixtures with no paid requests.

## 1. The model requests; your application executes

For a client-defined function, the model emits a name, arguments and call identifier. Your application validates and authorizes the request, runs the handler, and sends a correlated result. Hosted/server tools have a different execution owner; do not dispatch every returned content block as a local function.

| Integration | Tool request | Tool result |
|---|---|---|
| Claude Messages | `tool_use` content block | `tool_result` block in a **user** message, with matching `tool_use_id` |
| OpenAI Responses | `function_call` output item | `function_call_output` input item with matching `call_id` |
| OpenAI Chat Completions | Assistant `tool_calls` | `role: tool` message with `tool_call_id` |

These shapes are not interchangeable. Preserve all required assistant output items, including reasoning/continuation items where the model requires them. [Claude tool-call handling](https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls), [OpenAI function calling](https://developers.openai.com/api/docs/guides/function-calling).

## 2. Start with a small, truthful tool

The lab exposes `add_integers(a, b)` for bounded integer arithmetic. It really computes the sum and has no filesystem or network access. It does not pretend to run a GPU profiler.

```json
{
  "type": "object",
  "properties": {
    "a": {"type": "integer"},
    "b": {"type": "integer"}
  },
  "required": ["a", "b"],
  "additionalProperties": false
}
```

JSON Schema describes accepted inputs, while the executor enforces strict types, numeric bounds and its allowlist. A `default` annotation does not guarantee that a validator inserts a missing value. Schema validation is not authorization. Strict generation options also have provider/model-specific schema requirements. [Claude strict tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/strict-tool-use).

If you later add profiling, run an actual compiled executable under the correct profiler. Nsight Systems collects timelines; Nsight Compute collects detailed kernel metrics. A `.cu` path alone is not a runnable profiling command. Retain raw reports and distinguish a measurement from a fixture or estimate.

## 3. The bounded loop

Read the executable [provider_loops.py](labs/provider_loops.py) alongside the lab. It follows this sequence:

1. Call the selected SDK with a defined turn limit and explicit model identifier.
2. Inspect completion/stop status before acting. A truncated tool request is not authorization.
3. Preserve the assistant turn/output items.
4. Extract every client-function call and check the batch against a total call budget.
5. Validate arguments and dispatch only an allowed handler.
6. Return one result per identifier, including typed errors, using that provider's message format.
7. Continue until a completed final response or a bounded failure.

`end_turn` can contain multiple text blocks; join the appropriate blocks rather than assuming `content[0].text`. Other stop reasons, such as refusal, token exhaustion or a server-tool pause, need explicit branches. This minimal client-function lab rejects unsupported continuations instead of reporting a false successful completion. Its SDK transport tests do not evaluate a real model's ability to choose a tool.

## 4. Multiple calls and parallelism

A model may request several independent calls in one turn. With Claude, return the corresponding results together immediately after the assistant tool-use message; result blocks precede any additional text. Browser/computer toolsets have additional correlation fields; the minimal lab supports only its named client function. [Claude tool-call handling](https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls).

Multiple calls in a response do not automatically execute concurrently. Your scheduler decides. Parallelize independent reads when resource limits permit; serialize dependent operations and shared-state mutations. If two independent tools take 100 ms and 300 ms, ideal concurrent execution approaches 300 ms plus overhead, versus roughly 400 ms sequentially. There is no universal 3–5× improvement.

## 5. Errors, retries and side effects

An error result tells the model what failed; it does not guarantee recovery. Use a bounded retry policy for transient failures and distinguish invalid arguments, denied permission, missing resources and infrastructure failure. Do not return credentials or full exception internals in a tool result.

A network timeout after an external write is **ambiguous**: the server may have committed it. Persist an idempotency key and reconcile status before retrying. A call identifier correlates a conversation; it is not automatically a durable business idempotency key. The lab's in-memory duplicate-ID detection demonstrates one protocol failure, not exactly-once execution across process crashes.

## 6. Authorization belongs in the executor

Authorize the concrete operation for the authenticated user, resource and current policy. Any human review must approve the exact proposed arguments and become invalid if they change. Existing user authorization can cover routine reversible operations; a prompt rule demanding confirmation for everything is not a useful substitute for a permission model.

Do not expose `eval`, `exec`, unrestricted shell commands or arbitrary URLs as if removing Python builtins made a sandbox. A real execution environment needs isolation, bounded resources, credential/egress controls and a defined output contract. This lab intentionally uses a fixed arithmetic handler.

## 7. Check yourself

- Two tools were requested, but only one result was returned. **The protocol is incomplete; preserve both identifiers and results.**
- The model requests an unknown tool. **Return a bounded error; do not dynamically import or execute the name.**
- A tool's JSON contains duplicate `a` keys. **Reject ambiguity before execution.**
- A model keeps requesting tools forever. **The turn/call limit stops the run and records incomplete status.**
- A trace shows syntactically correct tool calls. **That proves neither authorization nor task success; inspect effects and the final task-specific criterion.**

Next: [interface selection and the scoped Reflex benchmark](Lecture-09.md).
