# Lecture 07 - Agent SDKs and Runtime APIs

**Track B · Agentic AI & GenAI** | [Previous: Lecture 06](Lecture-06.md) | [Next: Lecture 08](Lecture-08.md)

---

## Learning Objectives

By the end of this lecture you will be able to:

- Explain the difference between a raw model API, an agent SDK, a workflow runtime, MCP, and an agent gateway.
- Design a small provider-neutral runtime contract for model calls, tool calls, handoffs, streaming, and logs.
- Decide when to use a managed SDK, when to own the loop yourself, and when to move orchestration into a graph or gateway.
- Treat tools, MCP servers, and subagents as security boundaries instead of just convenience wrappers.
- Add the minimum runtime telemetry needed for debugging, evaluation, and audit.

---

## 1. Why This Lecture Replaces "One Vendor SDK"

Early agent tutorials usually taught one pattern:

1. Send a prompt to a model.
2. If the model asks for a tool, call the function.
3. Send the tool result back.
4. Repeat until the model stops.

That loop is still real, but **production agent systems** now have more structure. A modern **agent stack** usually has several layers:

| Layer | Simple meaning | Examples |
|-------|----------------|----------|
| Model API | The direct inference surface | responses/messages APIs, structured output, tool calls |
| Agent SDK | A managed loop around model calls | agents, tools, handoffs, sessions, guardrails, tracing |
| Workflow runtime | Durable control flow | graph execution, checkpoints, human review, retries |
| Tool protocol | Standardized external capabilities | MCP tools, resources, prompts |
| Gateway/control plane | Product-level routing and sessions | OpenClaw-style channels, sessions, agents, nodes |
| Runtime policy | Safety and governance at execution time | authorization, allowlists, audit logs, approval gates |

The important skill is not memorizing one package name. The important skill is knowing **which layer owns which responsibility**.

---

## 2. The 2026 Agent Runtime Map

Use this map when choosing architecture.

| Situation | Best starting point | Why |
|-----------|--------------------|-----|
| One short task, no tools | Raw provider API | Lowest complexity |
| One assistant with a few function tools | Agent SDK | Built-in loop, tool dispatch, streaming, traces |
| Multi-step workflow with retries and review | Workflow runtime | Durable execution and explicit state transitions |
| Many external tools or apps | MCP | Standard tool/resource/prompt integration boundary |
| Multi-channel assistant or local-first product | Gateway/control plane | Routing, sessions, auth, pairing, device/channel isolation |
| Regulated or high-risk actions | Runtime policy layer | Deterministic authorization outside the LLM |

Concrete examples:

- OpenAI Agents SDK documents agents, tools, handoffs, guardrails, sessions, tracing, and MCP integration.
- LangGraph is useful when the agent is a long-running, stateful workflow that needs checkpointing and human-in-the-loop controls.
- MCP is useful when you want tools and context servers to be reusable across IDEs, chat apps, local assistants, and agent runtimes.
- OpenClaw is a useful case study for gateway-based assistants: channels, sessions, routing, agent ownership, and local-first control.

---

## 3. The Runtime Contract You Should Teach Your Codebase

Before picking any SDK, define the shape of the work your system understands.

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    risk: Literal["read", "write", "external", "destructive"] = "read"
    requires_approval: bool = False


@dataclass
class ToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    call_id: str
    content: str
    is_error: bool = False


@dataclass
class AgentRequest:
    session_id: str
    user_id: str
    messages: list[dict[str, Any]]
    tools: list[ToolSpec] = field(default_factory=list)
    max_steps: int = 12
    budget_usd: float = 1.00


@dataclass
class RuntimeEvent:
    type: Literal["model_start", "model_delta", "tool_call", "tool_result", "handoff", "policy_block", "done"]
    session_id: str
    payload: dict[str, Any]


@dataclass
class AgentResponse:
    final_text: str
    events: list[RuntimeEvent]
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: list[ToolCall] = field(default_factory=list)
```

This contract matters because **provider APIs change faster** than your product architecture should. Keep **vendor-specific response formats** inside adapters. Keep your product semantics stable.

---

## 4. Own the Adapter Boundary

A clean **adapter** converts provider-specific responses into your **runtime contract**.

```python
import os
from typing import Protocol


class ModelAdapter(Protocol):
    def run_turn(
        self,
        messages: list[dict],
        tools: list[ToolSpec],
        model: str,
    ) -> tuple[str, list[ToolCall], dict]:
        """Return text, tool calls, and usage metadata."""
        ...


class AgentRuntime:
    def __init__(self, adapter: ModelAdapter, tool_registry: dict[str, callable]):
        self.adapter = adapter
        self.tool_registry = tool_registry

    def run(self, request: AgentRequest) -> AgentResponse:
        model = os.environ.get("AGENT_MODEL", "default-agent-model")
        messages = list(request.messages)
        events: list[RuntimeEvent] = []
        all_tool_calls: list[ToolCall] = []
        input_tokens = 0
        output_tokens = 0

        for _step in range(request.max_steps):
            text, tool_calls, usage = self.adapter.run_turn(messages, request.tools, model)
            input_tokens += int(usage.get("input_tokens", 0))
            output_tokens += int(usage.get("output_tokens", 0))

            if not tool_calls:
                events.append(RuntimeEvent("done", request.session_id, {"text": text}))
                return AgentResponse(text, events, input_tokens, output_tokens, all_tool_calls)

            all_tool_calls.extend(tool_calls)
            messages.append({"role": "assistant", "content": text, "tool_calls": tool_calls})

            for call in tool_calls:
                tool = next((t for t in request.tools if t.name == call.name), None)
                if tool is None:
                    result = ToolResult(call.call_id, f"Unknown tool: {call.name}", is_error=True)
                elif tool.requires_approval or tool.risk in {"write", "destructive"}:
                    events.append(RuntimeEvent("policy_block", request.session_id, {"tool": call.name}))
                    result = ToolResult(call.call_id, "Blocked: approval required", is_error=True)
                else:
                    handler = self.tool_registry[call.name]
                    result = ToolResult(call.call_id, str(handler(**call.arguments)))

                events.append(RuntimeEvent("tool_result", request.session_id, result.__dict__))
                messages.append({"role": "tool", "tool_call_id": call.call_id, "content": result.content})

        return AgentResponse(
            final_text="Stopped: max_steps reached before completion.",
            events=events,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tool_calls=all_tool_calls,
        )
```

The adapter can call OpenAI, Anthropic, a local model, or a routed gateway. The runtime code should not care.

---

## 5. Tool Boundaries and MCP

**MCP** is best understood as a standard way for an AI application to connect to **external context and capabilities**.

| MCP role | Plain English |
|----------|---------------|
| Host | The app the user is using, such as an IDE, desktop assistant, or chat client |
| Client | The connector inside the host that talks to one MCP server |
| Server | The service that exposes tools, resources, and prompts |
| Tool | An action the model may ask to execute |
| Resource | Context or data the model or user may read |
| Prompt | A reusable workflow or message template |

MCP does not remove the need for authorization. It makes the integration shape cleaner, but the host and runtime still need to decide:

- Which server is trusted?
- Which user is asking?
- Which tool is being requested?
- Is the tool read-only, write-capable, external, or destructive?
- Does the user need to approve this call?
- What data will leave the local boundary?

**Engineering rule:** Treat tool descriptions, retrieved resources, and MCP server output as untrusted input. A malicious tool description or retrieved document can try to steer the agent just like a malicious user prompt.

---

## 6. Handoffs and Subagents

The most common design mistake in **multi-agent systems** is using one word, "agent," for **three different things**.

| Pattern | Who owns the user conversation after delegation? | Context model | Best for |
|---------|-----------------------------------------------|---------------|----------|
| Agent as tool | Parent keeps ownership | Isolated, request/response only | Stateless specialist capability |
| Subagent | Parent keeps ownership | Usually filtered or summarized context | Complex bounded sub-problem |
| Handoff | Ownership moves to another agent/state | Shared state across turns | Multi-stage conversational flow |

Plain language:

- **Agent as tool** means "do this one expert function and return."
- **Subagent** means "take this bounded mission, work on it, and come back with a result."
- **Handoff** means "you now own the next part of the conversation."

If you do not **define ownership explicitly**, you will create duplicate work, token bloat, or dead-end flows where no agent knows who should answer the user.

### 6.1 Choosing the right pattern

Use this decision table first. It prevents most over-engineered agent systems.

| If the task looks like this | Use | Why |
|-----------------------------|-----|-----|
| "Generate SQL for this schema." | Agent as tool | Atomic, reusable, strict input/output |
| "Research three vendors and compare them." | Subagent | Multi-step but bounded; parent should still synthesize |
| "Collect account details, then transfer to refund specialist." | Handoff | Sequential stateful conversation with capability unlocking |
| "Search flights, hotels, and attractions at the same time." | Parallel subagents or router | Independent work can run concurrently |
| "Delete production resources after explicit user confirmation." | Handoff plus approval gate | Ownership and risk must be explicit |

Two practical rules:

1. If the specialist must talk to the user for several turns, prefer a handoff.
2. If the parent should remain the coordinator and only needs a result back, prefer a subagent.

### 6.2 Ownership is the real contract

Subagents are useful when ownership is explicit. They become dangerous when they turn into a vague group chat.

Good handoff contract:

```python
from dataclasses import dataclass


@dataclass
class HandoffSpec:
    target_agent: str
    reason: str
    allowed_tools: list[str]
    max_steps: int
    expected_output_schema: dict


def choose_handoff(task: str) -> HandoffSpec | None:
    if "PCB" in task or "schematic" in task:
        return HandoffSpec(
            target_agent="hardware_reviewer",
            reason="Needs hardware design review",
            allowed_tools=["read_repo", "search_datasheets"],
            max_steps=6,
            expected_output_schema={
                "type": "object",
                "properties": {
                    "findings": {"type": "array"},
                    "risk_level": {"type": "string"},
                },
                "required": ["findings", "risk_level"],
            },
        )
    return None
```

Bad handoff contract:

```text
Ask another agent to think about this and see what it says.
```

The bad version has no owner, no permissions, no stopping condition, and no verifiable output.

### 6.3 Context capsules beat full history dumps

The second major failure mode is **context management**. Do not dump the full conversation into every subagent call.

Use a filtered context capsule instead:

```python
from dataclasses import dataclass


@dataclass
class ContextCapsule:
    user_goal: str
    relevant_facts: list[str]
    constraints: list[str]
    accepted_decisions: list[str]
    allowed_tools: list[str]
    expected_output_schema: dict
    max_steps: int = 6
    trace_id: str = ""


@dataclass
class SubagentSpec:
    name: str
    mission: str
    read_only: bool = True
    can_run_in_parallel: bool = True


def build_capsule(task: str) -> ContextCapsule:
    return ContextCapsule(
        user_goal=task,
        relevant_facts=[
            "Board target: Jetson Orin Nano carrier",
            "Constraint: no BOM changes this sprint",
        ],
        constraints=[
            "Do not edit unrelated files",
            "Use only read-only inspection tools",
        ],
        accepted_decisions=[
            "Use UART for first RCP bring-up",
        ],
        allowed_tools=["read_repo", "search_datasheets"],
        expected_output_schema={
            "type": "object",
            "properties": {
                "findings": {"type": "array"},
                "recommended_action": {"type": "string"},
            },
            "required": ["findings", "recommended_action"],
        },
    )
```

What should usually go into a capsule:

- The user goal in one sentence.
- Only the facts relevant to this specialist.
- Non-negotiable constraints.
- Already accepted decisions, so agents do not reopen settled issues.
- Tool permissions.
- Output schema and budget.

What should usually stay out:

- Raw full chat history.
- Internal chain-of-thought.
- Unrelated tool traces.
- Every file ever touched by the parent.

### 6.4 Sequential handoffs vs parallel subagents

Handoffs and subagents are not interchangeable from a control-flow standpoint.

| Question | Handoff | Subagent |
|----------|---------|----------|
| Can it own the next user turn? | Yes | No, parent usually resumes |
| Is it naturally sequential? | Yes | Sometimes, but can often be parallel |
| Does it need shared conversational state? | Often yes | Usually no; pass filtered context |
| Is centralized orchestration preserved? | Less so | Yes |

Use **sequential handoffs** when capabilities unlock in order:

```text
triage -> collect details -> eligibility check -> refund specialist
```

Use **parallel subagents** when work streams do not depend on each other:

```text
research agent
security reviewer
cost estimator
        -> parent synthesizer
```

Parallel subagents are often cheaper than one giant generalist agent because each worker sees only the context it needs. They are a bad choice when every worker needs the same large shared conversational state and must keep talking to the user directly.

### 6.5 Safety patterns that actually help

Subagents are also useful as trust boundaries.

Good uses:

- A read-only research subagent for untrusted web content.
- A verifier subagent that checks a planner's output before execution.
- A red-team or policy subagent that blocks risky tool requests.
- A financial or production-write agent that only activates after explicit approval.

Bad uses:

- Giving every subagent shell access "just in case."
- Letting a reviewer subagent rewrite source files directly when it only needs to inspect.
- Passing secrets to agents that only need summaries.

The delegation boundary should narrow permissions, not widen them.

### 6.6 Implementation checklist

Before you add a handoff or subagent, answer these six questions:

1. Who owns the next user-facing response?
2. What exact context is being passed?
3. Which tools are allowed?
4. What is the stopping condition?
5. What shape must the result have?
6. What happens if the delegate fails or times out?

If you cannot answer those, the design is not ready.

### 6.7 Failure modes

| Failure | What it looks like | Fix |
|---------|--------------------|-----|
| Ownership gap | Both agents wait for each other or both answer the user | Define a single response owner per step |
| Context bloat | Every subagent gets the full transcript | Pass a capsule or summary, not raw history |
| Handoff without closure | Tool call happens but history is malformed | Record the handoff pair or equivalent transition artifact |
| Over-spawning | Five agents created for a simple lookup | Start with a tool or a single agent |
| Hidden side effects | Delegate both routes and mutates data | Separate routing tools from write tools |
| Verification gap | Planner output executes without review | Add a reviewer or approval gate before action |

### 6.8 Design rule

Use the smallest delegation mechanism that preserves correctness:

- Start with a tool.
- Move to a subagent when the work is multi-step or domain-specialized.
- Move to a handoff when conversational ownership must change across turns.

---

## 7. Streaming Is More Than Tokens

For production agents, stream **runtime events**, not only generated text.

Useful event types:

| Event | Why it helps |
|-------|--------------|
| `model_start` | Shows which model and policy profile is active |
| `model_delta` | Streams generated text |
| `tool_call` | Makes hidden action requests visible |
| `tool_result` | Shows what happened after execution |
| `handoff` | Records which agent took ownership |
| `policy_block` | Explains why an action was denied |
| `done` | Gives final output and usage summary |

This makes the system easier to debug and safer to operate. If the UI only streams words, users cannot see when the agent is calling tools or changing ownership.

---

## 8. Guardrails Belong Outside the Prompt

Prompt instructions help, but they are **not a security boundary**. **Runtime controls** must sit outside the LLM.

Minimum control set:

| Control | Example |
|---------|---------|
| Input validation | Reject unsupported file types or oversized prompts |
| Tool allowlist | A finance agent can read invoices but cannot execute shell commands |
| Identity binding | Tool calls execute as the requesting user, not as a global admin |
| Human approval | Deleting files, sending messages, or purchasing items pauses for review |
| Output validation | JSON schema, citation checks, PII scan, unsafe output filter |
| Audit log | Who asked, what context was used, what tools ran, what policy fired |

This matches the main lesson from modern agent security guidance: risk appears during execution, especially when tools, permissions, memory, and external data are involved.

---

## 9. SDK vs Graph vs Gateway

Use this decision table in design reviews.

| Choose | When |
|--------|------|
| Raw provider API | You want full control and the workflow is short |
| Agent SDK | You want managed tool loops, sessions, handoffs, guardrails, and tracing |
| LangGraph-style graph | You need durable state, retries, branches, review gates, and resumability |
| MCP | You want reusable tools/resources/prompts across multiple AI hosts |
| OpenClaw-style gateway | You need persistent sessions, channels, pairing, routing, nodes, or local-first operation |

Most serious products use more than one. For example:

```text
Web/mobile/voice channels
        |
Gateway: session, auth, routing, audit
        |
Workflow runtime: graph, retries, human review
        |
Agent SDK or owned model loop
        |
MCP/tools: files, shell, browser, database, devices
```

---

## 10. Hardware and Systems Implications

Agent runtime choices affect **infrastructure demand**:

- More tool loops mean more small model calls, not just one large call.
- Long sessions increase context, KV-cache pressure, and cost.
- Streaming requires low time-to-first-token and stable tail latency.
- Gateways create always-on workloads that look more like services than batch jobs.
- Local-first assistants increase demand for edge inference, audio pipelines, and device memory.
- Runtime telemetry becomes part of the workload because every tool call and handoff needs logging.

For hardware engineers, this is why agent workloads are not the same as one-shot chatbot workloads.

---

## Key Takeaways

1. Modern agent development is a runtime architecture problem, not just a prompt problem.
2. Keep provider-specific details inside adapters; keep your product contract stable.
3. MCP standardizes how tools, resources, and prompts are exposed, but it does not replace authorization.
4. Subagents need explicit ownership, tool permissions, budgets, and output schemas.
5. Runtime policy and telemetry are required for safe production agents.

---

## Exercises

1. Implement a `ModelAdapter` for one provider and make it return the `ToolCall` objects used in this lecture.
2. Add a `requires_approval=True` tool and make the runtime return a `policy_block` event instead of executing it.
3. Design an MCP server list for a hardware assistant. Mark each server as read-only, write-capable, external, or destructive.
4. Draw an architecture for an OpenClaw-style assistant that can receive a message from Telegram, route to a hardware agent, call a datasheet search tool, and return an answer with citations.

---

## References

- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/)
- [OpenAI API Agents guide](https://platform.openai.com/docs/guides/agents)
- [Model Context Protocol specification](https://modelcontextprotocol.io/specification/2025-11-25)
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)
- [LangChain handoffs documentation](https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs)
- [LangChain multi-agent architecture guide](https://www.langchain.com/blog/choosing-the-right-multi-agent-architecture)
- [Google Cloud ADK: sub-agents versus agents as tools](https://cloud.google.com/blog/topics/developers-practitioners/where-to-use-sub-agents-versus-agents-as-tools)
- [OWASP Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [NIST AI RMF Generative AI Profile](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-generative-artificial-intelligence)

---

**Previous:** [Lecture 06](Lecture-06.md) | **Next:** [Lecture 08 - Multi-Agent Systems](Lecture-08.md)
