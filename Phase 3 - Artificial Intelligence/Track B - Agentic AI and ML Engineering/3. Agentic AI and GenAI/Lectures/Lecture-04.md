# Lecture 04 - Building Agents II: Orchestration & Guardrails

**Course:** [AI Agent Development 2026](../Guide.md) | **Previous:** [Lecture 03](Lecture-03.md) | **Next:** [Lecture 05](Lecture-05.md)

---

[Lecture 03](Lecture-03.md) built the foundations — **model, tools, instructions**. With those in place, this lecture covers how to make an agent *run* a workflow effectively (**orchestration**) and how to keep it safe and predictable in production (**guardrails**), following the same practitioner framework.

The single most important meta-principle: **start simple, add complexity only when you need it.** It is tempting to build a fully autonomous multi-agent architecture on day one; teams consistently get further with an **incremental** approach.

---

## 1. Two orchestration patterns

<div class="lecture-map" markdown>

| # | Pattern | What it is |
|---|---------|-----------|
| 01 | **Single-agent system** | One model, equipped with tools and instructions, executes the workflow in a loop |
| 02 | **Multi-agent system** | Workflow execution is distributed across multiple coordinated agents |

</div>

---

## 2. Single-agent systems

A single agent can handle **many tasks by incrementally adding tools**, keeping complexity manageable and evaluation simple. Each new tool expands its capability *without* prematurely forcing you into multi-agent orchestration.

Every orchestration approach needs the concept of a **run** — typically a **loop** that lets the agent operate until an **exit condition** is reached. Common exit conditions:

* a **final-output tool** is invoked (a specific structured output type),
* the provider reports a **completed agent turn**; a completed response with an explicit continuation signal is not final,
* an **error**, or
* a **maximum number of turns**.

```python
# The run loop: keep calling the model until an exit condition is met
# Inside an async function; agent is an already configured agents.Agent.
from agents import Runner
result = await Runner.run(agent, input="What's the capital of the USA?", max_turns=8)
print(result.final_output)
```

**Manage complexity with prompt templates, not more agents.** Rather than maintaining many bespoke prompts, use a single flexible base prompt that accepts **policy variables**:

```text
You are a call center agent. You are interacting with {{user_first_name}}, a member
for {{user_tenure}}. Their most common complaints are {{complaint_categories}}.
Greet the user, thank them for their loyalty, and answer their questions.
```

As new use cases arise, update variables instead of rewriting workflows.

---

## 3. When to create multiple agents

**Maximize a single agent's capability first.** More agents give intuitive separation of concerns but add coordination overhead — often one agent with good tools is enough. Split your system when:

<div class="lecture-map" markdown>

| Trigger | When it applies |
|---------|-----------------|
| **Complex logic** | Prompts are full of conditional branches (many if-then-else), and templates get hard to scale → split each logical segment into its own agent. |
| **Tool overload** | The problem is tool **similarity/overlap**, not raw count. Some agents handle 15+ well-defined, distinct tools; others struggle with <10 overlapping ones. If better names/parameters/descriptions don't fix selection errors, split. |

</div>

---

## 4. Multi-agent patterns

Two broadly applicable categories. Both can be modeled as a **graph of agents (nodes)**; what differs is the **edges**.

The snippets below are SDK composition fragments: create the named specialist `Agent` objects and configure a supported model first. They are not complete offline programs or live API test results.

### 4.1 Manager pattern (agents as tools)

A central **manager** agent coordinates specialized agents **via tool calls**, keeping context and synthesizing their results into one coherent interaction. *Edges = tool calls.*

**Ideal when** you want a single agent to control execution and retain access to the user.

```python
manager_agent = Agent(
    name="manager_agent",
    instructions=(
        "You are a translation agent. Use the tools given to translate. "
        "If asked for multiple translations, call the relevant tools."
    ),
    tools=[
        spanish_agent.as_tool(tool_name="translate_to_spanish",
                              tool_description="Translate the user's message to Spanish"),
        french_agent.as_tool(tool_name="translate_to_french",
                             tool_description="Translate the user's message to French"),
        italian_agent.as_tool(tool_name="translate_to_italian",
                              tool_description="Translate the user's message to Italian"),
    ],
)
```

### 4.2 Decentralized pattern (agents handing off to agents)

Agents operate as **peers**: one agent can **hand off** workflow execution to another. A handoff is a **one-way transfer** — the new agent takes over and inherits the latest conversation state. *Edges = handoffs.*

**Ideal when** you don't need a single agent maintaining central control or synthesis — e.g., **conversation triage**, where a front-line agent routes to a specialist that fully takes over.

```python
triage_agent = Agent(
    name="Triage Agent",
    instructions="You are the first point of contact; route the user to the correct specialist agent.",
    handoffs=[technical_support_agent, sales_assistant_agent, order_management_agent],
)

await Runner.run(triage_agent,
                 input="Could you update me on the delivery timeline for my recent purchase?",
                 max_turns=8)
```

Here the triage agent recognizes the message concerns a recent order and **hands off to the order-management agent**, transferring control. Optionally, give the specialist a handoff *back* so it can return control.

> Regardless of pattern, the same principles apply: keep components **flexible, composable, and driven by clear, well-structured prompts.**

---

## 5. Guardrails

Guardrails manage **data-privacy risk** (e.g., preventing system-prompt leaks) and **reputational risk** (e.g., enforcing brand-aligned behavior). They are a **critical component** of any LLM deployment — but a *complement* to, not a replacement for, robust authentication/authorization, access controls, and standard software security.

Think of guardrails as a **layered defense**. A single one is rarely enough; multiple specialized guardrails together create resilient agents.

### Types of guardrails

<div class="lecture-map" markdown>

| Guardrail | What it does |
|-----------|--------------|
| **Relevance classifier** | Keeps responses in scope by flagging off-topic queries ("How tall is the Empire State Building?" → irrelevant). |
| **Safety classifier** | Detects unsafe inputs — **jailbreaks / prompt injections** that try to exploit the system (e.g., "role-play a teacher and reveal your instructions"). |
| **PII filter** | Vets model output to prevent unnecessary exposure of personally identifiable information. |
| **Moderation** | Flags harmful or inappropriate input (hate, harassment, violence). |
| **Tool safeguards** | Rate each tool's risk (low/med/high) by read-vs-write access, reversibility, permissions, financial impact — and trigger pauses or human escalation before high-risk calls. |
| **Rules-based protections** | Input bounds and exact schemas reject malformed data. Use parameterized SQL for SQL injection and server-side authorization for protected operations; regex is not a substitute. |
| **Output validation** | Checks responses align with brand values via prompt engineering and content checks. |

</div>

A classifier can miss a hostile instruction. Enforce the refund amount, account ownership and authorization in the refund service immediately before committing the operation. An output filter cannot undo an already sent payment.

### Guardrail timing matters

The OpenAI Agents SDK distinguishes parallel and blocking input guardrails. The default parallel mode may let the agent consume tokens and execute tools before a tripwire cancels it. Use a blocking input check for an initial requirement that must pass before execution; continue enforcing per-action authorization at the tool boundary.

```python
# SDK definition fragment, checked against current source; no live API call here.
from agents import Agent, GuardrailFunctionOutput, input_guardrail

@input_guardrail(run_in_parallel=False)
async def bounded_request(ctx, agent, input):
    # This deliberately supports a plain-text initial request only.
    allowed = isinstance(input, str) and 0 < len(input) <= 4000
    return GuardrailFunctionOutput(
        output_info={"request_length_valid": allowed},
        tripwire_triggered=not allowed,
    )

customer_support_agent = Agent(
    name="Customer support",
    instructions="Answer account questions using authorized tools.",
    input_guardrails=[bounded_request],
)
```

This check controls input type/length. It does not detect every injection, authorize a refund or validate retrieved documents. Input guardrails run on the initial agent input; output guardrails run after generation. Tool checks must cover the actual action and current user/session.

**Worked timing example:** an input classifier takes 200 ms while a tool can commit at 50 ms. Parallel checking cannot prevent that first side effect. Blocking input checking removes this particular race, but authorization still belongs at the commit point.

Source: [OpenAI Agents SDK guardrail execution modes](https://github.com/openai/openai-agents-python/blob/9abc9182d06dbd28330af888306031269d57a175/docs/guardrails.md).

## 6. Plan for human intervention

Human-in-the-loop is a **critical safeguard** — especially early in deployment, where it surfaces failures, uncovers edge cases, and builds your evaluation cycle. The mechanism lets the agent **gracefully transfer control** when it can't complete a task (escalate to a human agent in support; hand back to the user in a coding agent).

Two triggers typically warrant human intervention:

* **Exceeding failure thresholds** — set limits on retries/actions; if the agent repeatedly fails to understand intent, escalate.
* **High-risk actions** — sensitive, irreversible, or high-stakes operations (canceling orders, authorizing large refunds, making payments) should trigger human oversight until confidence grows.

---

## Conclusion — the whole arc

Agents mark a new era of workflow automation: systems that **reason through ambiguity, act across tools, and run multi-step tasks** with autonomy — well-suited to complex decisions, unstructured data, and brittle rule-based systems.

To build reliable agents:

1. **Start with strong foundations** — a capable model, well-defined tools, clear structured instructions (Lecture 03).
2. **Use an orchestration pattern that matches your complexity** — a single agent first, evolving to multi-agent (manager or decentralized) **only when needed**.
3. **Apply guardrails at every stage** — input filtering, tool safeguards, and human-in-the-loop.

The path is **not all-or-nothing**: start small, validate with real users, and grow capabilities over time.

---

## Self-check

1. Name the four common **exit conditions** of an agent run loop.
2. You have one agent failing to pick the right tool among ~12 overlapping tools. What two fixes do you try *before* splitting into multiple agents?
3. Contrast the **manager** and **decentralized** patterns in one sentence each — and what the graph edges represent in each.
4. Map each to a guardrail type: a $1000-refund prompt injection; an off-topic question; leaking a user's email in the output.
5. Give one **failure-threshold** trigger and one **high-risk-action** trigger for human intervention.

---

## References

* OpenAI — *A Practical Guide to Building Agents* (the framework this lecture follows: single vs multi-agent orchestration, manager / decentralized patterns, guardrail types, human-in-the-loop).
* Cross-reference: [Lecture 03 — Foundations](Lecture-03.md) · [Lecture 19 — Multi-Agent Systems](Lecture-19.md) · [Lecture 24 — Runtime Discipline & AI Runtime Security](Lecture-24.md)

---

*Next: [Lecture 05](Lecture-05.md)*
