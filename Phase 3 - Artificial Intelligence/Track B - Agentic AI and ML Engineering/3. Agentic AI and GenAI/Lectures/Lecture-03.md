# Lecture 03 - Building Agents I: Foundations (Model, Tools, Instructions)

**Course:** [AI Agent Development 2026](../Guide.md) | **Previous:** [Lecture 02](Lecture-02.md) | **Next:** [Lecture 04](Lecture-04.md)

---

This lecture and the next distill the **practitioner framework for building agents** — the design choices that separate a reliable production agent from a demo. It follows the structure popularized by OpenAI's practical guidance on building agents, grounded in patterns seen across many real deployments.

This lecture covers the **foundations**: what an agent actually is, when you should (and should not) build one, and the three components every agent is made of — **model, tools, and instructions**. Lecture 04 covers orchestration and guardrails.

---

## 1. What is an agent?

> **Agents are systems that independently accomplish tasks on your behalf.**

Conventional software lets a user *streamline and automate* a workflow. An agent goes further: it performs the workflow **on the user's behalf, with a high degree of independence**.

A **workflow** is a sequence of steps that must be executed to meet a goal — resolving a support ticket, booking a reservation, committing a code change, generating a report.

**What is *not* an agent:** applications that integrate an LLM but don't use it to *control workflow execution* — simple chatbots, single-turn LLM calls, sentiment classifiers. Wrapping a model call is not agency.

An agent has two core characteristics that let it act reliably on a user's behalf:

1. **It uses an LLM to manage workflow execution and make decisions.** It recognizes when a workflow is complete, can proactively correct its own actions, and — on failure — can halt and **transfer control back to the user**.
2. **It has access to tools** to interact with external systems (both to gather context and to take action) and **dynamically selects** the right tool for the current state — always within **clearly defined guardrails**.

---

## 2. When should you build an agent?

Building an agent means rethinking how your system makes decisions. Agents shine exactly where **deterministic, rule-based automation falls short**.

The canonical example is **payment fraud analysis**. A traditional rules engine works like a *checklist*, flagging transactions against preset criteria. An LLM agent works more like a **seasoned investigator** — weighing context, considering subtle patterns, and catching suspicious activity even when no hard rule is violated. That nuanced reasoning is what lets agents handle complex, ambiguous situations.

Prioritize workflows that have **resisted automation**, especially where traditional methods hit friction:

<div class="lecture-map" markdown>

| Signal | What it looks like | Example |
|--------|--------------------|---------|
| **Complex decision-making** | Nuanced judgment, exceptions, context-sensitive calls | Refund approval in customer service |
| **Difficult-to-maintain rules** | Rulesets grown unwieldy; updates costly or error-prone | Vendor security reviews |
| **Heavy reliance on unstructured data** | Interpreting natural language, extracting from documents, conversational interaction | Processing a home-insurance claim |

</div>

> **Validate first.** Before committing to an agent, confirm your use case clearly meets these criteria. If it doesn't, a **deterministic solution may suffice** — and will be cheaper, faster, and easier to reason about. (Same discipline as the inference course: don't reach for the heavy tool when a simple one fits.)

---

## 3. Agent design foundations — the three components

In its most fundamental form, an agent has three components:

<div class="lecture-map" markdown>

| # | Component | Role |
|---|-----------|------|
| 01 | **Model** | The LLM powering the agent's reasoning and decision-making |
| 02 | **Tools** | External functions or APIs the agent can use to take action |
| 03 | **Instructions** | Explicit guidelines and guardrails defining how the agent behaves |

</div>

In code (using an agents framework), this is as small as:

```python
weather_agent = Agent(
    name="Weather agent",
    instructions="You are a helpful agent who can talk to users about the weather.",
    tools=[get_weather],
)
```

The rest of this lecture takes each component in turn.

---

## 4. Selecting your model

Different models trade off **task complexity, latency, and cost**. Not every task needs the smartest model — a simple retrieval or intent-classification step can run on a smaller, faster model, while a hard decision (approve a refund?) benefits from a more capable one. You will often use **a mix of models** across one workflow.

The approach that works: **prototype with the most capable model for every task to establish a performance baseline.** Then swap in smaller models and see whether they still hit your accuracy target. This way you never prematurely cap the agent's ability, and you learn exactly where small models succeed or fail.

The principles for choosing a model:

1. **Set up evals** to establish a performance baseline.
2. **Meet your accuracy target** with the best models available.
3. **Optimize for cost and latency** by replacing larger models with smaller ones where possible.

> This is the agent-layer mirror of the inference course's whole thesis: the model is a knob you tune against a measured target, not a default you accept.

---

## 5. Defining tools

Tools extend an agent by calling the **APIs** of underlying applications. For **legacy systems without APIs**, agents can fall back on **computer-use models** that drive web and application UIs directly — just as a human would.

Each tool should have a **standardized, well-documented, tested, reusable definition**. That enables flexible many-to-many relationships between tools and agents, improves discoverability, simplifies versioning, and prevents redundant definitions.

Broadly, agents need three types of tools:

<div class="lecture-map" markdown>

| Type | What it does | Examples |
|------|--------------|----------|
| **Data** | Retrieve the context needed to execute the workflow | Query a transaction DB or CRM, read a PDF, search the web |
| **Action** | Interact with systems to *do* something | Send an email/text, update a CRM record, hand a ticket off to a human |
| **Orchestration** | Other **agents used as tools** (see the Manager pattern, Lecture 04) | Refund agent, research agent, writing agent |

</div>

```python
from agents import Agent, WebSearchTool, function_tool

@function_tool
def save_results(output):
    db.insert({"output": output, "timestamp": datetime.time()})
    return "File saved"

search_agent = Agent(
    name="Search agent",
    instructions="Help the user search the internet and save results if asked.",
    tools=[WebSearchTool(), save_results],
)
```

As the number of required tools grows, **consider splitting tasks across multiple agents** (Lecture 04, Orchestration).

---

## 6. Configuring instructions

High-quality **instructions** are essential for any LLM app and *especially* critical for agents: clear instructions reduce ambiguity, improve decision-making, and produce smoother execution with fewer errors.

**Best practices for agent instructions:**

<div class="lecture-map" markdown>

| Practice | Why |
|----------|-----|
| **Use existing documents** | Turn SOPs, support scripts, and policy docs into LLM-friendly **routines**. In customer service, routines roughly map to individual knowledge-base articles. |
| **Prompt agents to break down tasks** | Smaller, clearer steps from dense resources minimize ambiguity and help the model follow along. |
| **Define clear actions** | Every step should map to a specific action or output (ask for the order number; call an API). Being explicit — even about the wording of a user-facing message — leaves less room for misinterpretation. |
| **Capture edge cases** | Real interactions create decision points (incomplete info, unexpected questions). Anticipate common variations with conditional steps and branches. |

</div>

A practical accelerator: use an advanced model to **auto-generate instructions from existing documents**, e.g.:

```text
You are an expert in writing instructions for an LLM agent. Convert the following
help-center document into a clear, unambiguous, numbered set of instructions,
written as directions for an agent. The document to convert is: {{help_center_doc}}
```

---

## Key takeaways

* An **agent independently executes a multi-step workflow** using an LLM to drive control flow plus tools — not just a wrapped model call.
* Build an agent when the workflow needs **nuanced judgment, has unwieldy rules, or leans on unstructured data**; otherwise prefer a deterministic solution.
* Every agent = **model + tools + instructions**. Baseline with the strongest model, then optimize down; give it **standardized data / action / orchestration tools**; and write **explicit, routine-based instructions that capture edge cases**.

---

## Self-check

1. Give two examples of LLM applications that are **not** agents, and say what they're missing.
2. Your team wants to automate refund approvals that currently need a human to weigh exceptions. Is this a good agent candidate? Which of the three "when to build" signals applies?
3. Why prototype with the most capable model first, then swap smaller models in — rather than starting small?
4. Classify each as a data / action / orchestration tool: `read_pdf`, `send_email`, `research_agent.as_tool()`.
5. What is the risk of vague instructions for an agent specifically (vs a single-turn chatbot)?

---

## References

* OpenAI — *A Practical Guide to Building Agents* (the framework this lecture follows: agent definition, when-to-build criteria, model / tools / instructions).
* Cross-reference: [Lecture 08 — Tool Use & Function Calling](Lecture-08.md) · [Lecture 17 — Agent SDKs and Runtime APIs](Lecture-17.md) · [Lecture 04 — Orchestration & Guardrails](Lecture-04.md)

---

*Next: [Lecture 04](Lecture-04.md)*
