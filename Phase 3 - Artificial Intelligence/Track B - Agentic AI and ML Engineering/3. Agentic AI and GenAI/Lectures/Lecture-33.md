# Lecture 33 - Structured Tools Beat Computer Use: Interface Hierarchy for Agents

**Course:** [Agentic AI & GenAI](../Guide.md) | **Previous:** [Lecture 32](Lecture-32.md) | **Next:** [Lecture 34](Lecture-34.md)

---

**Computer-use agents** are visually impressive.

They can look at screenshots, click buttons, type into fields, and operate software like a human.

That does not mean **vision** is the right primary interface for agents.

The Reflex benchmark gives a useful data point:

```text
same app
same task
same model family
vision path: 53 ± 13 steps, ~551k input tokens, ~17 minutes
API path:    8 calls,       ~12k input tokens, ~20 seconds
```

The conclusion is not "never use vision."

The conclusion is:

```text
Vision-based computer use is a fallback interface.
Structured tools are the primary interface when you control the system.
```

For OpenClaw-style architecture, this is not a small optimization.

It is a **tool-design principle**.

---

## Learning objectives

By the end of this lecture, you should be able to:

1. Explain why screenshot-driven agents are expensive and nondeterministic.
2. Compare structured APIs, CLI/direct execution, DOM/accessibility, and vision interfaces.
3. Explain the Reflex benchmark and its limitations.
4. Design an interface hierarchy for agent tools.
5. Identify when vision is appropriate and when it is wasteful.
6. Connect structured tools to verification, security, and auditability.
7. Apply the principle to OpenClaw Gateway tools, node commands, exec, and App SDK surfaces.

---

## 1. The benchmark in plain terms

Reflex tested **two ways** for Claude Sonnet to operate the same admin panel.

Task:

```text
find the customer named Smith with the most orders
locate their most recent pending order
accept all of their pending reviews
mark the order as delivered
```

Path A:

```text
vision agent
  -> browser-use
  -> screenshots
  -> clicks
  -> rendered UI state
```

Path B:

```text
API agent
  -> tool calls
  -> HTTP endpoints mapped to app handlers
  -> structured JSON responses
```

Result:

| Metric | Vision agent | API agent |
|---|---:|---:|
| Steps / calls | 53 ± 13 | 8 ± 0 |
| Wall-clock time | 1003s ± 254s | 19.7s ± 2.8s |
| Input tokens | 550,976 ± 178,849 | 12,151 ± 27 |
| Output tokens | 37,962 ± 10,850 | 934 ± 41 |

The API path completed in 8 calls every time.

The vision path **initially missed work** because not all pending reviews were visible on screen. It needed a 14-step UI walkthrough to complete successfully.

That walkthrough is itself **engineering cost**.

---

## 2. Why vision is expensive

A vision agent pays for **perception every step**.

Each loop looks like:

```text
screenshot
  -> interpret pixels
  -> infer UI state
  -> decide action
  -> click/type
  -> wait
  -> screenshot again
```

Costs compound because:

- screenshots are large inputs
- UI state must be rediscovered repeatedly
- scrolling/pagination may be invisible
- every action is sequential
- each screen transition adds another model call
- the model must infer meaning from layout and pixels

Structured APIs avoid most of that.

API loop:

```text
call tool
  -> receive JSON
  -> choose next tool
  -> verify result
```

The agent **reads the data directly** instead of re-deriving it from pixels.

---

## 3. Interface bandwidth hierarchy

Use the **highest-bandwidth interface** available.

```text
Structured API / typed tool call       best
CLI / direct execution                 good
DOM / accessibility tree               acceptable
Vision / screenshots                   fallback
```

Why this hierarchy exists:

| Interface | Signal quality | Determinism | Cost | Security shape |
|---|---|---|---|---|
| Structured API | high | high | low | scoped contracts |
| CLI/direct exec | high if command is stable | medium/high | low/medium | command policy required |
| DOM/accessibility | medium | medium | medium | app/UI state dependent |
| Vision | low/medium | low | high | broad visible authority |

The rule:

```text
If a task can be expressed as a function, do not do it through screenshots.
```

---

## 4. Structured tools are more than cheaper

Cost is only **one dimension**.

Structured tools also improve:

### Determinism

```json
{ "tool": "update_order", "args": { "id": 123, "status": "delivered" } }
```

is more deterministic than:

```text
click the button below the order status field
```

### Composability

Tool calls can be **chained, retried, validated, and logged**.

### Verifiability

You can assert:

```text
response.status == "delivered"
```

and preserve exact evidence.

### Security

Structured tools can expose narrow capabilities:

```text
list_customers
accept_review
update_order_status
```

Vision exposes **whatever the agent can see and click**.

### Auditability

Structured logs tell you exactly what happened:

```text
tool=update_order_status id=123 from=pending to=delivered actor=session-abc
```

Screenshots require reconstruction.

---

## 5. Why the vision path failed first

The benchmark's first vision attempt missed pending reviews below the visible fold.

That is not primarily a **model-intelligence problem**.

It is an **interface problem**.

The UI showed a partial rendered state.

The API returned structured pagination and result data.

The agent using screenshots had to infer:

```text
is this all the data?
is there pagination?
should I scroll?
did the filter apply?
what changed after the click?
```

The API agent read:

```text
page
total results
review status
order status
```

The **same application logic** existed underneath both paths.

Only one path exposed it directly.

---

## 6. When vision still makes sense

Do not remove computer use entirely.

**Vision is useful** when:

- you do not control the target system
- there is no API
- the workflow is a third-party SaaS UI
- you are doing UX/QA validation
- you are reverse-engineering a legacy workflow
- the task is inherently visual
- you need human-parity behavior

Good use:

```text
vision agent explores unknown workflow
  -> extract actions and state
  -> design structured tools
  -> structured agent executes at scale
```

Bad use:

```text
vision agent repeatedly operates an internal app you control
even though the app can expose handlers or endpoints
```

Vision is an **exploration and fallback layer**.

It should not be the **default execution layer** for owned systems.

---

## 7. OpenClaw architecture implication

OpenClaw's **Gateway/tool direction** is aligned with this benchmark.

The preferred path:

```text
Agent
  -> typed tool schema
  -> Gateway RPC / tool invoke
  -> policy and approvals
  -> execution layer
  -> structured result
  -> session/artifact evidence
```

Vision/computer-use should plug in as:

```text
fallback tool: computer_use
```

not as the primary route.

Example interface priority:

| Task | Preferred interface |
|---|---|
| run local command | `exec` / node `system.run` with policy |
| query app state | Gateway RPC / structured API |
| update internal record | typed tool call |
| inspect rendered UI | DOM/accessibility snapshot |
| operate unknown SaaS UI | vision fallback |

The stable principle:

```text
Agents should behave like infrastructure when infrastructure interfaces exist.
```

They should behave like humans **only when forced to**.

---

## 8. Tool schema layer

A structured-first agent platform needs a **tool schema layer**.

Example:

```json
{
  "name": "update_order_status",
  "description": "Update one order's delivery status.",
  "input_schema": {
    "type": "object",
    "required": ["order_id", "status"],
    "properties": {
      "order_id": { "type": "integer" },
      "status": {
        "type": "string",
        "enum": ["pending", "delivered", "cancelled"]
      }
    }
  }
}
```

Execution path:

```text
tool schema
  -> validation
  -> auth scope check
  -> approval policy if needed
  -> handler call
  -> structured response
  -> audit event
```

This is the opposite of screenshot automation.

The model expresses intent through a **narrow typed action**.

The runtime decides **whether the action is allowed**.

---

## 9. Auto-generating tools

The Reflex article matters partly because Reflex 0.9 can **expose event handlers as HTTP endpoints**, reducing the engineering cost of the API surface.

General pattern:

```text
OpenAPI -> tool schemas
GraphQL -> tool schemas
internal service definitions -> tool schemas
CLI specs -> tool wrappers
typed app handlers -> agent-callable endpoints
```

The decision flips when **API generation is cheap**.

Old assumption:

```text
writing APIs is expensive
so use screenshots
```

New possibility:

```text
generate structured APIs cheaply
so avoid screenshots
```

For OpenClaw-like systems, this suggests:

- prefer Gateway RPC method surfaces
- expose typed node commands
- generate tool wrappers from API specs
- keep vision as a fallback adapter

---

## 10. Security boundary comparison

Structured tools:

```text
tool name
input schema
scope requirement
approval rule
handler
audit log
```

Vision agent:

```text
screenshot
natural-language reasoning
click/type action
implicit UI permissions
harder-to-parse evidence
```

Structured tools are **easier to secure** because the action is explicit before execution.

You can ask:

```text
Who called this?
Which scope allowed it?
Was approval required?
Which object changed?
What was the before/after state?
```

With screenshots, the action is often a **low-level click**.

The **semantic meaning** must be reconstructed.

That is weaker for **audit and incident response**.

---

## 11. Verification and Agent Skills

Lecture 29 argued:

```text
No evidence, no completion.
```

Structured tools make **evidence easier**.

Example:

```json
{
  "tool": "update_order_status",
  "result": {
    "order_id": 123,
    "old_status": "pending",
    "new_status": "delivered",
    "updated_at": "2026-05-06T12:00:00Z"
  }
}
```

That result can be:

- logged
- asserted
- replayed
- attached to a session
- shown in an App SDK UI
- checked by a final-answer hook

Vision evidence is heavier:

- screenshots
- cursor actions
- OCR
- natural-language descriptions
- brittle UI state

Use vision **when necessary**.

Do not choose it when **structured evidence** is available.

---

## 12. Design rule for OpenClaw tools

Adopt this rule:

```text
Every recurring agent action should graduate toward a structured tool.
```

Lifecycle:

```text
one-off human action
  -> vision exploration
  -> manual CLI/API discovery
  -> typed tool schema
  -> policy/approval
  -> test fixture
  -> production tool
```

Example:

```text
Use computer_use to learn how an admin workflow behaves.
Then build update_customer, list_reviews, approve_review, update_order tools.
Then stop using computer_use for that workflow.
```

The result is **faster, cheaper, safer, and more reviewable**.

---

## 13. Mini-lab: convert a UI workflow into tools

Pick one internal workflow:

- approve a user
- create a device pairing
- update an order
- run a deployment
- capture a node screenshot
- restart a service

Step 1: write the UI path:

```text
screen -> click -> form -> submit -> verify
```

Step 2: identify the underlying state transition:

```text
object
allowed states
required fields
side effects
permissions
```

Step 3: design the tool:

```text
name
input schema
output schema
required scope
approval rule
audit event
test fixture
```

Step 4: define when vision is still allowed:

```text
only if structured tool is unavailable
only for exploration
only with explicit operator approval
```

---

## 14. Benchmark exercise

Recreate the Reflex comparison on a small app you control.

Measure:

| Metric | Vision/DOM path | Structured tool path |
|---|---:|---:|
| steps |
| wall-clock time |
| input tokens |
| output tokens |
| failure modes |
| required prompt instructions |
| audit quality |

Then answer:

```text
What did the vision agent need to rediscover?
What did the structured tool expose directly?
What evidence was easy to capture?
Which path would you trust in production?
```

---

## Key takeaways

- Vision-based computer use is a fallback, not the primary interface for owned systems.
- The Reflex benchmark measured a large gap: roughly 551k input tokens and 53 steps for vision versus about 12k input tokens and 8 calls for API.
- The vision path required a 14-step walkthrough to succeed; that prompt is unpaid engineering cost.
- The right hierarchy is structured API, CLI/direct execution, DOM/accessibility, then vision.
- Structured tools are cheaper, faster, more deterministic, easier to secure, and easier to audit.
- OpenClaw's Gateway/tool direction is aligned with this principle.
- Recurring workflows should graduate from vision exploration into typed, policy-gated tools.

---

## References

- Reflex, "Computer use is 45x More Expensive Than Structured APIs": [https://reflex.dev/blog/computer-use-is-45x-more-expensive-than-structured-apis/](https://reflex.dev/blog/computer-use-is-45x-more-expensive-than-structured-apis/)
- Reflex benchmark repo: [https://github.com/reflex-dev/agent-benchmark](https://github.com/reflex-dev/agent-benchmark)
- Lecture 23 - Gateway RPC Protocol: [Lecture-23.md](Lecture-23.md)
- Lecture 29 - Agent Skills: [Lecture-29.md](Lecture-29.md)
- Lecture 31 - Runtime Strategy: [Lecture-31.md](Lecture-31.md)

---

*Next: [Lecture 34 - Nemotron 3 Nano Omni: Multimodal Perception Sub-Agents for Agent Systems](Lecture-34.md)*
