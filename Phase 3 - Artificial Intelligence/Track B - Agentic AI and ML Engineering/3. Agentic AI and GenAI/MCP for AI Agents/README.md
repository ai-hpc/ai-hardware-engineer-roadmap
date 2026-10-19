# MCP for AI Agents — The Model Context Protocol in Depth

**Executed current Python path:** [Lecture 05](Lecture-05.md) uses official SDK 2.3.0 `MCPServer`, four actual client tests and a stdio subprocess. It is distinct from earlier wire-compatibility examples.

<div class="course-identity" style="--course-accent: #ea580c; --course-accent-rgb: 234, 88, 12;" markdown="1">
<div class="course-identity__icon">MCP</div>
<div markdown="1">
<p class="course-identity__eyebrow">Phase 3 · Agentic AI · Special Course</p>
<p class="course-identity__title">The open protocol that connects agents to the world — architecture, the core features and historical reverse primitives, building and securing real servers, OAuth 2.1 for remote deployment, and the production stack, current 2026-07-28 requests with explicitly labelled 2025-11-25 compatibility lessons.</p>
<p class="course-identity__meta">Core artifact: a tested local server and failure transcript · Extension: remote authorization and deployment evidence</p>
</div>
</div>

> *Function calling lets one model use one set of tools you wired in by hand. MCP turns "tools" into a protocol — so any agent can talk to any tool, any data source, any system, through one standard interface. It is the integration layer the agent era was missing.*

When Anthropic open-sourced the **Model Context Protocol** in November 2024, the agent ecosystem had an M×N problem: every host (Claude, Cursor, an internal agent) needed a bespoke integration for every tool (GitHub, Postgres, your filesystem, a SaaS API). MCP collapses that to **M + N** — write a tool server *once* against the protocol, and every MCP-capable host can use it; build a host *once*, and it speaks to the whole ecosystem. Interoperability still requires compatible revisions, capabilities, schemas, and authorization; implementing the same protocol name does not make every host/server pairing work.

This course is the engineer's manual for that protocol. Not a tour of one SDK call — a ground-up build of the mental model (host / client / server, the core features and historical reverse primitives, the lifecycle), then the hands-on skills (write a server with the official SDK 2.3.0, exercise an actual stdio client, then study a separate Streamable HTTP and OAuth deployment), then the part that gets skipped and gets people breached (the MCP attack surface and how to shut it down), and finally the production stack (gateways, the registry, tool-context budgets, and evaluation with MCPMark).

**Layer mapping:** the **tool-protocol layer** of the agent stack — the standard interface between the [agent runtime](../Lectures/Lecture-02.md) and every external system it touches. It sits directly on top of [Tool Use & Function Calling](../Lectures/Lecture-08.md) and underneath everything an agent actually *does*.

**Role targets:** AI Agent Engineer · Agent Platform / Infrastructure Engineer · MCP Server Developer · AI Security Engineer · Developer-Tools Engineer.

**Prerequisites:**

* [AI Agent Development 2026 — Lecture 08 (Tool Use & Function Calling)](../Lectures/Lecture-08.md) and [Lecture 09 (Structured Tools Beat Computer Use)](../Lectures/Lecture-09.md) — MCP is the *protocolization* of exactly those ideas.
* [Lecture 02 (What Is an AI Agent Harness?)](../Lectures/Lecture-02.md) — the host/runtime MCP plugs into.
* Comfort with Python (the server lectures use official SDK 2.3.0 `MCPServer` in the executed lesson), JSON, and basic HTTP. OAuth familiarity helps for Lecture 06 but is taught from scratch.

**Start here:** [Current protocol: worked request, result, and failures](Protocol-2026-07-28.md). Lectures 01–04 and 06–08 retain explicitly dated compatibility examples; Lecture 05 is the executed current Python SDK path.

**Pairs with:** the agent-security lectures ([24 — Runtime Discipline](../Lectures/Lecture-24.md), [40 — OpenClaw Threat Model / MITRE ATLAS](../Lectures/Lecture-40.md)) and the [MCPMark benchmark in Lecture 23 §8](../Lectures/Lecture-23.md) — which this course's evaluation lecture builds on.

---

## Why this course is structured the way it is

MCP looks simple — "expose a tool, the agent calls it" — and that simplicity hides three things that bite in production: the protocol has several feature families (and older revisions use reverse requests that the current revision expresses through result/retry interactions); remote servers need **an authorization design appropriate to the selected protocol profile**; storing a credential in an environment variable does not establish caller identity, scopes or audience; and the protocol's openness is also its **attack surface**. The eight lectures climb that exact gradient:

```text
   understand it          build it              ship it safely
   ┌───────────────┐    ┌──────────────┐    ┌────────────────────┐
   01 why it exists     05 write a server    07 the attack surface
   02 architecture      06 transports +      08 production: gateways,
   03 core primitives      OAuth 2.1            registry, eval, scale
   04 reverse primitives
```

You leave understanding the protocol well enough to *read the spec*, build a server worth deploying, and defend it against the attacks that are already happening in the wild.

---

## Course Map (8 lectures)

<div class="lecture-map" markdown>

| # | Lecture | The thread |
|---|---------|-----------|
| [01](Lecture-01.md) | **Why MCP Exists — The M×N Problem & the Protocol** — the integration explosion, "USB-C for AI," JSON-RPC 2.0, the adoption timeline, and when MCP beats hand-wired function calling | the case for a protocol |
| [02](Lecture-02.md) | **Architecture & Lifecycle — Host, Client, Server** — the three roles, capability negotiation, the initialize → operate → shutdown lifecycle, and JSON-RPC message types | the shape of a connection |
| [03](Lecture-03.md) | **Core Primitives — Tools, Resources, Prompts** — the model-/app-/user-controlled trichotomy, JSON-Schema tool definitions, URI-addressed resources, and prompt templates | what a server exposes |
| [04](Lecture-04.md) | **Reverse Primitives — Sampling, Roots, Elicitation** — the server→client direction: borrowing the host's LLM, filesystem boundaries, and asking the user for input mid-task | what makes MCP bidirectional |
| [05](Lecture-05.md) | **Building an MCP Server (SDK 2.3.0)** — validated schemas, structured results, resources and prompts, actual in-memory and stdio tests; distinguish standalone FastMCP | hands on the keyboard |
| [06](Lecture-06.md) | **Transports, Remote Servers & OAuth 2.1** — stdio vs Streamable HTTP, sessions, the SSE deprecation, and authorization (client / resource server / auth server, RFC 9728) | from localhost to the internet |
| [07](Lecture-07.md) | **MCP Security — The Agent Attack Surface** — tool poisoning, prompt injection via tool results, the confused deputy, token passthrough, the lethal trifecta, and the defenses | the part that gets skipped |
| [08](Lecture-08.md) | **Production MCP — Gateways, Registry, Eval & Capstone** — composition and gateways, the registry, tool-context budgets, observability, MCPMark evaluation, and the 2026 roadmap | shipping at scale |

</div>

---

## Course Outcomes

By the end you should be able to:

* Explain **host / client / server**, current per-request versioning, and the older initialization lifecycle; identify which contract a raw exchange uses.
* Use tools/resources/prompts and elicitation under the selected revision; explain why deprecated roots/sampling examples are compatibility material.
* **Build, test, and package** a real MCP server with the official Python SDK / FastMCP, validated against the MCP Inspector.
* Deploy a **remote server over Streamable HTTP** with spec-compliant **OAuth 2.1** authorization (resource-server metadata, scoped tokens, the auth/resource server split).
* Identify and mitigate the **MCP attack surface** — tool poisoning, prompt injection via tool results, confused-deputy, token passthrough, supply-chain — and pass a basic security review.
* Run a server in **production**: compose servers behind a gateway, manage the tool-context budget, publish to the registry, and evaluate against MCPMark.

---

## Currency / Refresh Discipline

MCP is young and moving fast — pin your facts to a spec date:

* The official current revision is **2026-07-28**, verified 2026-10-18. Start with the linked current lesson; `2025-11-25` wire examples are historical compatibility study. Roots, Sampling, and Logging are deprecated; the old six-primitives grouping is not a recommendation to add all six today.
* Transports: **stdio** and **Streamable HTTP** are current; the legacy **HTTP+SSE** transport is **deprecated** (since 2025-03-26) — every lecture uses the current transport and names the deprecated one only to warn you off it.
* SDK surfaces change release-to-release. The official `mcp` SDK and the standalone `fastmcp` distribution are separate projects. Use the lock and tested API in Lecture 05; read dated compatibility examples under their stated revision.

---

## Exit Criteria

The **core learning path** is complete when you can:

1. Run the four tests in Lecture 05, retain the output and explain the stdio transport boundary.
2. Predict the result of a negative input, an unknown tool and an untrusted prompt argument.
3. Explain why schema validation, tool execution and caller authorization are different checks.
4. Identify the protocol revision and SDK version in a new example before combining it with this server.

The **deployment extension** adds an authenticated remote transport, scoped audience-bound tokens, cancellation and retry tests, and a documented threat model. Add gateway, registry or benchmark work only when that deployment needs it. The course's local tests do not claim that HTTP authorization, Inspector, a public registry or a production service has been executed.

A useful diagnostic: an MCP result contains a command to unlock a robot's motors. Is that authorization? **No.** It is tool-result data. A separate trusted application policy must authorize any operation, and physical control still belongs behind the robot's safety constraints.

---

*Related: [Tool Use & Function Calling](../Lectures/Lecture-08.md) · [Evaluation & the MCP benchmark ladder (L23 §8)](../Lectures/Lecture-23.md) · [OpenClaw Threat Model — MITRE ATLAS](../Lectures/Lecture-40.md) · official spec: [modelcontextprotocol.io](https://modelcontextprotocol.io)*

**Study method:** use the [shared lesson cycle and completion rubric](../../../../Learning-Guide.md) with this course's existing labs. Predict first, test one variation, retain a failure case, and revisit the self-check after two days and one week.


## Carry this course into the physical-AI system

**Start from:** A tested bounded task API and an authenticated development environment.

**Build and challenge:** Expose observation metadata and task-status tools first. Test schema errors, cancellation, authorization, and duplicate requests; protocol support alone does not authorize an actuator operation.

**Pass forward:** A versioned supervisory tool interface and failure/retry transcript. Continue with [the connected course](../../4.%20ML%20Engineering%20and%20MLOps/Guide.md) and keep the artifact in the [Physical AI Design Studio](../../../../Physical-AI-Design-Studio.md).
