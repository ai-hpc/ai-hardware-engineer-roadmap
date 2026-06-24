# MCP for AI Agents — The Model Context Protocol in Depth

<div class="course-identity" style="--course-accent: #ea580c; --course-accent-rgb: 234, 88, 12;" markdown="1">
<div class="course-identity__icon">MCP</div>
<div markdown="1">
<p class="course-identity__eyebrow">Phase 3 · Agentic AI · Special Course</p>
<p class="course-identity__title">The open protocol that connects agents to the world — architecture, the six primitives, building and securing real servers, OAuth 2.1 for remote deployment, and the production stack, current to the 2025-11-25 spec and the 2026 release candidate.</p>
<p class="course-identity__meta">Artifact: a tested, authenticated, hardened MCP server with tools, resources, and prompts · Measure: spec-compliant lifecycle, passing MCP Inspector, scoped auth, and a clean security review</p>
</div>
</div>

> *Function calling lets one model use one set of tools you wired in by hand. MCP turns "tools" into a protocol — so any agent can talk to any tool, any data source, any system, through one standard interface. It is the integration layer the agent era was missing.*

When Anthropic open-sourced the **Model Context Protocol** in November 2024, the agent ecosystem had an M×N problem: every host (Claude, Cursor, an internal agent) needed a bespoke integration for every tool (GitHub, Postgres, your filesystem, a SaaS API). MCP collapses that to **M + N** — write a tool server *once* against the protocol, and every MCP-capable host can use it; build a host *once*, and it speaks to the whole ecosystem. By early 2026 that bet had paid off comprehensively: OpenAI, Google DeepMind, and Microsoft all adopted MCP, the official `mcp` package crossed **~97M monthly downloads**, and the public registry passed **2,000 community servers**. MCP is now the default way agents reach the world, and "can it speak MCP?" is a question you will answer on the job.

This course is the engineer's manual for that protocol. Not a tour of one SDK call — a ground-up build of the mental model (host / client / server, the six primitives, the lifecycle), then the hands-on skills (write a server with FastMCP, test it with Inspector, deploy it over Streamable HTTP with OAuth 2.1), then the part that gets skipped and gets people breached (the MCP attack surface and how to shut it down), and finally the production stack (gateways, the registry, tool-context budgets, and evaluation with MCPMark).

**Layer mapping:** the **tool-protocol layer** of the agent stack — the standard interface between the [agent runtime](../Lectures/Lecture-02.md) and every external system it touches. It sits directly on top of [Tool Use & Function Calling](../Lectures/Lecture-08.md) and underneath everything an agent actually *does*.

**Role targets:** AI Agent Engineer · Agent Platform / Infrastructure Engineer · MCP Server Developer · AI Security Engineer · Developer-Tools Engineer.

**Prerequisites:**

* [AI Agent Development 2026 — Lecture 08 (Tool Use & Function Calling)](../Lectures/Lecture-08.md) and [Lecture 09 (Structured Tools Beat Computer Use)](../Lectures/Lecture-09.md) — MCP is the *protocolization* of exactly those ideas.
* [Lecture 02 (What Is an AI Agent Harness?)](../Lectures/Lecture-02.md) — the host/runtime MCP plugs into.
* Comfort with Python (the server lectures use the official SDK / FastMCP), JSON, and basic HTTP. OAuth familiarity helps for Lecture 06 but is taught from scratch.

**Pairs with:** the agent-security lectures ([24 — Runtime Discipline](../Lectures/Lecture-24.md), [40 — OpenClaw Threat Model / MITRE ATLAS](../Lectures/Lecture-40.md)) and the [MCPMark benchmark in Lecture 23 §8](../Lectures/Lecture-23.md) — which this course's evaluation lecture builds on.

---

## Why this course is structured the way it is

MCP looks simple — "expose a tool, the agent calls it" — and that simplicity hides three things that bite in production: the protocol has **six primitives, not one** (and two of them run *backwards*, server→client); remote servers need **real authorization**, not an API key in an env var; and the protocol's openness is also its **attack surface**. The eight lectures climb that exact gradient:

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
| [05](Lecture-05.md) | **Building an MCP Server (FastMCP)** — the official Python SDK end to end: tools, resources, prompts, context, lifespan, structured output; testing with MCP Inspector | hands on the keyboard |
| [06](Lecture-06.md) | **Transports, Remote Servers & OAuth 2.1** — stdio vs Streamable HTTP, sessions, the SSE deprecation, and authorization (client / resource server / auth server, RFC 9728) | from localhost to the internet |
| [07](Lecture-07.md) | **MCP Security — The Agent Attack Surface** — tool poisoning, prompt injection via tool results, the confused deputy, token passthrough, the lethal trifecta, and the defenses | the part that gets skipped |
| [08](Lecture-08.md) | **Production MCP — Gateways, Registry, Eval & Capstone** — composition and gateways, the registry, tool-context budgets, observability, MCPMark evaluation, and the 2026 roadmap | shipping at scale |

</div>

---

## Course Outcomes

By the end you should be able to:

* Explain the **host / client / server** architecture and the full **initialize → operate → shutdown** lifecycle, and read a raw JSON-RPC MCP exchange.
* Use all **six primitives** correctly — Tools, Resources, Prompts (server→client) and Sampling, Roots, Elicitation (the reverse direction) — and say which actor controls each.
* **Build, test, and package** a real MCP server with the official Python SDK / FastMCP, validated against the MCP Inspector.
* Deploy a **remote server over Streamable HTTP** with spec-compliant **OAuth 2.1** authorization (resource-server metadata, scoped tokens, the auth/resource server split).
* Identify and mitigate the **MCP attack surface** — tool poisoning, prompt injection via tool results, confused-deputy, token passthrough, supply-chain — and pass a basic security review.
* Run a server in **production**: compose servers behind a gateway, manage the tool-context budget, publish to the registry, and evaluate against MCPMark.

---

## Currency / Refresh Discipline

MCP is young and moving fast — pin your facts to a spec date:

* This course tracks the **2025-11-25** stable revision and flags what the **2026-07-28 release candidate** changes (a stateless core, **MCP Apps** for server-rendered UIs, a **Tasks** extension for long-running work, and tighter OAuth/OIDC alignment).
* Transports: **stdio** and **Streamable HTTP** are current; the legacy **HTTP+SSE** transport is **deprecated** (since 2025-03-26) — every lecture uses the current transport and names the deprecated one only to warn you off it.
* SDK surfaces (FastMCP, the official `mcp` package heading to a v2) change release-to-release; treat code as a snapshot and verify against the installed version. Every lecture closes with a **`## Current as of`** note.

---

## Exit Criteria

You are done with this course when you can stand up a **production-grade MCP server** end to end:

* Expose tools, resources, and prompts with correct schemas, and demonstrate it green in MCP Inspector.
* Serve it remotely over Streamable HTTP behind OAuth 2.1 with scoped, audience-bound tokens.
* Walk an auditor through its threat model — tool poisoning, injection-via-results, confused deputy, token passthrough — and the control that stops each.
* Register it, put it behind a gateway with other servers without blowing the tool-context budget, and report an MCPMark-style evaluation of an agent driving it.

If you can call one tool but can't defend the server or deploy it for someone else to trust, you have a demo. The point of this course is the server other people can safely depend on.

---

*Related: [Tool Use & Function Calling](../Lectures/Lecture-08.md) · [Evaluation & the MCP benchmark ladder (L23 §8)](../Lectures/Lecture-23.md) · [OpenClaw Threat Model — MITRE ATLAS](../Lectures/Lecture-40.md) · official spec: [modelcontextprotocol.io](https://modelcontextprotocol.io)*
