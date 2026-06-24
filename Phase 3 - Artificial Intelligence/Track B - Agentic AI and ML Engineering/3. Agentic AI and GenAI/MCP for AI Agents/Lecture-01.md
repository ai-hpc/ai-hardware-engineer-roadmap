# Lecture 01 - Why MCP Exists: The M×N Problem & the Protocol

**Collection:** [MCP for AI Agents](README.md) | **Previous:** [← Course index](README.md) | **Next:** [Lecture 02](Lecture-02.md)

---

Every agent you build eventually hits the same wall. The model is smart enough; the demo works; then someone asks it to read from Postgres, file an issue in GitHub, and check the on-call rotation in PagerDuty — and you spend the next two weeks writing glue. Each tool has its own auth, its own payload shape, its own error semantics, and you hand-wire all of it into your one agent. Then a second team builds a second agent and writes the *same* glue again, because your integration lived inside your harness and nobody can reuse it. That is the integration tax the agent era ran into, and it is the problem the **Model Context Protocol (MCP)** was created to remove.

MCP, open-sourced by Anthropic in **November 2024**, makes "a tool" a thing you publish *once* against a wire protocol instead of a thing you wire into one app. Write a server for GitHub once and every MCP-capable host — Claude Desktop, Claude Code, Cursor, your internal agent — can use it without knowing anything about GitHub's API. Build a host once and it speaks to the entire catalog of servers other people already wrote. The bet was that a shared protocol would beat a thousand bespoke integrations, and by 2026 it has paid off comprehensively: OpenAI, Google DeepMind, and Microsoft all adopted MCP through 2025, the `mcp` PyPI package crosses **~97M monthly downloads**, and the official registry lists **more than 2,000 servers**.

This first lecture makes the case for *why a protocol*, not yet how to build one. We will count the integration explosion precisely, name what MCP actually standardizes (a wire protocol over JSON-RPC 2.0 — not a framework, not an agent), walk the adoption timeline and the spec revisions you must pin your work to, compare MCP against the hand-wired function calling you already know, and close with the mental model — host, client, server, and the six primitives — that the rest of the course builds on.

---

## Learning objectives

1. Quantify the **M×N integration explosion** and explain precisely how MCP collapses it to **M + N**.
2. State what MCP standardizes — a **JSON-RPC 2.0 wire protocol** for discovering and invoking a server's tools, resources, and prompts — and what it deliberately does *not* (a framework or an agent).
3. Recount the **adoption timeline and spec revisions** (Nov 2024 launch → 2025-03-26 → 2025-06-18 → 2025-11-25 stable → 2026-07-28 RC) and explain why a protocol wins on **network effects** once everyone speaks it.
4. Decide, for a given system, **when MCP wins** over plain function calling and when plain function calling is the right call.
5. Sketch the **host / client / server** model and name the **six primitives** and the actor that controls each, forward-referencing Lectures 02–04.

---

## 1. The M×N integration explosion

Put numbers on the pain. Say you have **M hosts** — distinct LLM applications: Claude Desktop, Cursor, a customer-support agent, an internal data agent. And **N tools or data sources** — GitHub, Postgres, the local filesystem, Slack, a payments API. Before MCP, connecting them is a matrix: every host needs custom code to talk to every tool, because each host has its own way of declaring tools and each tool has its own API. That is **M × N** bespoke integrations. Four hosts and five tools is twenty integrations; the moment you add the sixth tool, you owe six more — one per host. Worse, the integration lives *inside* each host, so none of it is reusable and all of it rots independently.

MCP collapses the matrix into two lists. Each host implements the protocol **once** (that is the M). Each tool is wrapped in a **server** that speaks the protocol **once** (that is the N). Now any host talks to any server through the same interface, and the cost is **M + N**, not M × N. Write the GitHub server once → all four hosts get GitHub. Build a new host once → it speaks to all five servers (and the other ~2,000 in the registry) on day one.

```text
        BEFORE: M × N bespoke integrations          AFTER: M + N (write each once)

   Hosts                Tools                     Hosts          MCP          Servers
   ┌──────────┐        ┌──────────┐               ┌──────────┐   bus    ┌──────────────┐
   │ Claude   │━━━━━━━▶│ GitHub   │               │ Claude   │──┐    ┌──│ GitHub server│
   │ Desktop  │━━┓ ┏━━▶│ Postgres │               │ Desktop  │  │    │  └──────────────┘
   └──────────┘  ┃ ┃   └──────────┘               └──────────┘  │    │  ┌──────────────┐
   ┌──────────┐  ┃ ┃   ┌──────────┐               ┌──────────┐  ├────┼──│ Postgres srv │
   │ Cursor   │━━╋━╋━━▶│ Filesystem│              │ Cursor   │──┤ MCP│  └──────────────┘
   └──────────┘  ┃ ┃   └──────────┘               └──────────┘  │    │  ┌──────────────┐
   ┌──────────┐  ┃ ┃   ┌──────────┐               ┌──────────┐  ├────┼──│ Filesystem   │
   │ Support  │━━┛ ┗━━▶│ Slack    │               │ Support  │──┤    │  └──────────────┘
   │ agent    │━━━━━━━▶│ Payments │               │ agent    │──┘    └──│ Slack server │
   └──────────┘        └──────────┘               └──────────┘          └──────────────┘
     M = 4               N = 5                       4 + 5 = 9 implementations,
     4 × 5 = 20 integrations to build & maintain     each written once and reused
```

Two analogies make the shape stick. The first is the one the ecosystem uses: MCP is **"USB-C for AI."** Before USB-C, every device had its own connector and you carried a drawer of adapters; USB-C defined one physical-and-electrical contract, and now any compliant peripheral plugs into any compliant port. MCP is that contract for connecting models to tools and data — one standard plug instead of a drawer of integrations.

The second analogy is the direct technical precedent, and it is worth more than the USB-C image because it already proved the math works. The **Language Server Protocol (LSP)**, introduced by Microsoft in 2016, faced the identical explosion in a different domain: **E editors × L languages**, each editor needing a hand-built integration for each language's autocomplete, go-to-definition, and diagnostics. LSP defined a JSON-RPC protocol between editors and "language servers" so that a language is implemented once as a server and every LSP-speaking editor gets it. E × L became E + L, the ecosystem exploded, and editors stopped competing on "which languages do you support." MCP is LSP's idea applied to agents and tools — and, like LSP, it is built on JSON-RPC.

---

## 2. What MCP actually standardizes

Be precise about the noun, because the most common misconception is that MCP is a framework or an agent. It is neither. **MCP is a wire protocol** — a specification of the JSON messages two programs exchange — built on **JSON-RPC 2.0**. It does not run your agent loop, does not choose which tool to call, does not ship a runtime. It standardizes exactly one thing: **how a host discovers and invokes the tools, resources, and prompts a server exposes**, over a transport, with a defined lifecycle.

Concretely, the protocol pins down the message shapes. A host's client lists what a server offers and calls into it with JSON-RPC requests; the server answers with JSON-RPC results. Discovery and invocation of a tool look like this on the wire:

```json
// client → server: discover what the server exposes
{ "jsonrpc": "2.0", "id": 1, "method": "tools/list" }

// server → client: the catalog (name, human description, JSON-Schema for inputs)
{ "jsonrpc": "2.0", "id": 1, "result": { "tools": [
  { "name": "get_issue",
    "description": "Fetch a GitHub issue by number.",
    "inputSchema": { "type": "object",
      "properties": { "repo": {"type":"string"}, "number": {"type":"integer"} },
      "required": ["repo","number"] } } ] } }

// client → server: invoke it
{ "jsonrpc": "2.0", "id": 2, "method": "tools/call",
  "params": { "name": "get_issue", "arguments": { "repo": "anthropics/sdk", "number": 42 } } }
```

The payoff of standardizing *only* the wire is **decoupling**: the tool author and the host author never coordinate. The person who wrote the GitHub server has never heard of your agent; you have never read their code. You agree on `tools/list` and `tools/call` and the JSON-Schema in between, and that is the entire contract. This is what "M + N" actually buys you — not just fewer lines of code, but the removal of the human coordination that made the M × N matrix expensive in the first place.

What the protocol does *not* do is equally load-bearing for the rest of this course. It does not decide *whether* to call `get_issue` — that is the model's job, inside the host. It does not implement the GitHub call — that is the server author's job. MCP is the contract in the middle, and keeping it that narrow is exactly why so many independent parties could adopt it so fast.

---

## 3. Adoption & momentum

A protocol is only worth standardizing on if it wins, and MCP's trajectory is the argument. The timeline:

| Date | Milestone |
|---|---|
| **Nov 2024** | Anthropic open-sources MCP; first spec, stdio transport, the core primitives. |
| **2025-03-26** | Adds **Streamable HTTP** transport and an **authorization** framework (OAuth); deprecates the legacy HTTP+SSE transport. |
| **2025-06-18** | Spec revision refining the protocol surface (elicitation, structured tool output, security guidance). |
| **2025-11-25** | **Latest stable revision** — what you pin production work to today. |
| **2026-07-28** | **Release candidate**: a stateless core, **MCP Apps** (server-rendered UI), a **Tasks** extension for long-running work, and tighter OAuth alignment. |

Across that window the protocol went from "interesting thing one lab published" to industry default. **OpenAI, Google DeepMind, and Microsoft** all adopted MCP through 2025 — when the three other largest model and platform vendors back a competitor's protocol, that is not politeness, it is recognition that a shared standard is worth more than a proprietary one. The scale numbers track the adoption: the `mcp` PyPI package now does **~97M downloads a month**, and the official registry passed **2,000 servers**.

The mechanism behind this is **network effects**, and it is why a protocol wins once enough parties speak it. Every new MCP-capable host makes every existing server more valuable (one more place it runs), and every new server makes every host more valuable (one more capability it gains for free). Value compounds on both sides of the M + N. A bespoke integration helps exactly one host–tool pair; an MCP server helps every current and future host at once. Past a tipping point — which MCP crossed in 2025 — *not* speaking the protocol is the expensive choice, because you forfeit the entire ecosystem. "Can it speak MCP?" became a real procurement question, and that is the surest sign a protocol has won.

---

## 4. MCP vs hand-wired function calling

You already know function calling from [Tool Use & Function Calling](../Lectures/Lecture-08.md): you define a tool with a name, a description, and a JSON-Schema; the model emits a structured call; your code runs it and returns the result. MCP does not replace that loop — it sits **on top of it**. An MCP tool is surfaced to the model as an ordinary function-calling tool; the difference is *where the tool definition comes from and who can reuse it*. With plain function calling the definition is hardcoded in your app. With MCP it is discovered at runtime from a server that any host can connect to.

| Dimension | Hand-wired function calling | MCP |
|---|---|---|
| **Where tools live** | Hardcoded in one host's source | In standalone servers, reusable by any host |
| **Reuse across hosts** | None — re-implement per app | Write once, every host connects |
| **Discovery** | Static, baked in at build time | Dynamic — `tools/list` at runtime |
| **Team coupling** | Tool author *is* host author | Tool and host authors never coordinate |
| **Swapping the host** | Rewrite all integrations | Point a new host at the same servers |
| **Per-call overhead** | Direct in-process function call | Adds a protocol hop (JSON-RPC over a transport) |
| **Reverse capabilities** | Not available | Sampling, Roots, Elicitation (server → client) |

**When MCP wins:** you want the same tool usable across multiple hosts; you want to pull from the ecosystem of existing servers instead of building everything; you need **dynamic discovery** (the tool set changes without redeploying the host); the tool team and the host team are different people who should not have to coordinate; or you expect to **swap hosts** and keep your integrations.

**When plain function calling is fine:** a single application with a handful of private tools, no reuse story, where the tool author and app author are the same person — the protocol hop and the server scaffolding buy you nothing. And **latency-critical inner loops**, where the extra JSON-RPC round trip over a transport is overhead you cannot afford: an in-process function call is faster than any protocol, and if a tool runs in the hottest part of your loop and is never shared, call it directly. The honest framing is that MCP trades a little per-call overhead and some setup for reuse, decoupling, and ecosystem access; below the threshold where those matter, the simpler tool wins.

---

## 5. The mental model preview

The rest of this course hangs on one diagram and six nouns. The architecture is three roles. A **Host** is the LLM application — Claude Desktop, Claude Code, Cursor, your internal agent. The host runs one or more **Clients**, and each client holds a **1:1 connection to exactly one Server**. A server wraps a capability — GitHub, Postgres, your filesystem — and exposes it through the protocol. ([Lecture 02](Lecture-02.md) details this and the `initialize → operate → shutdown` lifecycle.)

```text
   ┌──────────────── Host (LLM app: Claude Code, Cursor, …) ─────────────┐
   │   Client A ──────1:1──────▶ Server (GitHub)                         │
   │   Client B ──────1:1──────▶ Server (Postgres)                       │
   │   Client C ──────1:1──────▶ Server (filesystem)                     │
   └────────────────────────────────────────────────────────────────────┘
```

The capabilities a connection carries are the **six primitives**, split by direction. Three go **server → client** (what a server offers the host): **Tools** are model-controlled actions the model chooses to invoke; **Resources** are app-controlled, URI-addressed data the host pulls in for context; **Prompts** are user-controlled templates a user deliberately selects. ([Lecture 03](Lecture-03.md) covers this trichotomy.) Three run the *other* way, **server → client requests** that make MCP bidirectional: **Sampling** lets a server ask the host to run an LLM completion on its behalf; **Roots** are the filesystem or URI boundaries the client grants a server to operate within; **Elicitation** lets a server ask the user for structured input mid-task. ([Lecture 04](Lecture-04.md) covers these reverse primitives.) Hold onto the directions and the controlling actor for each — that table is the spine of everything that follows.

---

## Current as of

This lecture is current as of **June 2026** and pins to the **2025-11-25** stable MCP specification — the revision to target for production work today. The protocol remains built on JSON-RPC 2.0 with the host/client/server architecture and six primitives described above; the adoption and scale figures (cross-vendor adoption by OpenAI, Google DeepMind, and Microsoft; ~97M monthly `mcp` downloads; 2,000+ registry servers) reflect early-2026 reporting. Watch the **2026-07-28 release candidate**, which introduces a stateless core, MCP Apps (server-rendered UI), a Tasks extension for long-running work, and tighter OAuth alignment — none of which changes the M×N argument or the mental model here, but which will shape the transport and deployment lectures. Treat SDK surfaces as snapshots and verify against your installed version; the protocol facts above are the stable ground.
