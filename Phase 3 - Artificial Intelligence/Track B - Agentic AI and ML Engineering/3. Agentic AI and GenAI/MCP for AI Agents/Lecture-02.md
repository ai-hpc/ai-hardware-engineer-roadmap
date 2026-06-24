# Lecture 02 - Architecture & Lifecycle: Host, Client, Server

**Collection:** [MCP for AI Agents](README.md) | **Previous:** [← Lecture 01](Lecture-01.md) | **Next:** [Lecture 03](Lecture-03.md)

---

Lecture 01 framed *why* the Model Context Protocol exists: Anthropic open-sourced MCP in November 2024 to replace bespoke, per-integration glue code with one wire protocol that any LLM application can speak to any capability provider. This lecture is about the *shape* of that protocol — the three roles it defines, the message framing it borrows from JSON-RPC 2.0, and the connection lifecycle that every MCP session walks through from `initialize` to shutdown.

The mental model worth fixing before anything else: MCP is a **client-server protocol with a strict 1:1 connection rule**. A single application — the Host — can talk to many servers at once, but it does so by running one Client per server, and each Client holds exactly one connection to exactly one Server. There is no fan-out inside a single Client, no shared socket, no multiplexing of two servers over one Client. Once you internalize that the Host is a *fleet of Clients* rather than a single connection manager, the rest of the architecture falls out cleanly.

Everything in this lecture pins to the latest stable spec, **2025-11-25**. Where behavior is versioned — capability negotiation, transports, the protocol-version handshake — I name the spec revision, because MCP's wire contract has changed materially across revisions and "it depends on the version" is the correct senior-engineer answer more often than not.

---

## Learning objectives

By the end of this lecture you should be able to:

- Distinguish the three MCP roles — Host, Client, Server — and explain why the Client↔Server relationship is strictly 1:1.
- Read and write the three JSON-RPC 2.0 message types (request, response, notification) and identify which fields each carries.
- Explain capability negotiation: how Host and Server advertise features in `initialize` and why an undeclared capability is never invoked.
- Walk the full connection lifecycle — `initialize` → `notifications/initialized` → operation → shutdown — and describe protocol-version negotiation.
- Articulate why an MCP session is logically stateful, why that complicates horizontal scaling, and how the 2026-07-28 RC stateless core changes the picture.

---

## 1. The three roles: Host, Client, Server

MCP defines exactly three roles. They are not interchangeable, and a real deployment always has all three.

The **Host** is the LLM application the user actually interacts with — Claude Desktop, Claude Code, an IDE like Cursor, or any agent runtime you build yourself. The Host owns everything that is *not* a capability: it holds the language model (or the API keys to call one), it owns the user-facing surface, and — critically for security — it owns the **approval UX**. When a server offers a tool that deletes files, it is the Host that decides whether to surface a confirmation prompt before that tool runs. The Host is the trust boundary; servers are untrusted-by-default code on the other side of a wire.

The **Client** is the protocol connector that lives *inside* the Host. It is not a separate process you deploy; it is the component the Host instantiates to speak MCP to one server. The Client handles the JSON-RPC plumbing — framing requests, matching responses to request IDs, dispatching notifications — and it holds **exactly one connection to exactly one Server**. If a Host wants to talk to three servers (a filesystem server, a GitHub server, a database server), it runs three Clients.

The **Server** is the capability provider. It exposes some combination of Tools, Resources, and Prompts (the server→client primitives) over the connection. A Server can be a **local subprocess** the Host launches and talks to over stdio, or a **remote service** the Host reaches over HTTP. The protocol is identical either way — only the transport differs.

```text
                          ┌──────────────────────────────────────────────┐
                          │                   HOST                        │
                          │   (LLM app: Claude Desktop / Code / Cursor)   │
                          │   owns: the model, user, API keys, approvals  │
                          │                                                │
                          │   ┌──────────┐   ┌──────────┐   ┌──────────┐  │
                          │   │ Client A │   │ Client B │   │ Client C │  │
                          │   └────┬─────┘   └────┬─────┘   └────┬─────┘  │
                          └────────┼──────────────┼──────────────┼────────┘
                                   │ 1:1          │ 1:1          │ 1:1
                                   ▼              ▼              ▼
                            ┌──────────┐   ┌──────────┐   ┌──────────┐
                            │ Server A │   │ Server B │   │ Server C │
                            │ (stdio   │   │ (remote  │   │ (remote  │
                            │ subproc) │   │  HTTP)   │   │  HTTP)   │
                            └──────────┘   └──────────┘   └──────────┘
```

The single most important invariant to carry forward: **one Client : one Server**. A Client never bridges two servers, and a Server is never shared by two Clients within the same Host session. The Host is the only component that sees the whole picture; each Client sees exactly its own server.

---

## 2. The JSON-RPC 2.0 foundation

MCP does not invent a wire format. It is built on **JSON-RPC 2.0**, which gives it three message types and a small, well-understood envelope.

A **request** expects a response. It carries `jsonrpc` (always `"2.0"`), an `id` (a string or number the sender chooses to correlate the eventual response), a `method` (the operation name, e.g. `tools/list`), and optional `params`.

A **response** answers exactly one request. It echoes the same `id` and carries either a `result` (on success) or an `error` (on failure) — never both.

A **notification** is a one-way message that expects **no response**. It looks like a request but has **no `id`** — that missing `id` is precisely what marks it as fire-and-forget. The receiver processes it and sends nothing back.

Here is a real `tools/list` request the Client sends to ask the Server what tools it exposes:

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/list",
  "params": {}
}
```

And the Server's response, correlated by the matching `id`:

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "tools": [
      {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "inputSchema": {
          "type": "object",
          "properties": {
            "city": { "type": "string" }
          },
          "required": ["city"]
        }
      }
    ]
  }
}
```

Contrast that round-trip with a notification. When the Client finishes initialization (Section 4), it sends:

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/initialized"
}
```

No `id`, no response. The Server simply transitions its own state and moves on. The presence or absence of `id` is the whole distinction: a message with an `id` is a request and *will* be answered; a message without one is a notification and *will not* be. Servers also use notifications in the other direction — for example, a `notifications/tools/list_changed` message tells the Client the tool set changed and it should re-fetch, again with no response expected.

---

## 3. Capability negotiation

MCP is deliberately a *negotiated* protocol, not an assumed one. Neither side may use a feature unless the other side has declared support for it during `initialize`. This is what lets a minimal server (tools only) and a rich server (tools, resources, prompts, logging) coexist behind the same Host without the Host guessing.

The negotiation is symmetric. The **Server declares what it offers**; the **Client declares what it can provide back**. The connection then uses only the intersection.

Server-declared capabilities map to the three server→client primitives, plus operational features:

| Capability (server) | What it enables |
| --- | --- |
| `tools` | Server exposes callable tools (`tools/list`, `tools/call`) |
| `resources` | Server exposes readable resources (`resources/list`, `resources/read`) |
| `prompts` | Server exposes prompt templates (`prompts/list`, `prompts/get`) |
| `logging` | Server can emit structured log notifications to the client |

Client-declared capabilities map to the three reverse (client→server) primitives:

| Capability (client) | What it enables |
| --- | --- |
| `sampling` | Server may ask the client's LLM to generate a completion |
| `roots` | Server may query the filesystem roots the client has granted |
| `elicitation` | Server may request additional input from the user via the client |

The rule that follows is strict and worth stating plainly: **a server that did not declare `tools` will never be sent `tools/list`**, and a client that did not declare `sampling` will never receive a sampling request. Capabilities are not hints or preferences — they are the contract. Code that issues a request for an undeclared capability is a protocol violation, and a correct peer will reject it. This is why the six primitives split cleanly into two directions: Tools/Resources/Prompts flow server→client, while Sampling/Roots/Elicitation flow client→server, and each end advertises only its own side.

---

## 4. The lifecycle

Every MCP connection moves through a defined lifecycle. There are no shortcuts — you cannot call a tool before the handshake completes.

The phases are: **initialize** (request/response), **initialized** (notification), **operation** (the working phase), and **shutdown**.

**Phase 1 — `initialize` request.** The Client opens the connection by sending an `initialize` request carrying three things: the `protocolVersion` it wants to speak (a date string), its `clientInfo`, and its `capabilities`.

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": {
    "protocolVersion": "2025-11-25",
    "clientInfo": { "name": "ExampleHost", "version": "1.4.0" },
    "capabilities": {
      "sampling": {},
      "roots": { "listChanged": true },
      "elicitation": {}
    }
  }
}
```

**Phase 2 — the Server responds.** The Server replies to that one request with its *own* `protocolVersion`, its `serverInfo`, and its `capabilities`. This single exchange is where both sides learn what the other supports.

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "protocolVersion": "2025-11-25",
    "serverInfo": { "name": "WeatherServer", "version": "2.0.1" },
    "capabilities": {
      "tools": { "listChanged": true },
      "resources": {},
      "logging": {}
    }
  }
}
```

**Phase 3 — `notifications/initialized`.** Once the Client has the Server's capabilities, it sends the `notifications/initialized` notification (shown in Section 2). This is fire-and-forget — it signals "handshake complete, I am ready to operate." Only after this point may either side issue operational requests.

**Phase 4 — operation.** This is the working phase: `tools/list`, `tools/call`, `resources/read`, `prompts/get`, and any reverse-direction `sampling`/`elicitation` requests the negotiated capabilities permit. Requests and notifications flow in both directions for the life of the session.

**Phase 5 — shutdown.** The connection is closed. For a stdio server the Host typically closes the input stream and the subprocess exits; for an HTTP server the session is torn down (and its `Mcp-Session-Id` retired). There is no dedicated `shutdown` JSON-RPC method — shutdown is a transport-level action.

**Protocol-version negotiation** happens entirely in Phase 1/2. Versions are **date strings** (`2025-11-25`, `2025-06-18`, and so on). The Client proposes a version in `initialize`; the Server responds with the version it will actually use. If the Server supports the requested version, it echoes it. If it does not, it responds with a version it *does* support, and the Client decides whether it can speak that. A Client unwilling to downgrade should treat the mismatch as a failed connection rather than proceeding on an unsupported contract. Negotiating the version *before* any operational traffic is what keeps two peers from talking past each other.

---

## 5. Stateful vs stateless

An MCP session is **logically stateful**. The `initialize` handshake establishes negotiated capabilities that hold for the whole session; resource and tool-list subscriptions persist; and over Streamable HTTP the session is bound to an `Mcp-Session-Id` that the Client returns on every subsequent request. The Server is expected to remember "this session negotiated these capabilities and holds these subscriptions."

That statefulness is the natural model for a long-lived connection, but it complicates **horizontal scaling**. If session state lives in the memory of one server instance, then every request for that session must land on that same instance — sticky routing — or the state must be externalized to a shared store that all instances consult. Both are real operational costs: sticky routing fights load balancers and complicates failover; a shared store adds latency and a new dependency. For a high-fan-out deployment serving many agents, a per-session-stateful server is a scaling bottleneck.

This is the gap the **2026-07-28 release candidate** addresses with a **STATELESS core**. The RC defines a core protocol mode that carries no required per-session server state, which lets a server scale on ordinary HTTP infrastructure — any instance can handle any request because there is no session memory to be sticky to. (The same RC also adds the MCP Apps and Tasks extensions, layered on top of that stateless core.) Note the transport history here: MCP's two transports are **stdio** and **Streamable HTTP** (with `Mcp-Session-Id` sessions); the older **HTTP+SSE** transport has been **deprecated since 2025-03-26**. The stateless core is the next step in that arc — moving the protocol toward deployments that don't need session affinity at all.

Whichever mode you run, the architectural constant is the same: **the Host mediates everything between the model and the servers.** The model never speaks to a server directly. The Host receives the model's intent, routes it through the appropriate Client to the right Server, applies its approval UX, and feeds results back to the model. That mediation role — the Host as the single point that owns the model, the user, the keys, and the trust boundary — is exactly the agent-harness pattern developed in the companion lecture, [`../Lectures/Lecture-02.md`](../Lectures/Lecture-02.md): MCP is one concrete realization of a harness standing between an LLM and the outside world.

---

## Current as of

This lecture is current as of **June 2026**, pinned to the latest stable MCP specification, **2025-11-25**. The **2026-07-28 release candidate** introduces a **stateless core** (allowing servers to scale on ordinary HTTP without per-session affinity), along with the MCP Apps and Tasks extensions; treat those as forthcoming until the RC is ratified. Transport status as of this writing: **stdio** and **Streamable HTTP** are current; **HTTP+SSE** has been deprecated since **2025-03-26**.
