# Lecture 03 - Core Primitives: Tools, Resources, Prompts

**Collection:** [MCP for AI Agents](README.md) | **Previous:** [← Lecture 02](Lecture-02.md) | **Next:** [Lecture 04](Lecture-04.md)

---

In [Lecture 02](Lecture-02.md) you watched a connection come up: host and server exchange `initialize`, negotiate capabilities, and settle into the operate phase. Now we open the operate phase and look at *what a server actually exposes*. The three server→client primitives — **Tools**, **Resources**, and **Prompts** — are the entirety of what a server offers a host on this side of the connection. (The reverse direction, server *asking the host* for things, is [Lecture 04](Lecture-04.md).) Everything an MCP server does for an agent lands in exactly one of these three buckets.

The temptation, coming from plain function calling, is to see all three as "tools with different names." Resist it. The protocol draws a deliberate line through them based on **who decides to use the thing** — the model, the application, or the human. That single distinction, the *control trichotomy*, is the most important idea in this lecture, because it is what makes MCP safe to wire into a host UI. A tool the model can fire on its own is a fundamentally different security and UX object from a document the application chose to attach or a slash-command the user typed. Get the trichotomy right and the rest — schemas, URIs, templates — is detail.

Everything here is JSON-RPC 2.0 over the transport from [Lecture 02](Lecture-02.md), pinned to the **2025-11-25** stable spec. We will read the actual wire messages, not SDK sugar; [Lecture 05](Lecture-05.md) puts FastMCP on top of them.

---

## Learning objectives

By the end of this lecture you should be able to:

- State the **control trichotomy** — Tools are model-controlled, Resources are application-controlled, Prompts are user-controlled — and explain why that separation matters for UX and safety.
- Define a tool with a JSON-Schema `inputSchema`, read a `tools/call` request and result, and interpret content blocks, `outputSchema`/structured content, and the `isError` flag.
- Address data with **resource URIs** and **resource templates** (RFC 6570), and subscribe to resource changes via `resources/subscribe` and `notifications/resources/updated`.
- Define a **prompt** with typed arguments and read the message list returned by `prompts/get`.
- Map each primitive to how a host *surfaces* it — tools as callable functions, resources as attachable context, prompts as user commands.

---

## 1. The control trichotomy — the key mental model

Three primitives, three different actors deciding when each is used. This is the table to memorize:

| Primitive | Control | Who decides to use it | Read-only? | Typical example |
|-----------|---------|-----------------------|------------|-----------------|
| **Tools** | model-controlled | the LLM, mid-generation, chooses to call it | No — actions, side effects | `create_issue`, `send_email`, `run_query` |
| **Resources** | application-controlled | the host app decides what context to pull in | Yes — read-only data | a file's contents, a DB row, an API response, a log |
| **Prompts** | user-controlled | the human invokes it (slash command, menu item) | N/A — a template | `/summarize`, "Plan a trip", a code-review template |

Read each row as a sentence: *the model* decides to call a **tool**; *the application* decides to surface a **resource**; *the user* decides to invoke a **prompt**. The actor in the middle column is the whole point.

Why does the protocol bother splitting them this way instead of shipping one generic "capability"? Two reasons, both load-bearing.

**UX.** Each actor needs a different control surface in the host. Tools become functions offered to the model. Resources become things a user can attach or an app can auto-include — `@`-mentions, file pickers, context panels. Prompts become commands the user fires — slash-menus, buttons. If you collapse all three into "tools," the host has no principled way to render them, and the user loses the distinction between *the model did this* and *I asked for this*.

**Safety.** The dangerous primitive is the one the **model** controls. Tools take actions and have side effects, and the model fires them on its own initiative — so tools are exactly where a host needs an **approval gate**, an audit trail, and the annotations we cover in §2. Resources are read-only and the *application* chose them, so the blast radius is "the model saw some data the app already decided to share." Prompts are inert templates the *user* explicitly invoked. The trichotomy is, at bottom, a statement about where the trust boundary sits: **model-controlled = guard it; user/app-controlled = it was already authorized by a human or the host.** Lecture 07 builds the entire MCP threat model on this foundation — for now, internalize that "who controls it" is a security claim, not a taxonomy convenience.

One more consequence that trips people up, and which §3 returns to: **resources are not auto-injected.** A server *offering* a resource does not put it in the model's context. The application decides. Keep that in your head as we go through each primitive in turn.

---

## 2. Tools in depth

Tools are the model-controlled primitive: the LLM, while generating, decides to call one. They are the MCP analogue of the function-calling you saw in [Tool Use & Function Calling](../Lectures/Lecture-08.md) — but *protocolized*, so any host can discover and invoke them without bespoke wiring.

### Discovery and invocation

Two methods carry the whole lifecycle:

- **`tools/list`** — the host asks the server for its catalog of tools. Each entry carries a `name`, a human-readable `description`, an `inputSchema`, and optionally an `outputSchema` and `annotations`.
- **`tools/call`** — the host invokes one tool by `name` with an `arguments` object, and gets back a result.

### The `inputSchema`

Every tool declares its parameters as a **JSON Schema** object in `inputSchema`. This is what lets the model produce well-formed arguments and what lets the host validate them before the call goes out. A tool definition from `tools/list`:

```json
{
  "name": "create_issue",
  "title": "Create GitHub Issue",
  "description": "Open a new issue in a repository.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "repo":  { "type": "string", "description": "owner/name, e.g. octo/hello" },
      "title": { "type": "string" },
      "body":  { "type": "string" },
      "labels": {
        "type": "array",
        "items": { "type": "string" }
      }
    },
    "required": ["repo", "title"]
  },
  "annotations": {
    "title": "Create GitHub Issue",
    "readOnlyHint": false,
    "destructiveHint": false,
    "idempotentHint": false,
    "openWorldHint": true
  }
}
```

### `outputSchema` and structured content (2025 addition)

Originally a tool result was just unstructured content blocks (text, images). The 2025 spec added an optional **`outputSchema`** on the tool definition plus **structured content** in the result, so a tool can return typed, machine-readable JSON that the host can validate against the schema — not just a blob of text the model has to re-parse. When a tool declares an `outputSchema`, its result carries a `structuredContent` field conforming to that schema. This is what turns a tool from "returns a string" into "returns a `WeatherReport` object."

### Tool annotations — and why a host uses them (forward-ref: security)

Annotations are **hints** about a tool's behavior that the host can use to drive its approval UX. They are advisory metadata, not enforcement — the server asserts them, and a careful host treats them as untrusted input from the server (Lecture 07). The four standard hints:

| Annotation | Meaning | How a host uses it |
|------------|---------|--------------------|
| `readOnlyHint` | The tool does not modify its environment | Skip the approval prompt; safe to call freely / in parallel |
| `destructiveHint` | The tool may perform irreversible updates (deletes, overwrites) | Require explicit confirmation; warn loudly |
| `idempotentHint` | Repeated calls with the same arguments have no additional effect | Safe to retry on timeout without duplicating side effects |
| `openWorldHint` | The tool touches external entities (the internet, a remote API) | Flag network egress; relevant to data-exfiltration risk |

There is also a `title` annotation — a human-friendly display name distinct from the machine `name`. The point of all four behavioral hints is the **approval UX**: a host that knows `delete_repo` carries `destructiveHint: true` can gate it behind a confirmation dialog, while letting a `readOnlyHint: true` tool like `search_code` run without interrupting the user. We return to *why a host must not blindly trust these hints* in Lecture 07 — a malicious server can lie — but the mechanism is here.

### Result content blocks and `isError`

A `tools/call` result is a list of **content blocks** plus an `isError` flag. Content blocks can be:

- **text** — the common case.
- **image** / **audio** — base64 data with a MIME type.
- **embedded or linked resources** — a tool result can hand back a resource (inline or by URI), bridging into the Resources world from §3.

The **`isError`** boolean distinguishes a *tool-level* error (the tool ran but failed — bad repo name, API 404) from a *protocol-level* error (a JSON-RPC error object, meaning the call itself was malformed). This split matters: a tool-level error with `isError: true` is fed *back to the model* so it can react and retry, whereas a protocol error is a fault in the host↔server plumbing. Here is a `tools/call` request and a successful result:

```json
{
  "jsonrpc": "2.0",
  "id": 42,
  "method": "tools/call",
  "params": {
    "name": "create_issue",
    "arguments": {
      "repo": "octo/hello",
      "title": "Build is flaky on ARM",
      "labels": ["bug", "ci"]
    }
  }
}
```

```json
{
  "jsonrpc": "2.0",
  "id": 42,
  "result": {
    "content": [
      { "type": "text", "text": "Created issue #128: Build is flaky on ARM" }
    ],
    "structuredContent": {
      "issue_number": 128,
      "url": "https://github.com/octo/hello/issues/128",
      "state": "open"
    },
    "isError": false
  }
}
```

A tool-level failure returns on the same shape with `isError: true` and an explanatory text block, so the model can read what went wrong and adjust.

---

## 3. Resources in depth

Resources are the **application-controlled**, read-only primitive: addressable data the server exposes, which the *host application* — not the model — chooses to pull into context. Files, database rows, API responses, logs, screenshots: anything the model might need to *read* but should not *act on*.

### Discovery and reading

- **`resources/list`** — the server enumerates its available resources, each with a `uri`, a `name`, optionally a `description` and `mimeType`.
- **`resources/read`** — the host requests the contents of one resource by URI, and gets back its data (text or binary).

### URIs and schemes

Every resource is identified by a **URI**. The scheme tells you what kind of thing it is:

- **`file://`** — local filesystem paths.
- **`https://`** — web resources.
- **custom schemes** — a server can mint its own, e.g. `postgres://`, `screen://`, `git://`. The scheme is server-defined; the host just treats the URI as an opaque address it can pass to `resources/read`.

### Resource templates (RFC 6570)

A server rarely wants to enumerate *every* row in a database as a static resource. Instead it exposes a **resource template** — a parameterized URI using **RFC 6570 URI Template** syntax — and the host fills in the variables to construct a concrete URI to read. A template entry looks like this:

```json
{
  "resourceTemplates": [
    {
      "uriTemplate": "postgres://db/users/{user_id}",
      "name": "User record",
      "description": "A single user row by id.",
      "mimeType": "application/json"
    },
    {
      "uriTemplate": "file:///logs/{date}/{service}.log",
      "name": "Service log",
      "description": "Daily log file for a given service.",
      "mimeType": "text/plain"
    }
  ]
}
```

The `{user_id}`, `{date}`, `{service}` placeholders are RFC 6570 expansions. The host substitutes values — `postgres://db/users/42` — and calls `resources/read` on the result. Templates are how a server says "I can give you *any* user, not a fixed list."

### Subscriptions and `notifications/resources/updated`

Resources can change. A log grows; a row is edited. MCP supports **subscriptions** so a host can keep its attached context fresh without polling:

- **`resources/subscribe`** — the host subscribes to a specific resource URI.
- **`notifications/resources/updated`** — the server pushes this notification when that resource changes, and the host can re-`read` it.

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/resources/updated",
  "params": {
    "uri": "file:///logs/2026-06-24/api.log"
  }
}
```

(There is also a `notifications/resources/list_changed` for when the *set* of resources changes, distinct from the contents of one resource.)

### The crucial point: resources are application-controlled, not silently injected

Here is the idea most function-calling veterans get wrong. **A server offering a resource does not push it into the model's context.** Offering ≠ injecting. The host *decides* what the model sees. The application might surface resources as `@`-mentions for the user to pick, as a file-tree the user browses, or it might auto-include some based on its own policy — but **that decision belongs to the host, not the server and not the model.**

This is the whole reason Resources are a separate primitive from Tools. A tool is the model reaching out and *doing* something on its own initiative. A resource is the application deciding *this data is relevant, show it to the model.* Read-only, human/app-curated, never a surprise. If your design needs the model to *autonomously fetch* arbitrary data, that is a **tool** (`fetch_url`), not a resource — because now the model is the one deciding, and the trust boundary from §1 moves accordingly.

---

## 4. Prompts in depth

Prompts are the **user-controlled** primitive: reusable, templated message sequences that the *human* invokes — typically as a slash command or a menu item. Where a tool is "the model can do X" and a resource is "the app can show X," a prompt is "the user can ask for X, pre-packaged."

### Discovery and retrieval

- **`prompts/list`** — the server enumerates its prompts, each with a `name`, a `description`, and a list of typed `arguments`.
- **`prompts/get`** — the host requests a specific prompt by name, passing values for its arguments, and gets back a **list of messages** ready to seed the conversation.

### Typed arguments

A prompt declares its parameters as `arguments`, each with a `name`, a `description`, and a `required` flag. The host renders these as fields the user fills in (or auto-supplies). A prompt definition from `prompts/list`:

```json
{
  "name": "code_review",
  "title": "Review a pull request",
  "description": "Generate a focused review of a code change.",
  "arguments": [
    { "name": "language", "description": "Programming language", "required": true },
    { "name": "diff", "description": "The unified diff to review", "required": true },
    { "name": "focus", "description": "e.g. security, performance", "required": false }
  ]
}
```

### Returning a sequence of messages

`prompts/get` does not return a single string — it returns a **list of messages** (each with a `role` and `content`), which the host injects as the start of the conversation. This lets a prompt set up a multi-turn frame (a system-style instruction plus a user turn, for instance), not just a one-liner. A `prompts/get` request and its result:

```json
{
  "jsonrpc": "2.0",
  "id": 7,
  "method": "prompts/get",
  "params": {
    "name": "code_review",
    "arguments": {
      "language": "Python",
      "focus": "security"
    }
  }
}
```

```json
{
  "jsonrpc": "2.0",
  "id": 7,
  "result": {
    "description": "A focused code review prompt",
    "messages": [
      {
        "role": "user",
        "content": {
          "type": "text",
          "text": "You are reviewing Python code. Focus on security. Flag injection risks, unsafe deserialization, and secrets in code. Be specific and cite line numbers."
        }
      }
    ]
  }
}
```

The user picked `code_review` from a menu; the host called `prompts/get`; the returned messages become the opening of the conversation. Note the **user-controlled** nature: nothing happened until the human chose this prompt. That is the defining property, and it is why prompts surface as commands, never as something the model or app fires on its own.

---

## 5. How a host surfaces each primitive

The trichotomy from §1 is not abstract — it dictates exactly how each primitive shows up in a real host UI. Three control models, three surfaces:

| Primitive | Control | How the host surfaces it | Maps to |
|-----------|---------|--------------------------|---------|
| **Tools** | model-controlled | Offered to the model as **callable functions** in its tool list; the model emits a call, the host (optionally gated by annotations) executes it | function calling — see [Tool Use & Function Calling](../Lectures/Lecture-08.md) |
| **Resources** | application-controlled | Offered to the user/app as **attachable context** — `@`-mentions, file/resource pickers, context panels; the app decides what to include | read-only context attachment |
| **Prompts** | user-controlled | Offered to the user as **commands** — slash-menu entries, buttons, menu items the human clicks | UI affordances / command palette |

The through-line: **tools go to the model, resources go to the application's context-assembly, prompts go to the user's command surface.** When you design an MCP server in [Lecture 05](Lecture-05.md), the first question for every capability is "which actor controls this?" — and the answer tells you which primitive to use and, therefore, how every compliant host will render it. Tools tie directly back to the function-calling machinery in [Tool Use & Function Calling](../Lectures/Lecture-08.md); MCP's contribution is making that machinery a *protocol* so the model can reach tools from any server, alongside the application's curated resources and the user's prompt commands.

With the three forward primitives in hand, [Lecture 04](Lecture-04.md) turns the connection around: Sampling, Roots, and Elicitation — the things a *server* asks of the *host*, the direction that makes MCP genuinely bidirectional.

---

## Current as of

- **Date:** June 2026
- **Spec revision:** MCP **2025-11-25** (stable). The control trichotomy (Tools = model-controlled, Resources = application-controlled, Prompts = user-controlled), the method names (`tools/list`, `tools/call`, `resources/list`, `resources/read`, `resources/subscribe`, `prompts/list`, `prompts/get`), `outputSchema`/structured content, tool annotations (`readOnlyHint`, `destructiveHint`, `idempotentHint`, `openWorldHint`), RFC 6570 resource templates, and `notifications/resources/updated` are all current as of this revision. Verify primitive shapes against the spec at [modelcontextprotocol.io](https://modelcontextprotocol.io) and your installed SDK before relying on exact field names.
