# Lecture 04 - Reverse Primitives: Sampling, Roots, Elicitation

**Collection:** [MCP for AI Agents](README.md) | **Previous:** [← Lecture 03](Lecture-03.md) | **Next:** [Lecture 05](Lecture-05.md)

---

Lecture 03 covered the three primitives that point *outward* from a server: Tools, Resources, and Prompts. In all three, the client (or its model) reaches into the server and pulls something out — calls a tool, reads a resource, fetches a prompt. That is the direction everyone pictures when they hear "MCP," and it is the direction most of the protocol runs. But it is not the whole protocol, and a server built only on those three is a *passive* server: it can answer, but it cannot reason, it cannot ask, and it cannot be told where its sandbox ends.

The three primitives in this lecture run the other way. **Sampling**, **Roots**, and **Elicitation** are *reverse* primitives — the server initiates a request *of the client/host*, and the host answers. This inversion is the thing that makes MCP genuinely *bidirectional* rather than a fancy RPC wrapper for function calling. A server that can call back to the host can borrow the host's LLM to reason mid-task (Sampling), it can be handed a least-privilege boundary it must stay inside (Roots), and it can pause and ask the human a structured question before it does something irreversible (Elicitation). None of those require the server to own an API key, ship a UI, or guess at the user's filesystem layout — the host already has all three, so the server *delegates* to it.

The reason this matters for agentic systems is control. A passive tool server is something the agent *uses*; a server with reverse primitives is something the agent *collaborates with* — and crucially, every reverse call is structured so the host keeps the final say. The host chooses the model and pays for it; the host draws the filesystem boundary; the host owns the user's attention. Get this inversion right and you understand why MCP is the integration layer for agents and not just a tool bus.

---

## Learning objectives

By the end of this lecture you should be able to:

* Explain why MCP needs **server→client** requests, and contrast them precisely with the **client→server** primitives of Lecture 03.
* Construct a **`sampling/createMessage`** request with `messages`, `modelPreferences`, `systemPrompt`, and `maxTokens` — and name the three control points the host *never* surrenders.
* Use **`roots/list`** as a least-privilege *boundary* (not a discovery mechanism) and handle **`notifications/roots/list_changed`**.
* Issue an **`elicitation/create`** request with a JSON Schema, and state the **SEP-2260** rule that governs *when* a server is allowed to elicit.
* **Capability-gate** all three and degrade gracefully when the client did not advertise the capability in `initialize`.

---

## 1. The inversion: which way does each primitive point?

The six MCP primitives split cleanly by *who initiates the request*. The three from Lecture 03 are server→client *offerings* that the client pulls from; the three here are server-initiated *requests* that the client/host services.

| Primitive | Direction | Initiated by | Serviced by | Method | Controlled by |
|---|---|---|---|---|---|
| **Tools** | server → client | client/model | server | `tools/call` | model |
| **Resources** | server → client | client/app | server | `resources/read` | application |
| **Prompts** | server → client | client/user | server | `prompts/get` | user |
| **Sampling** | **client ← server** | **server** | **host** | `sampling/createMessage` | host (model, keys, approval) |
| **Roots** | **client ← server** | **server** | **client** | `roots/list` | client (grants the boundary) |
| **Elicitation** | **client ← server** | **server** | **user** | `elicitation/create` | user (answers / declines) |

The first three flow one way; the last three flow back. That bidirectionality is the whole point:

```text
   CLIENT / HOST                           SERVER
   (owns model, keys,         tools/call ───────►  (owns tool logic,
    UI, filesystem)           resources/read ───►   data, business rules)
                              prompts/get ──────►
        ┌───────────────────────────────────────────────┐
        │           the inversion (this lecture)         │
        └───────────────────────────────────────────────┘
   (runs the LLM)    ◄─── sampling/createMessage   (wants to reason)
   (grants a sandbox)◄─── roots/list               (must stay in bounds)
   (owns the human)  ◄─── elicitation/create       (needs user input)
```

Why do reverse calls exist at all? Three needs that a passive server cannot meet on its own:

* **It wants to reason** — summarize a document, classify a record, draft text — but it must not ship its own model or API key. So it *borrows the host's LLM* (Sampling).
* **It must stay inside a sandbox** — touch only the directories the user actually authorized — but it cannot know that layout in advance. So the *client hands it a boundary* (Roots).
* **It needs information only the human has** — a missing argument, a yes/no on a destructive step — but it owns no UI. So it *asks through the host* (Elicitation).

In every case the capability the server lacks (a model, a filesystem boundary, a human) is something the **host already has**, so the server delegates instead of duplicating. That is the design principle behind all three reverse primitives.

---

## 2. Sampling — the server borrows the host's LLM

**Sampling** (`sampling/createMessage`) lets a server ask the host to run an LLM completion *on the server's behalf*. The server constructs the conversation it wants the model to see; the host runs it against whatever model it controls and returns the completion. This is what lets a server be *agentic* — it can nest LLM reasoning inside a tool call (call a tool → sample to interpret the result → call another tool) without ever holding model credentials.

A sampling request carries `messages`, optional `modelPreferences`, an optional `systemPrompt`, and a `maxTokens` ceiling:

```json
{
  "jsonrpc": "2.0",
  "id": 42,
  "method": "sampling/createMessage",
  "params": {
    "messages": [
      {
        "role": "user",
        "content": {
          "type": "text",
          "text": "Classify the sentiment of this review as positive, negative, or neutral, and return only the single word:\n\n\"Shipping was slow but the product is exactly what I needed.\""
        }
      }
    ],
    "modelPreferences": {
      "hints": [{ "name": "claude-sonnet" }],
      "costPriority": 0.6,
      "speedPriority": 0.7,
      "intelligencePriority": 0.3
    },
    "systemPrompt": "You are a precise text classifier. Output exactly one word.",
    "maxTokens": 16
  }
}
```

Two things about `modelPreferences` are worth pinning down. The three priorities — `costPriority`, `speedPriority`, `intelligencePriority` — are *normalized hints in the range 0–1*, not hard requirements; they tell the host how to trade money against latency against capability. The `hints` are *advisory model-name substrings*: a server can suggest `claude-sonnet`, but the host is free to map that onto whatever it actually has (a host with only Gemini access might satisfy a `sonnet` hint with a comparable Gemini model). The server expresses a *preference*; the host makes the *decision*.

That last sentence is the heart of Sampling. The host keeps three control points and **never** surrenders them to the server:

| Control point | Why the host keeps it |
|---|---|
| **Model selection** | The host knows which models it has, their cost, and their context limits; the server only hints. |
| **Keys & cost** | The completion runs on the *host's* credentials and the host's bill — the server never sees an API key. |
| **Human-approval gate** | The host should surface *what the server is asking the model to do* and let the user inspect, edit, or deny it before the call runs — and let the user review the result before it is returned. |

Sampling is therefore **human-in-the-loop by design**, not by convention. The recommended host UX is a two-sided gate: the user can see and approve the outbound prompt the server constructed, and can see the model's response before it is handed back to the server. A host that fires server-supplied prompts at an LLM with no visibility has built a confused-deputy machine — exactly the failure mode Lecture 07 dissects in the context of approval gates for `roots` and `sampling`.

**Canonical use case.** A "meeting-notes" server receives a raw transcript via a tool call and needs a three-bullet summary. It does *not* embed an Anthropic or OpenAI key and call out itself — that would put credentials, billing, and model choice inside a third-party server. Instead it issues a `sampling/createMessage` with the transcript in `messages`, a `costPriority` leaning cheap, and a tight `maxTokens`. The host runs it on the user's own model under the user's own approval, and the server gets its summary back without owning any of the machinery.

---

## 3. Roots — the client grants the boundary

**Roots** (`roots/list`) is the primitive by which the *client* tells the *server* which filesystem paths or URIs it is permitted to operate within. The mental model that trips people up is treating roots as a *discovery* mechanism ("here is where the interesting files are, go find them"). It is not. Roots is a **least-privilege boundary**: the client is granting a scope, and the server's contract is to *never act outside it*.

The flow is server-initiated — the server asks the client for the current roots:

```json
{
  "jsonrpc": "2.0",
  "id": 7,
  "method": "roots/list"
}
```

```json
{
  "jsonrpc": "2.0",
  "id": 7,
  "result": {
    "roots": [
      { "uri": "file:///home/dev/project-alpha", "name": "Project Alpha" },
      { "uri": "file:///home/dev/shared/specs",  "name": "Shared specs" }
    ]
  }
}
```

Each root is a URI (commonly a `file://` path, but the spec allows other URI schemes) with an optional human-readable `name`. The server should read this list, scope all of its file access to those subtrees, and refuse anything that resolves outside them — a request to read `file:///etc/passwd` against the roots above must be rejected by the *server*, not silently attempted and left to the OS to deny.

Roots are not static. When the user opens a new folder, closes a workspace, or changes what the agent may touch, the client emits a notification and the server re-fetches:

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/roots/list_changed"
}
```

A correct server treats `notifications/roots/list_changed` as a signal to call `roots/list` again and *re-derive its allowed set* — it must not cache the original roots forever. The security framing is direct and is revisited in Lecture 07: roots are a boundary the client *grants*, and a server that operates outside its granted roots is either buggy or hostile. Treat path-escape attempts (`../`, symlinks out of the tree, absolute paths outside a root) as something to validate and reject, exactly as you would untrusted input — because across a trust boundary, that is what they are.

---

## 4. Elicitation — the server asks the user, mid-task

**Elicitation** (`elicitation/create`) lets a server request *structured input from the user* in the middle of handling a request. Where Sampling borrows the host's *model* and Roots borrows the host's *filesystem boundary*, Elicitation borrows the host's *human* — it is the protocol's standard way for a server to say "I need one more thing from the person before I can finish," without the server owning any UI of its own.

The request describes the fields it wants with a **JSON Schema**, so the host can render a proper form and validate the answer:

```json
{
  "jsonrpc": "2.0",
  "id": 19,
  "method": "elicitation/create",
  "params": {
    "message": "Deploying to production. Confirm the target and window before I proceed.",
    "requestedSchema": {
      "type": "object",
      "properties": {
        "environment": {
          "type": "string",
          "enum": ["staging", "production"],
          "description": "Target environment"
        },
        "confirm": {
          "type": "boolean",
          "description": "I understand this is a production deploy"
        }
      },
      "required": ["environment", "confirm"]
    }
  }
}
```

The user fills in the form and the host returns a structured response. The `action` field tells the server whether the user accepted, declined, or dismissed the prompt — and the server must handle all three, not assume an answer arrived:

```json
{
  "jsonrpc": "2.0",
  "id": 19,
  "result": {
    "action": "accept",
    "content": {
      "environment": "production",
      "confirm": true
    }
  }
}
```

**Canonical use cases** for elicitation are: a **missing required parameter** the agent never supplied (ask for it rather than failing); a **confirmation before a destructive or irreversible action** (delete, deploy, send money); and **disambiguation** when an argument matched more than one thing ("did you mean repo `api` or `api-gateway`?").

Now the rule that makes elicitation safe — **SEP-2260**. A server may issue an elicitation request **only while it is actively handling a client request**. Every server→client request must be *associated with* an in-flight client→server request; the server cannot send an elicitation out of nowhere, between calls, or on a background timer. The practical guarantee is that a user is **never prompted out of the blue**: every elicitation traces back to something the user or their agent *just started*. A prompt that appears with no action behind it is, almost by definition, an attempt to manipulate the user, so the spec forbids the shape entirely.

This is also a *required* rule now, not a polite suggestion. Earlier revisions recommended associating server requests with a client request; recent spec revisions made it mandatory for elicitation — out-of-band elicitation is non-conformant. (SEP-2260 generalizes this to server→client requests broadly; the 2026-07-28 RC then reworks exactly *how* that association is carried in a stateless core — see §5 and the "Current as of" note.) When you design a server, the test is simple: if you cannot point at the client request that an elicitation belongs to, you are not allowed to send it.

---

## 5. Capability gating & graceful degradation

All three reverse primitives are **capability-gated**. The server cannot assume the client supports them — it must check what the client advertised during `initialize` and adapt. A client that can service these declares them under `capabilities`:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": {
    "protocolVersion": "2025-11-25",
    "capabilities": {
      "sampling": {},
      "roots": { "listChanged": true },
      "elicitation": {}
    },
    "clientInfo": { "name": "ExampleHost", "version": "3.1.0" }
  }
}
```

If a capability is **absent**, the rule is firm: the server **must not** send that request, and it **must degrade gracefully** — never hang waiting for a reply that the client will never service, and never hard-crash. The art is in the fallback, and each primitive has a sensible one:

| Capability missing | Wrong behavior | Graceful degradation |
|---|---|---|
| **`sampling`** | Embed your own API key and call an LLM directly | Return the raw data and let the *host's* model do the reasoning on the other side |
| **`roots`** | Assume access to the whole filesystem | Fall back to a configured/default working directory, or operate read-only on what you were explicitly given |
| **`elicitation`** | Block forever, or guess the missing value | Fail fast with a clear, actionable error naming the missing argument |

In code, the pattern is the same shape every time — check the negotiated capability before the reverse call, and branch:

```text
if client.supports("elicitation"):
    answer = elicitation_create(message="Confirm production deploy?", schema=...)
    if answer.action != "accept":
        return cancel("User declined the deploy.")
else:
    # capability not granted — never elicit; fail with a usable message
    raise ToolError("Missing required argument 'environment'. "
                    "Re-invoke this tool with environment set to 'staging' or 'production'.")
```

The graceful-degradation contract is what lets the same server run against a rich host (a full IDE that can sample, scope roots, and pop dialogs) *and* a minimal one (a headless client that supports none of the three) without two code paths and without ever leaving a request dangling. A reverse primitive is a request to the host for a capability the host *might* offer — write every one of them as "use it if it's there, work without it if it isn't."

---

## Current as of

* **Date:** June 2026.
* **Spec revision:** pinned to the **2025-11-25** stable revision — the current source of truth for Sampling (`sampling/createMessage`), Roots (`roots/list`, `notifications/roots/list_changed`), and Elicitation (`elicitation/create`), and for the capability-gating rules in `initialize`.
* **SEP-2260** (server→client requests must be associated with an in-flight client request — the rule that forbids out-of-band elicitation) is in force and is *required*, not merely recommended, in the current revision.
* **Watch:** the **2026-07-28 release candidate** reworks how mid-call server→client requests flow in a **stateless core**; the *contract* of these three primitives is stable, but the wire-level mechanics of associating a server request with its originating client request are exactly what that RC revisits. Re-verify the request-association details against the RC before relying on them.
