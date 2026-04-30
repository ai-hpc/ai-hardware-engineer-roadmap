# Lecture 22 - OpenClaw Case Study: App SDK Dogfooding and Typed Gateway RPCs

**Course:** [Agentic AI & GenAI](../Guide.md) | **Previous:** [Lecture 21](Lecture-21.md) | **Next:** [Lab 01](Lab-01-Research-Agent.md)

---

An agent runtime becomes a platform only when external applications can use it without knowing private internals.

That is the purpose of the OpenClaw App SDK.

The App SDK, published as `@openclaw/sdk`, is the public client API for applications that run **outside** the OpenClaw process:

- dashboards
- desktop clients
- mobile clients
- IDE extensions
- CI jobs
- admin tools
- integration tests
- companion apps such as OpenMeow

This lecture explains the current blueprint:

> make `@openclaw/sdk` usable by real external apps through typed Gateway RPCs, instead of forcing apps to scrape CLI output, transcripts, or runtime internals

---

## Learning objectives

By the end of this lecture, you should be able to:

1. Explain the difference between the App SDK and the Plugin SDK.
2. Describe the App SDK happy path for real external applications.
3. Understand why Gateway WebSocket RPC is the correct platform boundary.
4. Explain how agents, sessions, runs, artifacts, tools, environments, and tasks fit into the app-facing architecture.
5. Understand what the current SDK supports today versus what remains explicit future surface.
6. Design a narrow typed RPC surface without mixing unrelated responsibilities.
7. Explain how dogfooding with a real app stabilizes SDK contracts.
8. Build a testing strategy around normalized events, waits, cancellation, and unsupported feature errors.

---

## 1. Big picture

The architecture is:

```text
External app / OpenMeow
        |
        v
@openclaw/sdk
        |
        v
Gateway WebSocket RPC
        |
        +-- agents / sessions / runs     # happy-path app control
        +-- artifacts.*                  # rich outputs: files/images/logs/etc.
        +-- tools.invoke                 # controlled tool execution
        +-- environments.*               # discover where work can run
        +-- task ledger                  # durable app-visible run/task state
```

This is the important shift:

```text
Before:
  App knows internal Gateway/session/runtime details.

After:
  App uses typed SDK methods backed by discoverable Gateway RPCs.
```

That is how OpenClaw moves from "working internal system" to "external app platform."

---

## 2. App SDK versus Plugin SDK

OpenClaw has two different extension surfaces.

Do not mix them.

| SDK | Where Code Runs | Use It For |
|---|---|---|
| App SDK | Outside OpenClaw | External apps, dashboards, scripts, CI jobs, IDE clients |
| Plugin SDK | Inside OpenClaw | Providers, channels, hooks, tools, runtime plugins |

The App SDK connects to a Gateway.

The Plugin SDK extends the Gateway/runtime from inside.

Wrong mental model:

```text
"SDK is SDK; app code and plugin code can share the same assumptions."
```

Correct mental model:

```text
App SDK = remote client contract.
Plugin SDK = in-process extension contract.
```

This separation matters for auth, scopes, error handling, lifecycle, and compatibility.

---

## 3. What ships in `@openclaw/sdk`

The main entry is:

```ts
OpenClaw
```

It owns:

- transport
- connection
- request/response calls
- event handling
- high-level resource helpers

Basic connection example:

```ts
import { OpenClaw } from "@openclaw/sdk";

const oc = new OpenClaw({
  url: "ws://127.0.0.1:14565",
  token: process.env.OPENCLAW_GATEWAY_TOKEN,
  requestTimeoutMs: 30_000,
});

await oc.connect();
```

The default transport is:

```ts
GatewayClientTransport
```

Tests can pass a custom transport implementing the SDK transport interface, so integration logic can be tested without a real WebSocket server.

---

## 4. The current high-level SDK helpers

The SDK exposes resource helpers.

| Helper | Purpose |
|---|---|
| `oc.agents` | List agents, get agent handles, start runs from agents |
| `oc.runs` / `Run` | Create, get, wait, cancel, and stream runs |
| `oc.sessions` / `Session` | Create sessions, send messages, patch, compact, abort |
| `oc.models` | List models and inspect model auth status |
| `oc.tools` | List tool catalog and effective tools |
| `oc.approvals` | List and resolve approval requests |
| `oc.rawEvents()` | Inspect raw Gateway frames for advanced cases |

The SDK also exports types such as:

- `AgentRunParams`
- `RunResult`
- `RunStatus`
- `OpenClawEvent`
- related RPC and selection types

The design goal is that app authors use these helpers instead of hand-writing Gateway frames.

---

## 5. The SDK happy path

The happy path is the minimum app flow that must be boringly reliable.

```text
Connect
  -> Discover
  -> Create or resume session
  -> Start run
  -> Stream events
  -> Wait for result
  -> Cancel if needed
  -> Surface approvals
```

This path matters because most external apps need exactly this loop:

```text
User clicks "Run"
  -> app sends task
  -> assistant streams output
  -> tools emit progress
  -> approvals may be requested
  -> app shows final result
```

If this path is unstable, every external app becomes a pile of special cases.

---

## 6. Running an agent

A typical high-level app flow:

```ts
const agent = await oc.agents.get("default");

const run = await agent.run({
  message: "Summarize the current project status.",
  sessionKey: "main",
  model: "openai/gpt-5.5",
  timeoutMs: 30_000,
});

for await (const event of run.events()) {
  if (event.type === "assistant.delta") {
    process.stdout.write(String(event.data));
  }

  if (event.type === "run.completed") {
    break;
  }
}

const result = await run.wait();
```

Important SDK behavior:

- provider-qualified model refs such as `openai/gpt-5.5` are split into provider and model overrides before being sent to the Gateway
- SDK `timeoutMs` is milliseconds
- Gateway timeout values may be sent as seconds
- `Run.events()` filters events to one run
- `Run.events()` can replay already-seen events for fast runs
- `Run.wait()` maps Gateway lifecycle outcomes into stable SDK result shapes

The app does not need to know the internal agent loop implementation.

---

## 7. Sessions

Sessions are durable transcript holders.

They give apps stable context and session-affine behavior.

Example:

```ts
const session = await oc.sessions.create({
  agentId: "default",
  label: "Hardware debug session",
});

const run = await session.send({
  message: "Review the latest UART bring-up notes.",
});
```

Session handles can support operations such as:

- send
- abort
- patch
- compact
- inspect metadata

Use sessions when the app wants durable conversation state.

Use isolated runs when the app wants clean one-shot work.

---

## 8. Event streaming and normalization

External apps should not consume raw Gateway internals directly.

The SDK normalizes Gateway events into a stable envelope:

```ts
type OpenClawEvent = {
  version: 1;
  id: string;
  ts: number;
  type: OpenClawEventType;
  runId?: string;
  sessionId?: string;
  sessionKey?: string;
  taskId?: string;
  agentId?: string;
  data: unknown;
  raw?: GatewayEvent;
};
```

Common mapped event types include:

- `run.started`
- `run.completed`
- `run.failed`
- `run.cancelled`
- `run.timed_out`
- `assistant.delta`
- `assistant.message`
- `thinking.delta`
- `tool.call.*`
- `approval.requested`
- `approval.resolved`
- `session.created`
- `session.updated`
- `session.compacted`
- `task.updated`
- `artifact.updated`
- `raw`

The `raw` escape hatch is useful for advanced clients, but normal apps should prefer stable SDK event types.

---

## 9. Why event leakage is dangerous

If raw chat or runtime events leak through `run.events()`, the app becomes coupled to private implementation details.

Bad pattern:

```text
UI reducer handles internal provider chunks, chat frames, and lifecycle frames directly.
```

Good pattern:

```text
Gateway event
  -> SDK normalizer
  -> stable OpenClawEvent
  -> app adapter
  -> UI reducer
```

If internal events leak, clients break when the runtime changes.

The SDK should be the compatibility layer.

---

## 10. Wait and cancellation semantics

Run lifecycle must be consistent across three surfaces:

```text
Run.cancel()
Run.events()
Run.wait()
```

The SDK should avoid contradictory states.

Bad state:

```text
cancel() returns cancelled
events emit run.completed
wait() returns timed_out
```

Good state:

```text
cancel requested
events eventually emit run.cancelled
wait() returns cancelled
```

Important nuance:

`Run.cancel()` is a request to stop work.

The true terminal state is confirmed by the runtime.

`Run.wait()` should return normalized statuses such as:

- `completed`
- `failed`
- `cancelled`
- `timed_out`
- `accepted`

If the wait deadline expires while the run is still active, the SDK should return an accepted or still-active result rather than pretending the run itself failed.

---

## 11. Current supported versus future SDK surface

A mature SDK should not pretend missing Gateway RPCs exist.

The current App SDK approach is explicit:

| Namespace | Current Shape |
|---|---|
| `oc.agents` | App-facing helper surface for agents and agent handles |
| `oc.runs` | Run creation, wait, cancel, stream |
| `oc.sessions` | Durable session management and sending |
| `oc.models` | Model listing and auth status |
| `oc.tools` | Tool catalog and effective tools |
| `oc.approvals` | Approval listing and resolution |
| `oc.tasks` | Explicitly unsupported until Gateway APIs exist |
| `oc.artifacts` | Explicitly unsupported until artifact RPCs exist |
| `oc.environments` | Explicitly unsupported until environment RPCs exist |
| `oc.tools.invoke` | Explicitly unsupported until Gateway tool invocation exists |

This is good API design.

It prevents silent fallback to unsafe defaults.

If a caller passes future-only fields such as workspace, runtime, environment, or approval parameters before the Gateway supports them, the SDK should throw before sending the request.

That is safer than pretending the setting worked.

---

## 12. Blueprint: typed Gateway RPCs

Every app-facing capability should become a typed Gateway RPC.

The implementation pattern:

```text
1. Protocol schema
   src/gateway/protocol/schema/*.ts

2. Protocol exports + validators
   src/gateway/protocol/index.ts
   src/gateway/protocol/schema/protocol-schemas.ts

3. Gateway method handler
   src/gateway/server-methods/*.ts

4. Method discovery
   src/gateway/server-methods-list.ts

5. Scope gate
   src/gateway/method-scopes.ts

6. SDK wrapper
   packages/sdk/src/client.ts
   packages/sdk/src/types.ts
   packages/sdk/src/index.ts

7. Generated native protocol models
   apps/macos/Sources/OpenClawProtocol/GatewayModels.swift
   apps/shared/OpenClawKit/Sources/OpenClawProtocol/GatewayModels.swift

8. Docs + changelog
   docs/concepts/openclaw-sdk.md
   docs/gateway/protocol.md
   CHANGELOG.md
```

The exact file names may evolve, but the architecture rule is stable:

> schema first, server handler second, discovery third, SDK wrapper fourth, generated native models and docs before calling it public

---

## 13. Gateway RPC framing

Gateway RPC uses WebSocket JSON frames.

Request:

```json
{
  "type": "req",
  "id": "req-1",
  "method": "runs.create",
  "params": {}
}
```

Response:

```json
{
  "type": "res",
  "id": "req-1",
  "ok": true,
  "payload": {}
}
```

Event:

```json
{
  "type": "event",
  "family": "runs",
  "name": "run.delta",
  "payload": {}
}
```

This gives SDKs:

- request-response correlation
- streaming events
- typed errors
- feature discovery
- transport limits
- auth and scope enforcement
- compatibility checks

---

## 14. Capability boundary rule

Keep RPC families narrow.

```text
environments.* = read-only discovery
artifacts.*    = read-only output access/download
tools.invoke   = controlled execution with policy/approval
tasks.*        = durable task state
sessions/runs  = core app execution path
```

Do not bundle these into one broad "SDK platform" method.

Narrow RPC surfaces are easier to:

- review
- test
- secure
- document
- version
- expose in native SDKs

This is how the SDK grows without becoming unstable.

---

## 15. Artifacts as app-visible outputs

Apps need richer outputs than text transcripts.

Artifacts represent:

- generated files
- images
- logs
- downloaded documents
- reports
- tool output bundles

A clean artifact API surface looks like:

```text
artifacts.list
artifacts.get
artifacts.download
```

Why this matters:

```text
Without artifacts:
  App parses transcript text to find output files.

With artifacts:
  App asks the Gateway for structured output metadata and downloads.
```

Artifact APIs should be:

- typed
- read-only unless mutation is explicitly needed
- scope-gated
- payload-limit aware
- discoverable in `hello-ok.features.methods`
- backed by SDK wrappers
- covered by tests

Large artifact content should not be shoved blindly into WebSocket frames.

Use metadata, download handles, or chunked behavior where appropriate.

---

## 16. Environment discovery

Apps need to know where work can run.

Environment discovery is read-only at first:

```text
environments.list
environments.status
```

It can expose:

- Gateway-local runtime candidates
- node candidates
- capabilities
- health
- availability
- labels

It should not initially do:

- provisioning
- create/delete
- runtime selection
- remote mutation

That boundary is intentional.

Discovery is safer than control.

The app can show users where work can run without being allowed to create or destroy environments.

---

## 17. Controlled tool invocation

Direct tool invocation is powerful and risky.

A future SDK-facing method could mirror the existing HTTP tool invoke behavior as:

```text
tools.invoke
```

Example:

```json
{
  "type": "req",
  "id": "req-1",
  "method": "tools.invoke",
  "params": {
    "tool": "sessions_list",
    "action": "json",
    "args": {},
    "sessionKey": "main"
  }
}
```

The important rule:

> SDK tool invocation must reuse the same Gateway auth, tool policy, deny-list, approval semantics, and owner/actor semantics as the existing server path

It must not become a shortcut around policy.

Tool invocation touches:

- tool allow/deny policy
- session scoping
- agent scoping
- approval states
- confirmation and refusal states
- audit logs
- security boundaries

That is why it is harder than read-only methods.

---

## 18. Task ledger

Event streams are transient.

Apps also need durable task state.

A task ledger API gives UIs a stable way to ask:

```text
What work exists?
What is running?
What completed?
What failed?
What was cancelled?
What artifacts belong to this work?
```

Potential surface:

```text
tasks.list
tasks.get
tasks.cancel
```

The key design point:

> event streams are for live updates; task ledger APIs are for durable app-visible state

Do not make apps reconstruct durable state only from historical event streams.

---

## 19. Discovery and feature negotiation

Gateway connections should advertise capabilities.

The SDK can inspect:

```text
hello-ok.features.methods
hello-ok.policy.maxPayload
hello-ok.policy.maxBufferedBytes
hello-ok.policy.tickIntervalMs
```

Then the app can decide:

- whether artifact APIs exist
- whether environment discovery exists
- whether tool invocation exists
- whether the payload is too large
- whether to show or hide UI features

This avoids hard-coded version assumptions.

Feature detection beats guessing.

---

## 20. Auth, scopes, and fail-closed events

SDK methods must be scope-gated.

Examples:

- `operator.read`
- `operator.write`
- `operator.admin`
- plugin-defined scopes

Server-side checks are mandatory.

The SDK is not a security boundary.

Events should also be gated by visibility.

The safe rule:

```text
If the client should not see a session, run, task, artifact, or approval, do not broadcast the event to that client.
```

For app developers this means:

- expect permission errors
- design UI around missing capabilities
- treat feature discovery as dynamic
- do not assume owner-level access

---

## 21. Idempotency

Side-effecting methods need idempotency keys.

Examples:

- start a run
- cancel a run
- approve a tool call
- invoke a tool
- create or mutate an artifact

Why?

Because real clients retry.

Networks fail.

Mobile clients reconnect.

Users double-click.

Without idempotency:

```text
one user action -> two runs
```

With idempotency:

```text
same request key -> same accepted operation
```

Idempotency is part of the SDK contract, not an optimization.

---

## 22. Dogfooding with OpenMeow

OpenMeow-style dogfooding is valuable because it forces the SDK to behave like a real product dependency.

The dogfood client should validate:

- connection setup
- feature discovery
- agent discovery
- session creation
- run start
- event streaming
- wait behavior
- cancellation
- approval handling
- artifact display
- unsupported feature errors

The app should not call private Gateway internals.

It should use the same SDK surface an external developer would use.

If the app needs a workaround, the SDK contract probably needs work.

---

## 23. Testing strategy

A strong SDK test harness uses fixtures.

Test these paths:

- connect and discover
- list agents
- create a session
- start a run
- stream assistant deltas
- stream tool events
- approval requested and resolved
- run completed
- run failed
- run cancelled
- wait deadline expires while run remains active
- raw event normalization
- unknown event family
- unsupported `oc.artifacts.*`
- unsupported `oc.environments.*`
- unsupported `oc.tasks.*`
- unsupported `oc.tools.invoke`

The goal is not just correctness.

The goal is contract stability.

When the Gateway evolves, the SDK fixtures tell you whether external apps will break.

---

## 24. Practical app guidance

For external apps:

- use `Run.events()` for progress instead of polling
- use `Run.wait()` for final lifecycle result
- use sessions for durable transcripts
- use raw events only for advanced diagnostics
- feature-detect Gateway methods before showing UI
- treat unsupported SDK namespaces as intentional
- use a custom transport in tests
- do not parse transcripts to find files once artifact APIs exist
- do not assume direct tool invocation is available
- do not mix App SDK and Plugin SDK assumptions

For SDK implementers:

- keep namespaces narrow
- fail loudly on unsupported future fields
- generate types from protocol schemas
- keep auth and scope checks server-side
- normalize events before exposing them
- make cancellation semantics deterministic
- add fixtures before expanding surface area

---

## 25. Design exercise

Design a small OpenClaw dashboard using only the App SDK.

The dashboard must:

- connect to a Gateway
- list agents
- list models
- create a session
- start a run
- stream assistant and tool events
- show approval prompts
- wait for final state
- cancel a run
- show artifacts if the Gateway supports artifact APIs
- hide artifact UI if the Gateway does not support artifact APIs

Answer:

1. Which SDK namespaces do you need today?
2. Which future namespaces should be feature-detected?
3. Which operations need idempotency keys?
4. What events update the UI state reducer?
5. What happens if `Run.wait()` returns accepted because the wait deadline expired?
6. How do you test the app without a real Gateway?
7. What must stay in the SDK adapter instead of the UI component?

---

## Key takeaways

- `@openclaw/sdk` is the app-facing contract for code outside OpenClaw.
- The Plugin SDK is a separate in-process extension contract.
- Real external apps should use typed Gateway RPCs, not CLI output or runtime internals.
- The happy path is connect, discover, session, run, stream, wait, cancel, and approvals.
- Current SDK helpers cover agents, runs, sessions, models, tools catalog, approvals, raw events, and event normalization.
- Future app-facing surfaces should be narrow: `artifacts.*`, `environments.*`, `tools.invoke`, and `tasks.*`.
- Unsupported future surfaces should throw explicit errors rather than pretending to work.
- Dogfooding with OpenMeow-style clients is how the SDK contract becomes stable enough for external apps.

---

## References

- OpenClaw App SDK: [https://openclaw.knidal.com/openclaw-app-sdk](https://openclaw.knidal.com/openclaw-app-sdk)
- OpenClaw Gateway protocol: [https://openclaw.knidal.com/gateway-protocol](https://openclaw.knidal.com/gateway-protocol)
- OpenClaw RPC adapters: [https://openclaw.knidal.com/rpc-adapters](https://openclaw.knidal.com/rpc-adapters)
- OpenClaw Tools Invoke API: [https://openclaw.knidal.com/tools-invoke-api](https://openclaw.knidal.com/tools-invoke-api)
- Case-study source repo: [OpenClaw](https://github.com/openclaw/openclaw)

---

*Next: [Lab 01 - Research Agent](Lab-01-Research-Agent.md)*
