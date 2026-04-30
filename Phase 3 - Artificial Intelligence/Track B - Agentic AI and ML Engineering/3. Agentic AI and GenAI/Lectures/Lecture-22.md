# Lecture 22 - OpenClaw Case Study: SDK Dogfooding and Gateway RPC Contracts

**Track B - Agentic AI & GenAI** | [<- Lecture 21](Lecture-21.md) | [Next -> Lab 01](Lab-01-Research-Agent.md)

---

A real agent platform is not finished when the internal app works.

It becomes a platform when external developers can build against a stable contract.

This lecture uses the OpenClaw SDK dogfooding work as the case study.

The core idea:

> dogfooding an SDK means using a real client application to force the runtime contract to become stable, typed, discoverable, and testable

In OpenClaw's case, OpenMeow is useful because it behaves like a real app:

- it connects to the Gateway
- discovers agents and methods
- creates or resumes sessions
- starts runs
- streams events
- waits for results
- cancels runs
- handles approvals
- maps SDK events into UI state

That path exposes the bugs that unit tests often miss.

---

## Learning objectives

By the end of this lecture, you should be able to:

1. Explain why SDK dogfooding is a platform-stabilization milestone.
2. Define the OpenClaw SDK happy path.
3. Understand why event leakage breaks external clients.
4. Explain cancel, stream terminal, and wait-result consistency.
5. Design a Gateway RPC method that is discoverable, typed, scope-gated, and idempotent.
6. Map HTTP tool APIs into WebSocket RPC methods without weakening auth or policy.
7. Build a client-side adapter layer that shields UI code from raw runtime events.
8. Write tests that lock the SDK contract before broad external adoption.

---

## 1. What SDK dogfooding means

Dogfooding means using your own product the way an external user would.

For an SDK, dogfooding means:

```text
Do not only test the internal runtime.
Build a real app on the public SDK.
Then fix the SDK contract until the app becomes boringly reliable.
```

This is different from an internal smoke test.

An internal test may call private functions.

A dogfood app must use the public path:

```text
SDK -> Gateway RPC -> agent runtime -> Gateway events -> SDK -> app UI
```

That path is the product contract.

If it is unstable, the platform is unstable.

---

## 2. The happy path

The phrase "happy path" matters.

It means:

> the simplest correct end-to-end flow that must always work for a real developer

For OpenClaw, the SDK happy path is:

```text
Connect
  -> Discover
  -> Session
  -> Run
  -> Stream
  -> Result
  -> Cancel
  -> Approvals
```

Expanded:

1. Connect to the Gateway.
2. Receive `hello-ok`.
3. Discover supported agents, models, methods, policy, and limits.
4. Create or resume a session.
5. Start a run.
6. Stream normalized events.
7. Wait for the final result.
8. Cancel the run if needed.
9. Handle approvals if a tool or policy requires them.

If any one of these is ambiguous, every SDK client implements its own workaround.

That is how ecosystems become fragile.

---

## 3. Why OpenMeow is a good dogfood client

OpenMeow is a useful dogfood client because it is close enough to production behavior to expose SDK problems, but still controlled by the core team.

It needs:

- clean TypeScript types
- predictable run lifecycle state
- stable event names
- event-to-UI mapping
- cancellation semantics
- fixtures for regression tests
- adapter code that can survive protocol evolution

That combination creates pressure on the SDK.

The SDK cannot remain a thin wrapper around internal implementation details.

It must become a stable product boundary.

---

## 4. Problem found: event leakage

One dogfood finding was raw chat events leaking into `run.events()`.

That sounds small.

It is not.

The SDK is supposed to expose normalized events:

```text
run_started
assistant_delta
tool_started
tool_delta
tool_finished
run_completed
run_failed
run_cancelled
approval_required
```

But internal runtime events may look different:

```text
chat.delta
gateway.internal.message
raw.lifecycle.update
provider.chunk
tool.raw
```

If those internal events leak, the UI becomes coupled to private implementation details.

That causes:

- brittle UI code
- broken mobile clients
- version upgrade failures
- duplicate event handling
- impossible compatibility guarantees

The fix is an SDK adapter boundary:

```text
raw Gateway events
  -> normalize
  -> validate
  -> map to SDK event union
  -> expose to app
```

The app should not know the Gateway's internal event vocabulary.

---

## 5. Problem found: cancel inconsistency

Another dogfood finding was more serious:

```text
cancel() response
stream terminal event
wait() result
```

did not always agree.

That is a distributed systems bug.

Consider this bad state:

```text
cancel() returns: ok, cancelled
event stream emits: run.completed
wait() returns: timeout
```

The UI cannot know what happened.

Another bad state:

```text
cancel() returns: already finished
event stream emits: run.cancelled
wait() returns: ok
```

Now the client has a state machine contradiction.

For SDKs, this must be deterministic.

The platform needs one lifecycle truth:

```text
queued -> running -> terminal
```

Terminal states should be mutually exclusive:

```text
completed
failed
cancelled
timeout
skipped
```

After a terminal state, every surface should agree:

- `Run.cancel()`
- `Run.wait()`
- event stream terminal event
- run history
- UI state

---

## 6. The run lifecycle contract

A clean SDK should define run lifecycle behavior explicitly.

Example contract:

```text
Run.start()
  returns runId after server accepts the run.

Run.events()
  yields ordered normalized events until exactly one terminal event.

Run.wait()
  resolves to the same terminal state as the event stream.

Run.cancel()
  requests cancellation and returns the observed cancellation request result.
```

Important detail:

`Run.cancel()` is not the same thing as "the model instantly stopped."

Cancel is a request.

The final lifecycle state is confirmed when the runtime reaches a terminal state.

Good SDKs make that clear:

```text
cancel requested
  -> stream eventually emits run_cancelled
  -> wait returns cancelled
```

If the run already completed, cancel should say that clearly and not rewrite history.

---

## 7. Gateway RPC as the platform boundary

OpenClaw's external SDK path goes through Gateway RPC.

The Gateway protocol uses WebSocket JSON frames.

The basic shapes are:

```json
{ "type": "req", "id": "req-1", "method": "runs.create", "params": {} }
```

```json
{ "type": "res", "id": "req-1", "ok": true, "payload": {} }
```

```json
{ "type": "event", "family": "runs", "name": "run.delta", "payload": {} }
```

That framing matters because SDK clients need:

- request-response correlation
- typed errors
- ordered event streams
- method discovery
- feature negotiation
- auth scope enforcement
- payload limits
- idempotent retries

This is where "working app" becomes "platform API."

---

## 8. Discovery: `hello-ok.features.methods`

The server should advertise available RPC methods during connection setup.

In OpenClaw terms, methods appear in:

```text
hello-ok.features.methods
```

This lets SDK clients ask:

```text
Can this Gateway support artifacts.list?
Can this Gateway support tools.invoke?
Can this Gateway support run cancellation?
```

Discovery prevents hard-coded assumptions.

New methods must be registered in the server method list and any plugin method exports so clients can see them.

If a method is not discoverable, it is not a reliable SDK feature.

---

## 9. Schema and code generation

Gateway RPC schemas should be typed.

OpenClaw uses TypeBox-style schemas and generated protocol types.

The flow should be:

```text
define method schema
  -> generate protocol types
  -> server uses generated types
  -> SDK uses generated types
  -> tests verify compatibility
```

Developer workflow:

```bash
pnpm protocol:gen
pnpm protocol:check
pnpm protocol:gen:swift
```

The exact commands depend on the repo, but the principle is stable:

> schema first, generated clients, checked contract

This reduces drift between Gateway and SDKs.

---

## 10. Scopes and auth

SDK-facing RPC methods must be scope-gated.

Examples:

- `operator.read`
- `operator.write`
- `operator.admin`
- plugin-defined scopes

The server must enforce scopes.

Do not rely on the SDK to hide methods from unauthorized callers.

Events should also be scope-gated.

The safe default is fail-closed:

```text
If a client lacks visibility, do not broadcast the event to it.
```

This matters for multi-device and multi-user systems.

An SDK client should only see the sessions, runs, artifacts, and tool events that its token is allowed to see.

---

## 11. Idempotency

Side-effecting RPC methods need idempotency keys.

Examples:

- create artifact
- upload file
- start run
- approve tool call
- invoke tool
- cancel run

Why?

Because WebSocket clients reconnect.

Mobile networks fail.

User interfaces retry.

Without idempotency, a retry can create duplicate work.

Bad behavior:

```text
User taps "run"
network drops
client retries
two agent runs start
```

Better behavior:

```text
same idempotency key
  -> server returns same accepted run
```

Idempotency is not optional for serious SDKs.

---

## 12. Artifact APIs as SDK surface

Artifacts are a natural SDK feature.

They represent outputs such as:

- generated files
- downloaded documents
- logs
- images
- model outputs
- reports
- tool-produced bundles

An SDK-facing artifact surface may include:

```text
artifacts.list
artifacts.get
artifacts.download
artifacts.delete
artifacts.events
```

Design rules:

- register methods for discovery
- add TypeBox schemas
- generate SDK types
- require proper scopes
- gate broadcasts
- handle large payloads explicitly
- prefer metadata over huge inline blobs
- support download URLs or chunked transfer when needed

This is where transport policy matters.

Gateway hello policy can advertise limits such as:

- `maxPayload`
- `maxBufferedBytes`
- `tickIntervalMs`

Large artifacts should not silently break the WebSocket connection.

Use diagnostics such as `payload.large` and design around limits.

---

## 13. Mirroring HTTP `/tools/invoke` into Gateway RPC

OpenClaw already has a direct HTTP tool invoke path:

```text
POST /tools/invoke
```

A useful SDK improvement is a Gateway RPC equivalent:

```text
tools.invoke
```

Example request:

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

The critical rule:

> the RPC method must reuse the same policy, auth, deny-list, and execution semantics as the HTTP endpoint

Do not create a second, weaker path.

---

## 14. Tool policy and deny-list consistency

Direct tool invocation is dangerous if it bypasses policy.

The RPC method must enforce the same policy chain as HTTP:

- `tools.profile`
- `tools.byProvider.profile`
- `tools.allow`
- `tools.byProvider.allow`
- `agents.<id>.tools.allow`
- group policies
- sub-agent policies

It should also apply the same default deny-list for risky tool families, such as:

- shell and exec tools
- filesystem writes and deletes
- patch application
- session spawning and sending
- cron mutation
- gateway and node administration
- login or channel-management tools

Operators can customize Gateway allow/deny policy.

But the SDK path must not silently bypass the existing policy model.

---

## 15. Owner semantics and approvals

Gateway auth modes matter.

Shared-secret modes may act as full operator or owner credentials.

Device tokens may have narrower role semantics.

Trusted identity-bearing ingress may pass explicit scopes.

The SDK method must preserve those semantics.

For tool invocation:

- if the Gateway auth and tool policy allow the action, do not invent a second approval path
- if policy requires approval, surface that approval state cleanly
- preserve owner or actor identity in logs
- make permission failures typed and predictable

This is the difference between "SDK convenience" and "security hole."

---

## 16. Event model for SDK clients

SDK events should be normalized and stable.

A practical event union might look like:

```ts
type RunEvent =
  | { type: "run.started"; runId: string }
  | { type: "assistant.delta"; text: string }
  | { type: "tool.started"; toolCallId: string; tool: string }
  | { type: "tool.output"; toolCallId: string; text: string }
  | { type: "approval.required"; approvalId: string }
  | { type: "run.completed"; result: RunResult }
  | { type: "run.failed"; error: SDKError }
  | { type: "run.cancelled"; reason?: string };
```

The SDK adapter should:

- reject unknown required fields
- tolerate explicitly optional fields
- map raw Gateway events to stable SDK events
- preserve ordering per run
- expose terminal events exactly once
- keep raw debugging data behind an escape hatch

The UI should consume the SDK event union, not raw Gateway messages.

---

## 17. Adapter layer pattern

The dogfood work described an adapter layer over the SDK.

That is a strong pattern.

```text
Gateway event
  -> SDK protocol decoder
  -> app adapter
  -> UI state reducer
```

Responsibilities:

| Layer | Job |
|---|---|
| Gateway | Runtime truth, auth, scopes, ordering |
| SDK | Typed protocol, normalized events, retries, cancellation |
| Adapter | App-specific mapping and compatibility |
| UI reducer | Display state only |

Do not put Gateway protocol logic directly into React components, mobile screens, or CLI output code.

Keep protocol interpretation in one place.

---

## 18. Test harness and fixtures

The dogfood work included:

- adapter layer
- type definitions
- event-to-UI mapping
- fixture-based test harness
- 23 event fixtures
- passing tests

This is the right stabilization method.

A good SDK test suite should include:

- successful run
- streaming assistant output
- tool call with output
- approval required
- approval granted
- cancel before start
- cancel while running
- cancel after terminal state
- provider failure
- Gateway reconnect
- unknown event family
- oversized payload diagnostic
- artifact created
- artifact download failure

The purpose is not only correctness.

It is contract freezing.

When a future Gateway change breaks an SDK fixture, the team must consciously decide whether to preserve compatibility or version the contract.

---

## 19. Failure semantics

SDK users need typed failures.

Avoid vague errors like:

```text
Something went wrong.
```

Prefer:

```text
auth.denied
scope.missing
method.not_found
payload.too_large
run.cancelled
run.timeout
tool.denied
approval.required
provider.unavailable
protocol.invalid_event
```

Typed failures let external apps:

- show useful messages
- retry safely
- request re-authentication
- ask for approval
- degrade gracefully
- report actionable bugs

---

## 20. SDK stabilization checklist

Before calling an agent SDK production-ready, verify:

- `hello-ok` advertises supported methods, limits, and policy.
- Agents and models can be discovered without hard-coded names.
- Sessions can be created, resumed, and listed.
- Runs have a typed lifecycle.
- Event streams expose only normalized SDK events.
- `wait()` and terminal stream events agree.
- `cancel()` has deterministic semantics.
- Approvals are typed and testable.
- Artifact APIs handle metadata and large payloads safely.
- Tool invocation follows the same policy as existing HTTP paths.
- Side-effecting calls support idempotency.
- Auth scopes are enforced server-side.
- Events are scope-gated by default.
- Protocol schemas generate SDK types.
- Fixtures cover normal and failure paths.

---

## 21. Design exercise

Design the SDK happy path for a small external OpenClaw app.

The app should:

- connect with a device token
- discover agents and methods
- start a session
- send a task
- stream assistant and tool events
- show approval prompts
- list artifacts
- download one artifact
- cancel a long-running run
- show final result

Draw the state machine.

Then answer:

1. Which Gateway methods are required?
2. Which events are required?
3. Which methods need idempotency keys?
4. Which scopes are required?
5. What does the UI do if the event stream says cancelled but `wait()` says completed?
6. How will your SDK adapter prevent raw internal events from leaking into the UI?
7. What fixtures prove the contract is stable?

If you cannot answer these, your SDK is not ready for external developers.

---

## Key takeaways

- SDK dogfooding is a platform-stabilization milestone, not a polish task.
- The OpenClaw SDK happy path is connect, discover, session, run, stream, result, cancel, and approvals.
- Event leakage is a contract bug because external clients should see normalized SDK events, not internal runtime messages.
- Cancel semantics must be consistent across `cancel()`, event streams, `wait()`, and run history.
- Gateway RPC methods must be discoverable, typed, scope-gated, and generated from schemas.
- SDK-facing artifacts and tool invocation APIs must honor payload limits, idempotency, auth, and policy.
- A fixture-based adapter test harness is how the team freezes the client contract.

---

## References

- OpenMeow SDK RPC contract proposals: [https://github.com/OpenCoven/open-meow-sdk/blob/main/docs/rpc-contract-proposals.md#artifactslistgetdownload](https://github.com/OpenCoven/open-meow-sdk/blob/main/docs/rpc-contract-proposals.md#artifactslistgetdownload)
- OpenClaw Gateway protocol: [https://openclaw.knidal.com/gateway-protocol](https://openclaw.knidal.com/gateway-protocol)
- OpenClaw RPC adapters: [https://openclaw.knidal.com/rpc-adapters](https://openclaw.knidal.com/rpc-adapters)
- OpenClaw Tools Invoke API: [https://openclaw.knidal.com/tools-invoke-api](https://openclaw.knidal.com/tools-invoke-api)
- Case-study source repo: [OpenClaw](https://github.com/openclaw/openclaw)

---

*Next: [Lab 01 - Research Agent](Lab-01-Research-Agent.md)*
