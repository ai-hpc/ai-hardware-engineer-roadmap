# Lecture 23 - OpenClaw Case Study: Gateway RPC Protocol

**Course:** [Agentic AI & GenAI](../Guide.md) | **Previous:** [Lecture 22](Lecture-22.md) | **Next:** [Lecture 24](Lecture-24.md)

---

Lecture 22 explained the App SDK from the outside.

This lecture goes one layer lower:

```text
App SDK / CLI / UI / node
  -> Gateway WebSocket RPC
  -> OpenClaw control plane
```

The Gateway RPC is the stable protocol boundary that lets clients, SDKs, automation, and nodes talk to OpenClaw without scraping private runtime internals.

The core idea:

> a production agent system needs a typed, authenticated, scope-gated control plane, not random ad-hoc HTTP endpoints and terminal output parsing

---

## Learning objectives

By the end of this lecture, you should be able to:

1. Explain the Gateway RPC frame model: `req`, `res`, and `event`.
2. Describe the connect handshake and `hello-ok` response.
3. Understand roles, scopes, and method-level access control.
4. Explain device pairing, device tokens, and node authentication.
5. Understand why feature discovery lives in `hello-ok.features`.
6. Explain broadcast scoping and per-client event ordering.
7. Describe why side-effecting methods need idempotency keys.
8. Understand shared-secret auth, trusted-proxy auth, private-ingress mode, and device-token reconnect behavior.
9. Design a new RPC method without leaking secrets or bypassing policy.

---

## 1. What the Gateway RPC is

The Gateway RPC is a WebSocket-based control plane.

It is used by:

- CLI clients
- desktop apps
- web UIs
- automation tools
- App SDK clients
- companion nodes
- headless node hosts

It carries:

- request/response RPC calls
- agent and tool events
- node transport frames
- presence updates
- pairing state
- diagnostics
- admin operations

In simple terms:

```text
Gateway RPC = OpenClaw's command bus
```

It is not just a chat stream.

It is the control plane for the whole runtime.

---

## 2. The frame model

Gateway RPC uses WebSocket text frames containing JSON.

There are three core frame shapes.

### Request

```json
{
  "type": "req",
  "id": "req-123",
  "method": "agents.list",
  "params": {}
}
```

### Response

```json
{
  "type": "res",
  "id": "req-123",
  "ok": true,
  "payload": {
    "agents": []
  }
}
```

Error response:

```json
{
  "type": "res",
  "id": "req-123",
  "ok": false,
  "error": {
    "type": "FORBIDDEN",
    "message": "Missing required scope"
  }
}
```

### Event

```json
{
  "type": "event",
  "event": "agent.delta",
  "payload": {},
  "seq": 42,
  "stateVersion": 7
}
```

Mental model:

```text
req/res = ask the Gateway to do or return something
event   = Gateway tells you something happened
```

---

## 3. Why WebSocket instead of plain REST

REST works well for simple request/response APIs.

Agent systems need more:

- live assistant deltas
- tool progress
- approval requests
- node presence
- pairing events
- stream lifecycle events
- reconnect behavior
- per-client event filtering

WebSocket gives OpenClaw one long-lived bidirectional channel:

```text
client -> Gateway: requests
Gateway -> client: responses and events
node -> Gateway: capabilities and command results
Gateway -> node: node.invoke commands
```

This is why the Gateway can serve both:

- operator clients
- node transports

on the same protocol family.

---

## 4. Connect handshake

The first frame must be a connect request.

Example shape:

```json
{
  "type": "req",
  "id": "connect-1",
  "method": "connect",
  "params": {
    "minProtocol": 3,
    "maxProtocol": 3,
    "client": {
      "id": "cli",
      "version": "1.2.3",
      "platform": "macos",
      "mode": "operator"
    },
    "role": "operator",
    "scopes": ["operator.read", "operator.write"],
    "auth": {
      "token": "..."
    },
    "device": {
      "id": "device_fp",
      "publicKey": "...",
      "signature": "...",
      "signedAt": 1737264000,
      "nonce": "..."
    }
  }
}
```

The important parts:

- protocol version range
- client metadata
- requested role
- requested scopes
- authentication material
- device identity
- signed nonce

The signed nonce matters because the server needs to know the client controls the device key.

Without nonce signing, a copied device ID would be too easy to fake.

---

## 5. The `hello-ok` response

A successful handshake returns `hello-ok`.

Conceptually it contains:

```text
protocol version
server metadata
connection id
auth outcome
role
granted scopes
device token, if issued
features.methods
features.events
policy limits
snapshot state
```

Example shape:

```json
{
  "type": "res",
  "id": "connect-1",
  "ok": true,
  "payload": {
    "type": "hello-ok",
    "protocol": 3,
    "server": {
      "version": "x.y.z",
      "connId": "conn-abc"
    },
    "auth": {
      "role": "operator",
      "scopes": ["operator.read", "operator.write"],
      "deviceToken": "..."
    },
    "features": {
      "methods": ["agents.list", "sessions.create", "agent.wait"],
      "events": ["agent.delta", "session.updated"]
    },
    "policy": {
      "tickIntervalMs": 30000,
      "maxPayload": 1048576,
      "maxBufferedBytes": 4194304
    }
  }
}
```

The key lesson:

> clients should trust `hello-ok.features`, not hard-code method availability

This is the same principle from Lecture 22:

```text
feature detection beats version guessing
```

---

## 6. Startup unavailable state

During startup, the Gateway may not be ready.

It can return a retryable unavailable error such as:

```json
{
  "type": "res",
  "id": "connect-1",
  "ok": false,
  "error": {
    "type": "UNAVAILABLE",
    "reason": "startup-sidecars",
    "retryable": true
  }
}
```

Client behavior:

- do not crash permanently
- back off
- retry connection
- surface "Gateway starting" in UI

This is part of deterministic startup behavior.

---

## 7. Roles

The two main roles are:

```text
operator
node
```

### Operator

An operator is a control-plane client.

Examples:

- CLI
- admin UI
- macOS app in operator mode
- App SDK automation
- dashboard

Operators ask the Gateway to:

- list agents
- create sessions
- start runs
- resolve approvals
- inspect status
- manage config if allowed

### Node

A node is a capability host.

Examples:

- macOS node mode
- iOS companion device
- Android companion device
- headless node host

Nodes expose capabilities such as:

- `canvas.*`
- `camera.*`
- `screen.*`
- `device.*`
- `notifications.*`
- `system.*`

The important boundary:

```text
operator = controls Gateway
node     = exposes device capability to Gateway
```

Nodes are not gateways.

---

## 8. Scopes

Roles say what kind of client this is.

Scopes say what that client is allowed to do.

Common operator scopes:

| Scope | Meaning |
|---|---|
| `operator.read` | Read status, sessions, agents, events |
| `operator.write` | Create or mutate normal runtime resources |
| `operator.admin` | Admin operations such as config/update/exec policy |
| `operator.approvals` | Approval management |
| `operator.pairing` | Device and node pairing operations |
| `operator.talk.secrets` | Secret-sensitive talk operations |

Reserved admin prefixes should require admin-level access:

```text
config.*
exec.approvals.*
update.*
```

Good rule:

> method-level access is the first gate, not the only gate

Some methods add deeper checks.

Example:

```text
config.get    -> read
config.patch  -> admin
node.invoke   -> role/scope check plus node command policy
```

---

## 9. Device identity and pairing

Device identity exists so the Gateway can recognize clients over time.

A connecting client may present:

- device ID
- public key
- signature
- timestamp
- nonce

If the device is not approved yet, the Gateway creates a pairing request.

Operator flow:

```bash
openclaw devices list
openclaw devices approve <requestId>
openclaw devices reject <requestId>
```

Once approved, the Gateway can issue a device token.

The client should persist the device token and reuse it for reconnects.

Why this matters:

```text
shared password/token:
  useful for bootstrap

device token:
  useful for durable least-privilege reconnects
```

Nodes should include a stable `device.id` derived from a keypair fingerprint.

Gateway tokens are issued per:

```text
device + role + approved scope set
```

Pairing approvals are required for new device IDs unless a tightly scoped local auto-approval path is enabled.

The safe default:

```text
new device ID
  -> pairing request
  -> operator approval
  -> device token issuance
```

Pairing auto-approval should be centered on direct local loopback connects.

Same-host tailnet or LAN connects should still be treated as remote unless explicitly trusted by configuration.

There are a few device-less operator exceptions, but they should be narrow:

- localhost-only insecure Control UI compatibility, if explicitly enabled
- successful trusted-proxy operator Control UI auth
- break-glass `dangerouslyDisableDeviceAuth`, which is a severe downgrade
- direct-loopback backend RPCs authenticated with the shared Gateway token/password

The rule:

> if a client is not in a narrow explicit trust path, require device identity and pairing

---

## 10. Device token lifecycle

Device tokens are first-class credentials.

The Gateway can rotate or revoke them:

```text
device.token.rotate
device.token.revoke
```

Safe behavior:

- non-admin callers can only manage their own device entries
- pairing scope rules still apply
- token rotation must not upgrade roles
- token revocation should make future reconnect fail
- token mutation cannot target a device role that pairing approval never granted
- non-admin callers cannot rotate or revoke a broader operator token than they already hold

This gives OpenClaw a cleaner security model than long-lived shared tokens everywhere.

### Persisting device tokens

After any successful connect, clients should persist the primary token:

```text
hello-ok.auth.deviceToken
```

On reconnect, the stored device token should reuse the approved scope set for that token.

Why this matters:

```text
first connect:
  approved scopes = operator.read + operator.write

reconnect:
  reuse stored device token
  preserve approved read/write access
```

Bad reconnect behavior:

```text
client reconnects with stored token
  -> silently collapses to narrower implicit scope
  -> status/probe/read UI breaks
```

Good reconnect behavior:

```text
stored device token
  -> approved role + scope set restored
```

If the caller supplies explicit scopes or an explicit device token, that caller-requested scope set stays authoritative.

Cached scopes are only reused when the client is reusing the stored per-device token.

### Rotation behavior

`device.token.rotate` returns rotation metadata.

It should echo the replacement bearer token only for same-device calls already authenticated with that device token.

That lets token-only clients persist their replacement before reconnecting.

Shared/admin rotations should not echo the bearer token.

Why:

```text
same-device token rotation:
  client needs the replacement token to keep working

shared/admin rotation:
  should not leak bearer tokens to broader control-plane clients
```

---

## 11. Auth paths

Gateway auth may support multiple paths:

- shared token
- shared password
- device token
- bootstrap token
- trusted proxy headers
- private-ingress / none, only in intentionally private deployments

### Shared-secret auth

Shared-secret Gateway auth uses one of:

```text
connect.params.auth.token
connect.params.auth.password
```

depending on configured auth mode.

On the client side, password and token are not identical:

```text
auth.password:
  orthogonal
  forwarded when set

auth.token:
  selected by priority
```

Token selection priority:

```text
1. explicit shared token
2. explicit deviceToken
3. stored per-device token keyed by deviceId + role
```

Bootstrap token behavior:

```text
auth.bootstrapToken is sent only when no auth.token was resolved
```

That means a shared token or any resolved device token suppresses bootstrap auth.

### Trusted proxy and private ingress

Identity-bearing modes can satisfy connect auth from request headers rather than `connect.params.auth.*`.

Examples:

```text
gateway.auth.allowTailscale = true
gateway.auth.mode = "trusted-proxy"
```

These modes are for deployments where an upstream layer already authenticates identity.

Private-ingress mode:

```text
gateway.auth.mode = "none"
```

skips shared-secret connect auth.

Use it only behind trusted private ingress.

Do not expose private-ingress mode on public or untrusted networks.

### Bootstrap handoff tokens

`hello-ok.auth.deviceTokens` can contain additional bootstrap handoff tokens.

Persist them only when the connection used bootstrap auth on a trusted transport such as:

```text
wss:// with appropriate trust
loopback / local pairing path
```

Do not blindly persist handoff tokens from an untrusted public connection.

The client should handle auth failures with recovery logic.

Useful error hints include:

```text
canRetryWithDeviceToken
recommendedNextStep
nonce/signature diagnostic code
```

Practical client behavior:

```text
auth failed
  -> check whether device token exists
  -> retry if server suggests it
  -> otherwise show pairing/login guidance
```

### `AUTH_TOKEN_MISMATCH`

For `AUTH_TOKEN_MISMATCH`, trusted clients may attempt one bounded retry with a cached per-device token.

Trusted means:

```text
loopback
or
wss:// with pinned tlsFingerprint
```

Public `wss://` without pinning does not qualify for automatic token promotion.

If the retry fails:

```text
stop automatic reconnect loop
surface operator action guidance
```

Do not spin forever with bad credentials.

Recovery hints may include:

| Field | Purpose |
|---|---|
| `error.details.code` | Stable machine-readable auth failure code |
| `error.details.canRetryWithDeviceToken` | Whether a device-token retry may help |
| `error.details.recommendedNextStep` | Suggested client/operator action |

Example recommended next steps:

```text
retry_with_device_token
update_auth_configuration
update_auth_credentials
wait_then_retry
review_auth_configuration
```

---

## 12. Device auth migration diagnostics

All connections should sign the server-provided `connect.challenge` nonce.

Legacy clients may still use pre-challenge signing behavior.

For those clients, Gateway auth should return stable `DEVICE_AUTH_*` detail codes.

| Message | details.code | details.reason | Meaning |
|---|---|---|---|
| device nonce required | `DEVICE_AUTH_NONCE_REQUIRED` | `device-nonce-missing` | Client omitted `device.nonce` or sent it blank |
| device nonce mismatch | `DEVICE_AUTH_NONCE_MISMATCH` | `device-nonce-mismatch` | Client signed with stale or wrong nonce |
| device signature invalid | `DEVICE_AUTH_SIGNATURE_INVALID` | `device-signature` | Signature payload does not match expected payload |
| device signature expired | `DEVICE_AUTH_SIGNATURE_EXPIRED` | `device-signature-stale` | Signed timestamp is outside allowed skew |
| device identity mismatch | `DEVICE_AUTH_DEVICE_ID_MISMATCH` | `device-id-mismatch` | `device.id` does not match public key fingerprint |
| device public key invalid | `DEVICE_AUTH_PUBLIC_KEY_INVALID` | `device-public-key` | Public key format or canonicalization failed |

Migration target:

```text
1. wait for connect.challenge
2. sign the payload that includes the server nonce
3. send the same nonce in connect.params.device.nonce
```

Preferred signing payload:

```text
v3 signature payload
  binds platform
  binds deviceFamily
  binds device/client/role/scopes/token/nonce fields
```

Legacy v2 signatures may remain accepted for compatibility, but paired-device metadata should still control command policy on reconnect.

---

## 13. TLS and pinning

Gateway WebSocket connections can use TLS.

Clients may optionally pin the Gateway certificate fingerprint.

Relevant configuration/CLI concepts:

```text
gateway.tls
gateway.remote.tlsFingerprint
--tls-fingerprint
```

Pinning matters because it upgrades "encrypted connection" into "I know exactly which Gateway certificate I expected."

This is why automatic stored-device-token promotion should be limited to:

```text
loopback
or
wss:// with pinned fingerprint
```

Without pinning, a public `wss://` endpoint is encrypted but not trusted enough for aggressive credential fallback.

---

## 14. Feature discovery

`hello-ok.features` is the discovery surface.

It advertises:

```text
methods
events
```

The client should use this to decide whether to show UI.

Example:

```ts
if (features.methods.includes("artifacts.list")) {
  showArtifactsPanel();
} else {
  hideArtifactsPanel();
}
```

Do not assume:

```text
OpenClaw version x.y.z means method exists
```

Assume:

```text
method exists only if hello-ok advertises it
```

This makes clients safer across version skew.

---

## 15. Size limits and payload safety

Before connect, frames are capped tightly.

The provided summary describes a pre-connect cap of:

```text
64 KiB
```

After handshake, clients should honor:

```text
hello-ok.policy.maxPayload
hello-ok.policy.maxBufferedBytes
hello-ok.policy.tickIntervalMs
```

If a client sends oversized data, the Gateway can emit diagnostics such as:

```text
payload.large
```

Then it may drop or close the connection.

Practical rule:

> do not push large files, videos, or screenshots as arbitrary JSON frames

Use:

- artifact metadata
- download handles
- chunking
- media attachments
- explicit file APIs

---

## 16. Events and broadcast scoping

Events are not broadcast blindly to everyone.

They are scope-gated.

Examples:

```text
chat / agent / tool frames:
  require operator.read

plugin broadcasts:
  default to operator.write or operator.admin depending on registration

status / heartbeat / presence / tick:
  generally not scope-restricted
```

The secure rule:

> if a client should not see a session, run, task, artifact, approval, or secret-adjacent event, do not broadcast it to that client

The Gateway also keeps ordering monotonic per socket.

That means each client gets its own sequence view after filtering.

This matters because scope filtering could otherwise make event ordering ambiguous.

---

## 17. Common RPC families

The Gateway has many method families.

Think in categories.

| Family | Examples | Purpose |
|---|---|---|
| System | `health`, `status`, `system-presence` | Liveness and runtime status |
| Config | `config.get`, `config.patch`, `config.apply`, `config.schema` | Controlled configuration |
| Update | `update.run`, `update.status` | Runtime update workflow |
| Agents | `agents.list`, `agents.create`, `agents.update` | Agent management |
| Sessions | `sessions.list`, `sessions.create`, `sessions.send`, `sessions.abort`, `sessions.compact` | Conversation state |
| Chat | `chat.history`, `chat.send` | Chat-facing operations |
| Runs | `agent.wait` | Wait for run lifecycle |
| Models | `models.list` | Model catalog and picker support |
| Usage | `usage.status`, `usage.cost` | Cost and usage reporting |
| Channels | `channels.status`, `web.login.start`, `web.login.wait`, `channels.logout` | External message surfaces |
| Nodes | `node.invoke`, `node.pair.*`, `node.pending.*` | Device and node transport |
| Approvals | `exec.approval.request`, `exec.approval.list`, `exec.approval.resolve` | Human approval flow |
| Automation | `cron.*`, `wake` | Scheduled and wake-based execution |
| Skills | `skills.*` | Skill discovery and management |
| Tools | `tools.catalog`, `tools.effective` | Tool visibility |

You do not need to memorize every method.

You need to understand the pattern:

```text
typed method
  -> schema
  -> scope gate
  -> handler
  -> discovery in hello-ok.features.methods
```

---

## 18. Idempotency

Side-effecting methods need idempotency keys.

Why?

Because real clients retry.

Bad behavior:

```text
mobile reconnects
  -> repeats sessions.send
  -> Gateway starts two runs
```

Good behavior:

```text
mobile reconnects
  -> repeats same request with same idempotency key
  -> Gateway returns same accepted operation
```

Methods that should use idempotency:

- create session
- send message
- start run
- cancel run
- approve action
- rotate token
- invoke side-effecting node command
- patch config
- schedule cron job

Idempotency is not polish.

It is required for reliable distributed clients.

---

## 19. TypeBox schemas and generated clients

Gateway RPC shapes should be schema-owned.

OpenClaw uses TypeBox-style schemas for canonical protocol shapes.

The workflow:

```text
define schema
  -> generate validators/types
  -> register method handler
  -> advertise method
  -> update SDK wrapper
  -> test client/server parity
```

Commands mentioned in the source material:

```bash
pnpm protocol:gen
pnpm protocol:check
```

The engineering goal:

> client and server should not silently disagree about method shapes

---

## 20. Secret safety

Diagnostics and discovery must not leak secrets.

Do not expose:

- raw chat bodies in diagnostic snapshots
- webhook request bodies
- tokens
- cookies
- secret values
- raw authorization headers

Expose summaries instead:

```text
capability: configured
auth: token-present
channel: connected
lastSeenAtMs: 1737264000
```

This is a core rule for control planes:

> observability is necessary, but raw secrets do not belong in status payloads

---

## 21. Nodes over Gateway RPC

Nodes connect to the same Gateway WebSocket protocol, but with:

```json
{ "role": "node" }
```

Nodes declare:

- device ID
- roles
- scopes
- capabilities
- commands
- permissions

The Gateway treats these as claims.

Claims are not enough.

The Gateway still enforces server-side policy:

```text
node declared command
  + gateway allowlist permits command
  + caller has scope
  + plugin policy permits command, if present
  -> node.invoke allowed
```

Presence methods can expose:

- `deviceId`
- roles
- scopes
- `lastSeenAtMs`
- reason
- capabilities

But again:

> capability reporting should not expose secrets

---

## 22. Model listing views

Model listing is a useful example of one method with multiple views.

`models.list` accepts a `view`.

| View | Meaning |
|---|---|
| omitted / `default` | runtime-allowed catalog, respecting default model policy |
| `configured` | picker-sized configured models |
| `all` | full Gateway catalog for diagnostics |

This is better than creating three unrelated methods.

It gives clients a clear contract:

```text
normal UI:
  default or configured

diagnostics:
  all
```

---

## 23. Reconnect and timeouts

Clients need predictable timeout behavior.

Typical values from the source material:

```text
per-RPC request timeout:
  30,000 ms

default tick interval before handshake:
  30,000 ms

reconnect backoff:
  initial 1s
  max 30s
```

After handshake, use the server policy:

```text
hello-ok.policy.tickIntervalMs
```

Client behavior:

- do not busy-loop reconnects
- back off
- reset faster only when the protocol says it is safe
- treat protocol mismatch as hard failure
- treat startup unavailable as retryable
- stop automatic reconnect loops after a failed bounded device-token retry

---

## 24. Extending the RPC surface

When adding a new method, use this checklist.

### 1. Define the method purpose

Bad:

```text
platform.doEverything
```

Good:

```text
artifacts.list
artifacts.get
artifacts.download
```

Keep the method narrow.

### 2. Add schema

Define request and response shapes with TypeBox.

### 3. Add scope gate

Examples:

```text
read-only discovery:
  operator.read

mutation:
  operator.write

admin:
  operator.admin

pairing:
  operator.pairing
```

### 4. Add idempotency if side-effecting

If retrying the request could duplicate work, require an idempotency key.

### 5. Advertise in `hello-ok.features.methods`

Clients should discover the method, not guess it.

### 6. Add events if needed

Scope-gate event families too.

### 7. Protect secrets

Never include raw secret values in status, diagnostics, or discovery.

### 8. Add SDK wrappers and generated native models

The public API is not complete until clients can use it safely.

---

## 25. Gateway protocol versus App SDK

Lecture 22 focused on:

```text
App SDK
  oc.agents
  oc.sessions
  oc.runs
  oc.models
  oc.approvals
```

This lecture focused on:

```text
Gateway RPC
  req/res/event frames
  handshake
  scopes
  pairing
  features
  policy limits
  node transport
```

Relationship:

```text
Gateway protocol = wire contract
App SDK          = developer-friendly wrapper
```

App authors should normally use the SDK.

SDK authors and platform engineers must understand the Gateway protocol.

---

## 26. Design exercise

Design a new RPC family:

```text
artifacts.*
```

Answer:

1. Which methods should exist?
2. Which methods are read-only?
3. Which scopes are required?
4. Do any methods need idempotency keys?
5. Which event family should announce artifact changes?
6. How should large files avoid `maxPayload` violations?
7. What should appear in `hello-ok.features.methods`?
8. What must never appear in diagnostics?

Then repeat the same design for:

```text
environments.*
```

Compare the difference between discovery-only APIs and mutation APIs.

---

## Key takeaways

- Gateway RPC is OpenClaw's WebSocket control plane and node transport.
- The wire model is small: `req`, `res`, and `event`.
- The connect handshake negotiates protocol, role, scopes, features, and policy.
- `hello-ok.features.methods` and `hello-ok.features.events` are the discovery surface.
- Roles identify the client type; scopes authorize specific actions.
- Device pairing and device tokens make durable authenticated clients possible.
- Broadcasts must be scope-gated and ordered per client socket.
- Side-effecting methods need idempotency keys.
- TypeBox schemas keep client and server protocol shapes aligned.
- Diagnostics must summarize state without leaking secrets.
- Nodes are capability hosts over the same Gateway protocol, not separate gateways.
- The App SDK wraps this protocol; it should not replace the protocol contract.

---

## References

- OpenClaw Gateway protocol: [https://openclaw.knidal.com/gateway-protocol](https://openclaw.knidal.com/gateway-protocol)
- OpenClaw App SDK: [https://openclaw.knidal.com/openclaw-app-sdk](https://openclaw.knidal.com/openclaw-app-sdk)
- OpenClaw Nodes: [https://openclaw.knidal.com/nodes](https://openclaw.knidal.com/nodes)
- OpenClaw Tools Invoke API: [https://openclaw.knidal.com/tools-invoke-api](https://openclaw.knidal.com/tools-invoke-api)
- Case-study source repo: [OpenClaw](https://github.com/openclaw/openclaw)

---

*Next: [Lecture 24 - What Is an AI Agent Harness? The Runtime Around the Model](Lecture-24.md)*
