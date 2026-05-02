# Lab 05 — Test the OpenClaw App SDK on macOS with the OpenCoven OpenMeow Dogfood Adapter

**Track B · Agentic AI & GenAI** | [← Index](README.md) | [Previous → Lab 04](Lab-04-TokenJuice-Output-Compaction.md)

---

## Overview

This lab tests the OpenClaw App SDK from the point of view of a real external macOS app.

The dogfood client is OpenCoven's `open-meow-sdk` repository.

OpenMeow is described as a native macOS notch/inbox client. The `open-meow-sdk` repo is the adapter and test harness that proves an app can use `@openclaw/sdk` without importing OpenClaw internals.

The core boundary:

```text
OpenMeow macOS app
  -> OpenMeow SDK adapter
  -> @openclaw/sdk
  -> OpenClaw Gateway RPC
  -> OpenClaw runtime
```

This is an app SDK lab, not a plugin SDK lab.

That distinction matters:

```text
App SDK:
  external app talks to Gateway

Plugin SDK:
  code runs inside OpenClaw
```

**Estimated time:** 60-90 minutes

**Difficulty:** Intermediate

---

## What you will test

The OpenMeow dogfood path focuses on the P0 app happy path:

```text
connect
  -> list agents
  -> create or reuse lane session
  -> send message
  -> stream normalized events
  -> wait for result
  -> cancel active run
  -> map state into app UI
```

The lab has two levels.

| Level | Goal | Requires live OpenClaw Gateway? |
|---|---|---|
| Fixture tests | Validate adapter, events, wait/cancel, UI reducer behavior | No |
| Live smoke test | Connect OpenMeow adapter to a local Gateway | Yes |

Do the fixture tests first.

They are deterministic and catch app-contract regressions before you debug networking.

---

## Prerequisites

On macOS:

```bash
xcode-select --install
brew install git node pnpm
node --version
pnpm --version
```

Recommended:

- Node.js 20 or newer
- Git
- A terminal with full disk access only if your local agent workflows need it
- Optional: a local OpenClaw Gateway
- Optional: Coven if you also want to test local harness sessions

Do not start with the native macOS UI.

Start with the adapter tests. The app UI should be the last layer you trust.

---

## Step 1 — Clone the dogfood adapter

```bash
mkdir -p ~/Developer/opencoven-labs
cd ~/Developer/opencoven-labs
git clone https://github.com/OpenCoven/open-meow-sdk.git
cd open-meow-sdk
```

Inspect the repo:

```bash
find . -maxdepth 3 -type f | sort
```

Key files:

| File | Why it matters |
|---|---|
| `src/index.js` | OpenMeow-side adapter over `@openclaw/sdk` |
| `src/index.d.ts` | app-facing TypeScript surface |
| `test/openmeow-sdk-client.test.js` | Node test coverage for adapter behavior |
| `fixtures/openclaw-events/*.jsonl` | normalized Gateway event fixtures |
| `scripts/validate-event-fixtures.mjs` | fixture schema and terminal-event validation |
| `docs/openmeow-sdk-adapter-shape.md` | smallest adapter OpenMeow needs |
| `docs/gateway-rpc-gap-map.md` | SDK method versus Gateway RPC readiness |
| `docs/openmeow-dogfood-plan.md` | phased dogfood plan |

---

## Step 2 — Run the deterministic tests

The package test command runs fixture validation and Node tests:

```bash
npm test
```

Expected shape:

```text
Validated N fixture events.
ok ...
```

If you want to run the pieces separately:

```bash
node scripts/validate-event-fixtures.mjs
node --test
```

These tests prove that the OpenMeow adapter can:

- wrap the SDK happy path
- list agents
- create lane sessions
- send messages
- stream events
- wait for results
- cancel active runs
- map normalized events into UI state
- keep the composer in send-or-stop mode
- distinguish wait deadline from runtime timeout
- keep unknown events debug-only

This is the most important part of the lab.

If these tests fail, do not debug the macOS UI yet.

---

## Step 3 — Read the adapter contract

Open the adapter shape:

```bash
sed -n '1,220p' docs/openmeow-sdk-adapter-shape.md
```

The important surface:

```ts
connect()
close()
listAgents()
getAgentIdentity(agentId)
createLaneSession({ agentId, label, sessionKey })
send(sessionKey, message)
events(runId)
wait(runId, timeoutMs)
cancel(runId, sessionKey)
effectiveTools(sessionKey)
```

This is intentionally smaller than the full OpenClaw Gateway.

That is good.

An app should not start with every internal capability.

It should start with the operations the UI actually needs.

---

## Step 4 — Understand the lane model

OpenMeow uses the idea of lanes.

A lane is an app-facing conversation target:

```text
Lane:
  agentId
  label
  sessionKey
  active run state
  streamed assistant draft
  compact tool activity
  approval cards
```

The session is durable.

The run is active work.

The lane is the UI container.

Mental model:

```text
agent = who should do the work
session = where memory/transcript lives
run = this specific execution
lane = how the macOS app presents it
```

This is the right split for a desktop client.

Do not let the UI confuse session identity with run identity.

---

## Step 5 — Inspect normalized event fixtures

List fixtures:

```bash
find fixtures/openclaw-events -type f -maxdepth 1 -print | sort
```

Preview one:

```bash
sed -n '1,20p' fixtures/openclaw-events/happy-path.jsonl
```

Every fixture event should have:

- `version`
- `id`
- `ts`
- `type`
- `runId`
- `data`

The validator also expects exactly one terminal run event:

- `run.completed`
- `run.failed`
- `run.cancelled`
- `run.timed_out`

This matters because UI reducers need a deterministic end state.

If a stream has no terminal event, the app may spin forever.

If a stream has multiple terminal events, the UI state becomes ambiguous.

---

## Step 6 — Trace the UI reducer

Open the test:

```bash
sed -n '1,260p' test/openmeow-sdk-client.test.js
```

Find these functions:

```text
mapOpenClawEventToOpenMeowUIEvent
initialOpenMeowUIState
reduceOpenMeowUIState
reduceOpenMeowRunState
markOpenMeowRunCancelling
reduceOpenMeowCancelResult
normalizeOpenMeowWaitResult
```

The important behaviors:

| Input event | UI behavior |
|---|---|
| `run.started` | active run indicator, stop enabled |
| `assistant.delta` | streaming assistant draft |
| `assistant.message` | finalized assistant message |
| `tool.call.started` | compact tool card |
| `tool.call.delta` | tool preview update |
| `tool.call.completed` | tool card completed |
| `approval.requested` | approval card |
| `approval.resolved` | approval card resolved |
| `run.completed` | composer returns to idle |
| `run.cancelled` | composer returns to idle with cancelled state |
| `run.timed_out` | terminal timeout state |

The UI rule:

```text
send or stop, never both
```

That is a practical product invariant.

---

## Step 7 — Write the dogfood checklist

Create a local checklist:

```bash
cat > DOGFOOD-CHECKLIST.md <<'EOF'
# OpenMeow App SDK Dogfood Checklist

## Fixture path

- [ ] `npm test` passes
- [ ] fixture validation passes
- [ ] each fixture has exactly one terminal run event
- [ ] unknown events remain debug-only
- [ ] send-or-stop composer invariant holds
- [ ] wait deadline and runtime timeout are distinct
- [ ] cancel failure leaves the run recoverable

## Live Gateway path

- [ ] Gateway reachable
- [ ] auth configured
- [ ] `agents.list` works
- [ ] session create/reuse works
- [ ] send returns stable `runId` and `sessionKey`
- [ ] events stream without losing early lifecycle events
- [ ] wait returns terminal result or accepted wait deadline
- [ ] cancel returns deterministic UI state
- [ ] approval request renders as a card

## Bugs to report upstream

- missing Gateway RPC:
- event ambiguity:
- wait/cancel mismatch:
- auth/discovery pain:
- native bridge pain:
EOF
```

This checklist is your test plan.

Dogfooding is not "it worked once."

Dogfooding is a repeatable pressure test against a real app workflow.

---

## Step 8 — Optional live Gateway smoke test

Only do this after the fixture tests pass.

You need:

- a local OpenClaw Gateway
- a Gateway URL
- auth token or local auth mode
- `@openclaw/sdk` available in your environment

The OpenMeow example documents this target shape:

```bash
pnpm add @openclaw/sdk
OPENCLAW_GATEWAY_URL=http://127.0.0.1:18789 pnpm tsx index.ts
```

Adapt the URL to your actual Gateway.

Create a local smoke script:

```bash
cat > live-smoke.mjs <<'EOF'
import { OpenClaw } from "@openclaw/sdk";
import { createOpenMeowSDKClient } from "./src/index.js";

const url = process.env.OPENCLAW_GATEWAY_URL;
const token = process.env.OPENCLAW_GATEWAY_TOKEN;
const agentId = process.env.OPENMEOW_AGENT_ID || "default";

if (!url) {
  throw new Error("Set OPENCLAW_GATEWAY_URL first");
}

const oc = new OpenClaw({
  url,
  token,
  requestTimeoutMs: 30_000,
});

const client = createOpenMeowSDKClient({ openClaw: oc });

await client.connect();

try {
  const agents = await client.listAgents();
  console.log("agents:", agents.map((agent) => agent.id || agent.name || agent.label));

  const session = await client.createLaneSession({
    agentId,
    label: "OpenMeow dogfood lane",
  });
  console.log("session:", session);

  const sent = await client.send(
    session.sessionKey,
    "Reply with one short sentence: OpenMeow SDK smoke test is connected."
  );
  console.log("sent:", sent);

  for await (const event of client.events(sent.runId)) {
    console.log("event:", event.type, event.runId || "");
    if (["run.completed", "run.failed", "run.cancelled", "run.timed_out"].includes(event.type)) {
      break;
    }
  }

  const result = await client.wait(sent.runId, 30_000);
  console.log("wait:", result);
} finally {
  await client.close();
}
EOF
```

Run:

```bash
export OPENCLAW_GATEWAY_URL="http://127.0.0.1:18789"
export OPENCLAW_GATEWAY_TOKEN="<token-if-required>"
export OPENMEOW_AGENT_ID="default"
node live-smoke.mjs
```

Expected behavior:

- connects to Gateway
- lists agents
- creates a lane session
- sends one message
- receives events
- exits after one terminal run event
- wait result is distinguishable from a wait deadline

If this fails, classify the failure:

| Failure | Likely layer |
|---|---|
| package import fails | local SDK install/package issue |
| connection refused | Gateway not running or wrong URL |
| auth rejected | token/auth mode mismatch |
| `agents.list` missing | Gateway feature/discovery mismatch |
| no events | event correlation or stream issue |
| wait says accepted forever | run still active or wait deadline too short |
| cancel state mismatches events | cancel contract bug |

---

## Step 9 — Optional Coven local harness check

This step is not required for App SDK validation, but it is useful if your OpenMeow workflow also wants local harness visibility.

Coven is the OpenCoven local harness substrate.

It gives Codex, Claude Code, and future harnesses a project-scoped, observable, attachable PTY runtime.

Clone and build:

```bash
cd ~/Developer/opencoven-labs
git clone https://github.com/OpenCoven/coven.git
cd coven
cargo build --workspace
```

Run the basic loop from a project:

```bash
cd /path/to/your/project
coven doctor
coven daemon start
coven run codex "summarize this repository"
coven sessions
coven attach <session-id>
```

If you do not have Codex or Claude Code installed, `coven doctor` should still help identify missing harnesses.

The architecture rule:

```text
OpenMeow consumes app state.
Coven supervises local harness sessions.
OpenClaw orchestrates agent runtime through Gateway.
Do not merge these authority boundaries.
```

---

## Step 10 — Report dogfood findings

Use this issue template:

```text
## OpenMeow App SDK Dogfood Finding

### Environment
- macOS version:
- Node version:
- OpenMeow SDK commit:
- OpenClaw version/commit:
- Gateway URL:
- Auth mode:

### Path
- fixture test / live Gateway / Coven harness / native UI

### Expected

### Actual

### Minimal repro

### Layer classification
- App adapter
- @openclaw/sdk
- Gateway RPC
- Gateway auth/discovery
- Event normalization
- wait/cancel semantics
- Approval semantics
- Native bridge
- Coven harness

### Proposed contract change
```

Good dogfood feedback should include a minimal repro.

Bad feedback is only:

```text
"the app feels broken"
```

Good feedback:

```text
"run.cancel() returns cancelled immediately, but the stream later emits run.completed.
Here is the fixture. The UI reducer enters idle twice."
```

---

## Step 11 — What to verify before calling it done

You are done when:

- `npm test` passes in `open-meow-sdk`
- fixture validation passes
- you can explain the adapter API
- you can explain the difference between wait deadline and runtime timeout
- you can explain send-or-stop state
- you can classify at least three possible failures by layer
- optional: live Gateway smoke test sends one run and receives one terminal event
- optional: Coven can launch or diagnose a local harness session

---

## Why this lab matters

This lab teaches a production pattern:

```text
Use a real app to test the SDK boundary.
```

An SDK that works only in unit tests is not enough.

An SDK that forces app authors to know internal Gateway details is also not enough.

The OpenMeow dogfood adapter is useful because it asks app-shaped questions:

- Can a real app discover agents?
- Can it create durable lanes?
- Can it stream events without race conditions?
- Can it stop a run deterministically?
- Can it show tool and approval state compactly?
- Can it avoid importing runtime internals?

These are the questions that make an agent runtime become a platform.

---

## Extensions

1. **Add one new fixture** — Create a fixture for `run.failed` with a tool error and verify the UI reducer shows a recoverable error state.
2. **Add approval UI logic** — Extend the fixture path to test multiple unresolved approvals.
3. **Build a tiny Swift bridge sketch** — Mirror the adapter methods as Swift protocol signatures.
4. **Add a live cancel test** — Start a long-running run, cancel it, and verify stream/wait/cancel all converge.
5. **Connect Coven state** — Add a fake Coven session-status card to the OpenMeow lane UI state model.

---

## References

- OpenCoven organization: [https://github.com/OpenCoven](https://github.com/OpenCoven)
- OpenMeow SDK architecture: [https://github.com/OpenCoven/open-meow-sdk](https://github.com/OpenCoven/open-meow-sdk)
- OpenMeow dogfood plan: [https://github.com/OpenCoven/open-meow-sdk/blob/main/docs/openmeow-dogfood-plan.md](https://github.com/OpenCoven/open-meow-sdk/blob/main/docs/openmeow-dogfood-plan.md)
- OpenMeow SDK adapter shape: [https://github.com/OpenCoven/open-meow-sdk/blob/main/docs/openmeow-sdk-adapter-shape.md](https://github.com/OpenCoven/open-meow-sdk/blob/main/docs/openmeow-sdk-adapter-shape.md)
- Gateway RPC gap map: [https://github.com/OpenCoven/open-meow-sdk/blob/main/docs/gateway-rpc-gap-map.md](https://github.com/OpenCoven/open-meow-sdk/blob/main/docs/gateway-rpc-gap-map.md)
- Gateway RPC contract proposals: [https://github.com/OpenCoven/open-meow-sdk/blob/main/docs/rpc-contract-proposals.md](https://github.com/OpenCoven/open-meow-sdk/blob/main/docs/rpc-contract-proposals.md)
- TypeScript basic-run example: [https://github.com/OpenCoven/open-meow-sdk/tree/main/examples/typescript-basic-run](https://github.com/OpenCoven/open-meow-sdk/tree/main/examples/typescript-basic-run)
- Coven local harness substrate: [https://github.com/OpenCoven/coven](https://github.com/OpenCoven/coven)

---

*End of Lab 05. Return to the [Track Index](README.md).*
