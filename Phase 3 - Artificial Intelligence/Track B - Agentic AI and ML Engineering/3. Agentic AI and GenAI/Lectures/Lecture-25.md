# Lecture 25 - OpenCoven Case Study: Agent-Native Workspace and Local Harness Substrate

**Course:** [Agentic AI & GenAI](../Guide.md) | **Previous:** [Lecture 23](Lecture-23.md) | **Next:** [Lecture 26](Lecture-26.md)

---

Modern agent products are no longer just chat boxes.

They are becoming workspaces:

```text
agents
  + codebases
  + terminals
  + browsers
  + desktop apps
  + sessions
  + approvals
  + traces
  + local tools
  + external clients
```

OpenCoven is a useful case study because it is organized around this idea:

> an open-source AI-native workspace for agentic builders

The important lesson is not one specific repository.

The lesson is how to split an agent workspace into clean runtime pieces without collapsing everything into one unsafe monolith.

---

## Learning objectives

By the end of this lecture, you should be able to:

1. Explain what an agent-native workspace is.
2. Describe the OpenCoven ecosystem at a high level.
3. Understand Coven as a local harness substrate for project-scoped agent sessions.
4. Explain why the Rust daemon is the authority boundary in Coven.
5. Distinguish OpenClaw, OpenCoven, Coven, OpenMeow, desktop-use, and PsiClaw.
6. Understand why external app SDKs and internal plugin SDKs must stay separate.
7. Explain how desktop automation should be isolated behind a narrow adapter.
8. Describe observable, attachable PTY sessions as a product primitive.
9. Identify safety boundaries for local agents that can run tools and touch files.
10. Design a local-first agent workspace for hardware and AI systems work.

---

## 1. The problem OpenCoven is addressing

Most agent systems start with this shape:

```text
user
  -> chat UI
  -> model
  -> tool calls
```

That is too small for real engineering work.

Real builders move across:

- repositories
- terminals
- code editors
- browser sessions
- design documents
- local model tools
- long-running agents
- multiple harnesses such as Codex and Claude Code
- external clients such as desktop apps and dashboards

The failure mode is fragmentation:

```text
one tool owns the terminal
another owns the chat
another owns traces
another owns approvals
another owns desktop automation
another owns session history
```

The workspace becomes impossible to reason about.

OpenCoven's direction is to make these surfaces coherent while keeping the authority boundaries explicit.

---

## 2. The OpenCoven ecosystem map

OpenCoven is an organization/ecosystem, not one single runtime.

The useful mental model:

```text
OpenCoven ecosystem
  |
  +-- Coven
  |     local Rust harness substrate for project-scoped sessions
  |
  +-- OpenMeow SDK architecture
  |     app-client dogfood path for OpenClaw's external SDK
  |
  +-- desktop-use
  |     external desktop automation adapter for OpenClaw/OpenCoven surfaces
  |
  +-- PsiClaw
  |     desktop companion model workspace, operator console, eval harness prototype
  |
  +-- distribution repos
        downloads, Homebrew tap, package plumbing
```

The architectural theme:

```text
Each component owns one job.
Each boundary has a contract.
No component should silently widen another component's authority.
```

That is the main production lesson.

---

## 3. The workspace architecture pattern

An agent-native workspace needs at least five layers.

```text
UI layer
  - desktop client
  - cockpit
  - inbox
  - web console

SDK / protocol layer
  - typed app SDK
  - Gateway RPC
  - local socket API
  - normalized events

Runtime layer
  - agent loop
  - harness sessions
  - PTY lifecycle
  - approvals

Tool and automation layer
  - desktop-use
  - terminal
  - browser
  - file system
  - APIs

Memory / trace layer
  - session events
  - task logs
  - eval traces
  - durable memories
```

OpenCoven is interesting because its pieces map onto these layers instead of pretending one app should own everything.

---

## 4. Coven: the local harness substrate

Coven is the clearest technical center of the OpenCoven ecosystem.

Its thesis:

```text
One project. Any harness. Visible work.
```

Coven runs coding agents and future harnesses inside explicit local project boundaries.

The first target harnesses are local CLI-style agents such as:

- Codex
- Claude Code
- future harness adapters

Core capabilities:

| Capability | Meaning |
|---|---|
| Project-root boundary | every run is tied to an explicit repository/project root |
| Harness-neutral runtime | supports different agent CLIs through adapters |
| Attachable PTY sessions | users can follow and reattach to live work |
| Local daemon API | clients can list, launch, observe, attach, input, and kill |
| SQLite-backed history | session metadata and event logs survive restarts |
| Rust authority layer | launch, cwd, input, kill, and path checks are revalidated in Rust |

This is not "another chat UI."

It is closer to:

```text
local process supervisor + PTY manager + session/event database + safe project boundary
```

---

## 5. Coven command flow

The user-facing loop is intentionally simple:

```bash
cd /path/to/project
coven doctor
coven daemon start
coven run codex "fix the failing tests"
coven run claude "polish this UI"
coven sessions
coven attach <session-id>
```

What happens underneath:

```text
coven run
  -> validate project root
  -> validate cwd is inside project root
  -> choose known harness adapter
  -> launch harness through managed PTY
  -> persist session metadata
  -> append session events
  -> expose status and events through local socket API
```

This gives agents a shared "room" to run in.

The user or another client can attach later instead of losing the work when a terminal closes.

---

## 6. The local API as the product contract

Coven exposes a small local API over a Unix socket.

The important endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /health` | daemon health and metadata |
| `GET /sessions` | list sessions |
| `POST /sessions` | launch a session |
| `GET /sessions/:id` | fetch one session |
| `GET /events?sessionId=...` | read session events |
| `POST /sessions/:id/input` | send input to a live session |
| `POST /sessions/:id/kill` | kill a live session |

Design rule:

```text
The socket API is the contract.
The Rust daemon is the authority.
Clients are presentation/integration layers.
```

This matters because it prevents a UI or plugin from becoming the real policy engine by accident.

---

## 7. Why Rust is the authority boundary

Coven's operational model puts authority in the Rust layer.

That means Rust owns:

- project-root validation
- cwd canonicalization
- symlink escape rejection
- PTY lifecycle
- process launch
- input forwarding
- kill requests
- daemon state
- event persistence
- local socket behavior

TypeScript, UI clients, OpenClaw plugins, and wrappers may validate inputs for UX.

But they do not get final authority.

Correct model:

```text
Client validates for convenience.
Rust revalidates for authority.
```

Wrong model:

```text
The UI checked the path, so the daemon can trust it.
```

For local agents that can edit repositories and run commands, this boundary is not optional.

---

## 8. OpenClaw integration: external plugin, not core merge

OpenCoven's docs are explicit about one important boundary:

```text
OpenClaw core does not include Coven code.
OpenClaw integrates through an external @opencoven/coven plugin.
```

The flow:

```text
OpenClaw ACP runtime
  -> external @opencoven/coven plugin
  -> local Coven socket
  -> Rust daemon
  -> harness PTY
```

OpenClaw remains responsible for:

- chat/session routing
- ACP bindings
- task state
- permissions UX
- user-facing delivery

Coven remains responsible for:

- local harness supervision
- project-root enforcement
- PTY lifecycle
- socket contract
- session/event persistence

This is a clean integration pattern:

```text
OpenClaw orchestrates.
Coven supervises local harness sessions.
The plugin adapts between them.
```

Do not collapse these roles.

---

## 9. OpenMeow: dogfooding the App SDK boundary

OpenMeow is a native macOS notch/inbox style client direction.

The OpenMeow SDK architecture repo is useful because it clarifies the app SDK boundary.

The key rule:

```text
OpenMeow should call @openclaw/sdk.
@openclaw/sdk should call OpenClaw Gateway RPCs.
OpenMeow should not import OpenClaw internals or shell out to private commands.
```

Architecture:

```text
OpenMeow app
  -> OpenMeow SDK adapter
  -> @openclaw/sdk
  -> OpenClaw Gateway WebSocket RPC
  -> OpenClaw runtime
```

This is the same product lesson as Coven:

```text
Use a boring public contract.
Do not build apps against internals.
```

For external apps, the stable nouns are:

- agents
- sessions
- runs
- models
- tools
- approvals
- normalized events

The app should build UI state from normalized events, not from provider-specific raw streams.

---

## 10. desktop-use: keep OS automation outside the core

Desktop automation is powerful and dangerous.

OpenCoven's `desktop-use` repo takes a narrow-adapter approach.

The pattern:

```text
OpenClaw desktop_use tool
  -> execFile("coven-desktop-use", args)
  -> platform backend
```

The adapter owns platform-specific desktop automation.

OpenClaw owns:

- tool registration
- tool policy
- approvals
- session routing
- user-visible delivery

The adapter owns:

- screenshots
- frontmost app inspection
- clicks
- typing
- keypresses
- scrolls
- platform-specific implementation details

Safety properties from the current design:

- no shell interpolation
- process arguments are passed directly
- interactive actions require `--confirm`
- typed text is redacted from command echoes and stdout where relevant
- unsupported platforms return clean JSON instead of pretending support exists

This is a good pattern for computer-use tooling:

```text
Narrow binary.
JSON envelope.
No shell interpolation.
Explicit confirmation.
Runtime policy outside the adapter.
```

---

## 11. PsiClaw: desktop companion model and operator console

PsiClaw is a desktop companion model workspace direction.

The repo frames it as:

- training ground
- evaluation harness
- operator console
- specialized VLM workspace for desktop control

Important status note:

```text
PsiClaw is currently a prototype/UI harness direction, not a fully wired live ML backend.
```

The interesting product concepts are still valuable:

| Concept | Why it matters |
|---|---|
| Desktop-native model | browser-only agents are not enough for local workflows |
| Operator console | proposed actions should be visible and approvable |
| Trace explorer | state, reasoning, action, and outcome need replay |
| Eval dashboard | desktop agents need measurable success and intervention rates |
| Fine-tuning harness | recovery episodes can become training data |

The safe action loop:

```text
observe
  -> route
  -> propose
  -> approve or deny
  -> execute
  -> re-read state
  -> trace
```

This is the correct direction for desktop agents.

An agent that controls a machine must be observable before it is autonomous.

---

## 12. How OpenCoven relates to OpenClaw

The recent lectures introduced two adjacent pieces:

| System | Primary role |
|---|---|
| OpenClaw | Gateway, agent runtime, sessions, tools, nodes, approvals |
| OpenCoven | agent-native workspace and local harness ecosystem |

Practical split:

```text
OpenClaw
  controls agent runtime and gateway protocol

OpenCoven / Coven
  runs and observes local harness sessions inside project boundaries

OpenMeow
  presents external app UI through the App SDK

desktop-use
  performs narrow desktop automation under policy

PsiClaw
  explores local desktop VLM training, evals, and operator UX
```

This split is more important than any individual tool name.

It teaches a durable architecture principle:

```text
Separate orchestration, harness execution, UI, desktop actuation, and model training.
```

---

## 13. Why this matters for AI hardware engineers

This course is about AI hardware, but agent workspaces matter because they shape inference demand.

OpenCoven-style systems create workloads that look different from simple chatbot inference:

- long-running sessions
- local model inference
- multimodal desktop perception
- event streaming
- trace storage
- low-latency action loops
- tool-heavy execution
- multiple concurrent harnesses
- local-first security constraints

For hardware and systems engineers, this affects:

- memory footprint for local VLMs
- CPU/GPU scheduling under interactive workloads
- storage and event logging patterns
- edge inference latency requirements
- model quantization choices
- UI perception workload design
- safe command execution on developer machines

A desktop companion agent is not just "one model call."

It is a system of:

```text
perception + planning + tools + approvals + traces + local process control
```

That is why Jetson, Apple Silicon, workstation GPUs, and local inference runtimes matter.

---

## 14. Design principles from OpenCoven

### Principle 1: Make work visible

If an agent is modifying a project, the user should be able to inspect:

- what session exists
- what harness is running
- what project root it owns
- what output has streamed
- what action is currently pending
- how to attach or stop it

Invisible autonomy is a debugging and safety failure.

### Principle 2: Keep boundaries boring

Good boundaries are small:

```text
Gateway RPC
local socket API
JSON command envelope
PTY event stream
SDK event model
```

Bad boundaries are implicit:

```text
scrape terminal output
import private runtime files
trust UI validation
let plugin config widen daemon authority
```

### Principle 3: Revalidate at the authority layer

The daemon that launches processes must revalidate:

- project root
- cwd
- harness id
- kill target
- input target
- path-sensitive requests

Never outsource that to a UI.

### Principle 4: Prefer adapters over core coupling

OpenClaw should not absorb every runtime.

Better:

```text
OpenClaw plugin
  -> local socket/API
  -> external daemon/binary
```

That keeps failures, release cycles, and trust roots separate.

### Principle 5: Treat desktop actions as high risk

Clicks, typing, keypresses, screenshots, file operations, and terminal commands need explicit policy.

Even if the model is good, the action surface is high-impact.

---

## 15. Practical architecture: local agent workspace for hardware bring-up

Imagine a hardware engineer debugging a Jetson + ESP32-C6 + microphone-array setup.

OpenCoven-style workspace:

```text
OpenMeow / cockpit UI
  -> shows tasks, sessions, approvals, traces

OpenClaw Gateway
  -> routes user messages and agent runs

Trace and memory store
  -> stores docs, logs, board-specific notes, and session evidence

Coven daemon
  -> runs Codex or Claude Code inside the repo/project root

desktop-use adapter
  -> captures screenshots or desktop state when explicitly approved

PsiClaw-style eval harness
  -> records trace quality and task success over time
```

Example workflow:

```text
1. User asks: "Analyze why OTBR stays detached after Thread start."
2. OpenClaw creates an agent run.
3. The trace/memory store retrieves previous OTBR logs and kernel config notes.
4. Coven launches a coding/research harness inside the project repo.
5. The harness reads docs, proposes a markdown update, and runs local checks.
6. OpenClaw streams progress and approvals to the UI.
7. The final trace is saved for future memory/eval.
```

This is realistic engineering automation.

It keeps the agent productive without giving every component unlimited authority.

---

## 16. Risk checklist

When evaluating or building an OpenCoven-style workspace, check these risks.

| Risk | Failure mode | Practical control |
|---|---|---|
| Project escape | agent runs from wrong cwd or follows symlink outside repo | canonicalize and validate in daemon |
| Shell injection | prompt becomes part of shell command | use argv APIs, avoid `sh -c` |
| Hidden authority | plugin config expands daemon permissions | daemon revalidates all sensitive requests |
| Event leakage | logs capture secrets or tokens | redact and avoid intentional secret storage |
| UI false confidence | UI says safe but daemon acts differently | daemon is source of truth |
| Desktop overreach | agent clicks/types without user awareness | confirmations and action traces |
| SDK instability | app depends on private runtime behavior | typed SDK + Gateway RPC only |
| Unbounded sessions | stale agents keep running | list, attach, kill, timeouts, daemon supervision |

The practical rule:

```text
If a component can cause external side effects, give it a narrow contract and an observable audit trail.
```

---

## 17. Lab-style exercise

Design a minimal OpenCoven-inspired local agent workspace for this roadmap repo.

Requirements:

1. The user can launch a coding agent only inside the roadmap repository.
2. The user can list active sessions.
3. The user can attach to an existing run.
4. The user can kill a stuck run.
5. The UI receives normalized events.
6. The runtime stores session metadata and event history.
7. Any desktop action requires explicit confirmation.
8. OpenClaw integration happens through an external plugin, not core code.

Draw the boundary diagram:

```text
UI client
  -> SDK / Gateway
  -> plugin adapter
  -> local socket API
  -> Rust daemon
  -> harness PTY
```

Then answer:

1. Which layer validates project root?
2. Which layer owns user-facing approval UX?
3. Which layer owns process launch?
4. Which layer owns event normalization?
5. Which layer stores durable memory?
6. Which layer should never import OpenClaw internals?

If the answers are unclear, the architecture is not ready.

---

## Key takeaways

- OpenCoven is an AI-native workspace ecosystem for agentic builders.
- Coven is the local harness substrate: project-scoped, observable, attachable sessions.
- The Rust daemon is the authority boundary for launch, cwd, PTY, input, kill, and path-sensitive requests.
- OpenClaw integration should stay external through a plugin and local socket API.
- OpenMeow is a useful App SDK dogfood client pattern: apps use `@openclaw/sdk`, not internals.
- `desktop-use` shows how to keep OS automation in a narrow adapter under runtime policy.
- PsiClaw is a prototype direction for desktop companion models, operator consoles, traces, and evals.
- The durable architecture lesson is separation of concerns: workspace UI, gateway runtime, memory, harness supervision, and desktop actuation are different layers.

---

## References

- OpenCoven GitHub organization: [https://github.com/OpenCoven](https://github.com/OpenCoven)
- OpenCoven profile README: [https://github.com/OpenCoven/.github/blob/main/profile/README.md](https://github.com/OpenCoven/.github/blob/main/profile/README.md)
- Coven repository: [https://github.com/OpenCoven/coven](https://github.com/OpenCoven/coven)
- Coven product spec: [https://github.com/OpenCoven/coven/blob/main/docs/PRODUCT-SPEC.md](https://github.com/OpenCoven/coven/blob/main/docs/PRODUCT-SPEC.md)
- Coven operational model: [https://github.com/OpenCoven/coven/blob/main/docs/OPERATIONAL-MODEL.md](https://github.com/OpenCoven/coven/blob/main/docs/OPERATIONAL-MODEL.md)
- OpenMeow SDK architecture: [https://github.com/OpenCoven/open-meow-sdk](https://github.com/OpenCoven/open-meow-sdk)
- desktop-use adapter: [https://github.com/OpenCoven/desktop-use](https://github.com/OpenCoven/desktop-use)
- PsiClaw: [https://github.com/OpenCoven/psi-claw](https://github.com/OpenCoven/psi-claw)
- OpenClaw Gateway protocol: [https://openclaw.knidal.com/gateway-protocol](https://openclaw.knidal.com/gateway-protocol)

---

*Next: [Lecture 26 - OpenKnots Case Study: Trustworthy Agent Interfaces and Local-First Coding Surfaces](Lecture-26.md)*
