# Lecture 41 — Read the current Pi agent architecture

**Course:** [AI Agent Development](../Guide.md) | [Previous](Lecture-40.md) | [Next](Lecture-42.md)

**Source review: 18 October 2026.** The former `badlogic/pi-mono` repository now resolves to **earendil-works/pi**. This lesson inspects immutable revision `1402973a9dbe8c627341e04c9744c54953e285ca` (18 October, 20:43 UTC); its coding-agent package declares version 1.3.0. It does not claim to have installed Pi, called a model, or tested a desktop integration.

The prior five-package, `@mariozechner/*`, “No MCP” description is historical and does not describe this revision. Start from the [pinned repository](https://github.com/earendil-works/pi/tree/1402973a9dbe8c627341e04c9744c54953e285ca) and [package manifest](https://github.com/earendil-works/pi/blob/1402973a9dbe8c627341e04c9744c54953e285ca/packages/coding-agent/package.json). Star counts and release counts are not architectural evidence.

**Time:** 60–90 minutes. **Outcome:** trace a turn, explain session branching and MCP support, and identify the actual permission boundary. Complete the [executed local workflow](../Local-RAG-Workflow-Lab/Guide.md) first if you have not built a small harness.

## 1. Separate the runtime from the product surface

A model produces a response; the surrounding application assembles context, exposes tools, executes requests, records state and decides whether another turn is needed. Pi is an example of that application architecture. A CLI is one interface to it, not the definition of an agent.

```text
terminal / print / JSON / RPC / SDK
                 ↓
        session + agent orchestration
           ↙                 ↘
 provider/model adapter     tools / extensions / MCP
                 ↓
       recorded session events and results
```

The pinned README lists seven packages under the `@earendil-works` scope:

| Directory | Package | Responsibility described upstream |
|---|---|---|
| `chord` |`chord`|Typed plugin host, contracts, replicated state and RPC |
| `telemetry` |`pi-telemetry`|Telemetry contracts, adapter and conformance schemas |
| `ai` |`pi-ai`|Provider/model API abstraction |
| `durable` |`pi-durable`|Durable conversations, tasks and documents |
| `agent` |`pi-agent-core`|Agent tool loop and state management |
| `coding-agent` |`pi-coding-agent`|Interactive coding-agent CLI |
| `tui` |`pi-tui`|Terminal rendering library |

This is the checked source tree, not a promise that every consumer uses every package. A downstream project such as OpenClaw has its own dependency lock and integration revision. Inspect that lock before transferring an upstream API assumption. [Pinned package overview](https://github.com/earendil-works/pi/blob/1402973a9dbe8c627341e04c9744c54953e285ca/README.md).

## 2. Trace one agent turn

The active branch supplies conversation history. Pi combines it with system/context instructions, available tools and model settings. A provider response can contain both text and proposed tool calls. Tool execution and its results become part of the session. Another model request follows when the run still has work; otherwise it settles.

A tool schema describes an operation. It does not make the operation safe or correct. Think of a file edit: validate the intended path/content, execute it within the process's available permissions, capture the actual result and decide what to do after a failure. Merely logging a model's proposed edit is not evidence that the file changed.

Steering and follow-up messages enter at different lifecycle boundaries. That helps explain why a runtime needs queued messages and a run state instead of recursively appending every message to one unstructured string. [Pinned turn/context architecture](https://github.com/earendil-works/pi/blob/1402973a9dbe8c627341e04c9744c54953e285ca/packages/coding-agent/docs/how-pi-works.md).

## 3. A session tree is different from model context

Persistent sessions are JSONL entries linked by IDs and parent IDs. The active branch determines the history supplied to a model call. Keeping all entries on disk does not mean every branch is sent to the model.

| Action | Effect in the checked documentation |
|---|---|
| `/tree` |Navigate within the current session file |
| `/fork` |Create a separate session from an earlier user message |
| `/clone` |Copy the active branch into a new session |
| `/compact` |Use a summary/recent history for later model context; retain original entries |

A branch summary and a full event record serve different purposes. Summaries can lose details; retained events preserve evidence but may contain sensitive tool results. An ephemeral `--no-session` run cannot be resumed after exit. [Pinned session behavior](https://github.com/earendil-works/pi/blob/1402973a9dbe8c627341e04c9744c54953e285ca/packages/coding-agent/docs/sessions.md).

**Worked example:** messages A→B→C form one branch. Returning to B and adding D leaves both C and D in the event tree, but the active branch is A→B→D. A model request on that branch must not accidentally include C as if it were the current decision.

The [session format reference](https://github.com/earendil-works/pi/blob/1402973a9dbe8c627341e04c9744c54953e285ca/packages/coding-agent/docs/session-format.md) is the source for exact entry types and versioning. Do not write a parser that assumes every JSONL row is a user/assistant chat message. Model changes, compaction and custom events have different meanings.

## 4. Current Pi supports MCP directly

The pinned implementation documents local stdio and remote Streamable HTTP MCP servers, a `pi mcp` command surface and an interactive `/mcp` view. The old “No MCP; use a workaround” claim is no longer current.

Configuration can come from user-level `~/.pi/agent/mcp.json` and trusted project-level `.pi/mcp.json`. Project trust affects resource loading; it is not automatic permission for every tool result or operation. The transport, authentication, tool exposure and server process are separate contracts.

Use the [pinned MCP guide](https://github.com/earendil-works/pi/blob/1402973a9dbe8c627341e04c9744c54953e285ca/packages/coding-agent/docs/mcp.md) for its exact configuration. For a runnable server/client foundation, use this course's [executed SDK 2.3 lesson](../MCP%20for%20AI%20Agents/Lecture-05.md). No Pi-to-server connection was executed in this review.

**Diagnostic question:** if a server connects and lists tools, has the application proved that those tools are authorized for the current user? **Answer:** no; connection/discovery, identity, operation authorization and argument validity are separate checks.

## 5. Extensions run code, while skills provide instructions

Extensions register runtime functionality: tools, commands, lifecycle handlers, providers and other hooks. Skills supply instructions/supporting files loaded for relevant work. A prompt template expands input text. These mechanisms are not interchangeable.

The checked extension API distinguishes notification events, transformations and gates. A return value can be meaningful for one hook and ignored for another. Read the actual declared event/result type rather than assuming every callback can block a tool call. Long-running or asynchronous extension work needs an explicit lifecycle and cleanup contract. [Pinned extension documentation](https://github.com/earendil-works/pi/blob/1402973a9dbe8c627341e04c9744c54953e285ca/packages/coding-agent/docs/extensions.md).

`/reload` reloads declared resources such as extensions, skills, templates, themes and context files according to the implementation. It is not a universal promise to reload arbitrary module-level state, dependency code or every in-flight operation. Check the resource-specific behavior before depending on live reconfiguration. [Pinned command surface](https://github.com/earendil-works/pi/blob/1402973a9dbe8c627341e04c9744c54953e285ca/packages/coding-agent/docs/slash-commands.md).

## 6. Find the real permission boundary

The source documentation says Pi runs with its launching process's permissions and does not provide a built-in per-operation permission system. The working directory is a discovery/default path, not a filesystem sandbox. Extensions execute in the Pi process; restricting only built-in tools does not necessarily restrict those extensions.

Project trust controls which project resources load. It does not make their instructions trustworthy or constrain subsequent tool access. If stronger limits are required, the OS/container/VM boundary must restrict the files, credentials, processes and network exposed to the application. A broad shell tool makes this distinction especially important. [Pinned security model](https://github.com/earendil-works/pi/blob/1402973a9dbe8c627341e04c9744c54953e285ca/packages/coding-agent/docs/security.md).

Do not describe an unauthenticated local pipe as having no trust decisions, or a remote tool connection as automatically granting unrestricted execution. Define capabilities and enforcement at the actual boundary.

## 7. Connect this architecture to robotics and hardware

A high-level agent can inspect logs, retrieve a manual, propose a plan or request a bounded application operation. It should not replace a deterministic controller, watchdog or physical safety boundary. A persisted plan is not a guarantee that the physical world still matches its assumptions after restart.

Measure the actual workload: prompt/context length, tool calls, model latency, retained state and memory use. A minimal tool count does not imply a small context or low total inference cost. Tool outputs, image/audio inputs and repeated retries can dominate.

The [local workflow crash experiment](../Local-RAG-Workflow-Lab/Guide.md#6-inject-the-dangerous-crash-window) gives a concrete prerequisite: a checkpoint does not undo an action, and a retry needs an effect-specific idempotency contract. Transfer that reasoning before connecting any planning agent to a robot.

## 8. Source-reading exercise

Use the pinned links without installing or executing Pi:

1. Find the CLI package name/version and the seven-package overview. Explain which old course claims changed.
2. Trace the session branch A→B→D example and distinguish persistent history from model input.
3. Find current MCP support and explain why a historical README quote is insufficient evidence for present behavior.
4. Pick one extension hook and identify whether its return value notifies, transforms or gates.
5. Describe what a process with a shell tool could access under its actual OS identity, and what boundary would restrict it.

**Answers:** the current package scope is `@earendil-works`; the inspected CLI manifest is 1.3.0; MCP support is built in; the active branch selects model history; extension semantics are hook-specific; OS isolation, not a working-folder convention, supplies a resource boundary.

**Submit:** a short version/source manifest and one turn diagram with context, execution, persistence and permission boundaries. If you later install a pinned release and test an extension, record that separately. Source inspection alone is not an executed integration result.
