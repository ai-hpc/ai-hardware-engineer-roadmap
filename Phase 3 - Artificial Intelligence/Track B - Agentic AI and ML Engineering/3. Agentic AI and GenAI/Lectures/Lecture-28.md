# Lecture 28 - Pi (pi-mono): A Detail Reading of a Minimal Coding Agent

**Course:** [Agentic AI & GenAI](../Guide.md) | **Previous:** [Lecture 27](Lecture-27.md) | **Next:** [Lecture 29](Lecture-29.md)

---

This lecture is a close reading of the actual Pi repository: [github.com/badlogic/pi-mono](https://github.com/badlogic/pi-mono). Pi is the coding-agent substrate that sits beneath OpenClaw and several other agent products. Rather than describe the surface, this lecture pulls apart the repo and shows you how each design decision is wired in code: which packages exist, which tools are built in, how extensions register, how sessions are stored as branchable JSONL, what hot reload actually reloads, and what `No MCP` looks like as a real engineering stance with a concrete workaround.

Primary sources:

- [`badlogic/pi-mono`](https://github.com/badlogic/pi-mono) — the repository itself, MIT-licensed, currently at v0.73.0 with 212 releases and 44.9k stars.
- [`packages/coding-agent` README](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/README.md) — the CLI / TUI surface.
- [`packages/agent` README](https://github.com/badlogic/pi-mono/tree/main/packages/agent) — the agent runtime framework.
- Armin Ronacher, *Pi: The Minimal Agent Within OpenClaw* (Jan 31, 2026) — context and motivation.

Where this lecture quotes specifics (slash commands, file paths, command-line flags, CLI tool names), those come from the project's own README at the time of writing.

---

## Learning objectives

By the end of this lecture, you should be able to:

1. Name the five packages in pi-mono and explain what each one does.
2. List Pi's four built-in tools and the three optional ones available via CLI flags.
3. Read a Pi session JSONL file and explain how `id` / `parentId` produce a tree.
4. Explain the difference between `/tree`, `/fork`, and `/clone` semantically.
5. Write a minimal Pi extension that registers one tool, one slash command, and one event handler.
6. Explain what `/reload` reloads, what hot-reloads automatically, and what does not reload at all.
7. Tell a beginner why Pi has no MCP and what to do instead when MCP is required.
8. Map every Pi design decision back to a section of Lecture 24 (harness concerns) and Lecture 24b (event-sourced session).

---

## 1. What Pi actually is

Pi (`pi-mono`) is a TypeScript monorepo that ships, as its primary product, a coding-agent CLI installed with one command:

```bash
npm install -g @mariozechner/pi-coding-agent
export ANTHROPIC_API_KEY=sk-ant-...
pi
```

Authentication can also use `/login` for subscription-based providers; the CLI then runs as an interactive TUI. The package is one of five in the monorepo, all MIT-licensed, all under the `@mariozechner/*` npm scope.

The repo description: *"Tools for building AI agents."* That is more accurate than calling Pi "a coding agent" — Pi is an *agent runtime kit*, of which the coding-agent CLI is the most visible product but not the only one.

---

## 2. The monorepo, package by package

```
pi-mono/
  packages/
    ai/               @mariozechner/pi-ai             multi-provider LLM API
    agent/            @mariozechner/pi-agent-core     agent runtime + tool calling
    coding-agent/     @mariozechner/pi-coding-agent   the `pi` CLI / TUI
    tui/              @mariozechner/pi-tui            differential-rendered TUI library
    web-ui/           @mariozechner/pi-web-ui         web components for chat UI
  .pi/                                                 self-config for development
  .github/                                             CI workflows
  scripts/                                             build / release scripts
```

Read top-down, this is a stack: `ai` is the model abstraction, `agent` is the runtime that calls models and dispatches tools, `coding-agent` is the CLI that wires `agent` to a TUI, `tui` is the rendering library, `web-ui` is the equivalent surface for a browser. Each package is independently published; consumers can pick the layer they need.

A consequence: **a non-coding-agent product (a Slack bot, a Telegram bot, OpenClaw itself) consumes `@mariozechner/pi-agent-core` and `@mariozechner/pi-ai` directly** and supplies its own front-end. The coding-agent CLI is one consumer of the runtime, not the only one.

---

## 3. The built-in tool surface

Pi's **default** tool set is four:

| Tool | Purpose |
|---|---|
| `read` | Read a file's contents |
| `write` | Create or overwrite a file |
| `edit` | Structural edit on an existing file |
| `bash` | Execute a shell command |

Three more are available through CLI flags but are not enabled by default:

| Tool | Purpose |
|---|---|
| `grep` | Text search across the workspace |
| `find` | File-name search |
| `ls` | Directory listing |

The README's framing is exact: *"by default, pi gives the model four tools: `read`, `write`, `edit`, and `bash`."* The optional three exist for cases where the model would otherwise burn tokens spawning `bash -c "grep ..."`; they are convenience, not capability expansion.

The structural argument for keeping the default at four: **most filesystem and shell capabilities compose from `bash`**. A model that can write a script and run it can do `grep`, `find`, `ls`, `git`, `curl`, `npm`, and any other CLI without each one having to be a separately registered tool whose schema lives in the prompt. This is the same principle as Lecture 24 §6.5: too many tools confuse selection and waste tokens.

---

## 4. The slash command surface

Pi ships approximately twenty built-in slash commands. They group into clear categories.

**Authentication and identity**

```
/login          OAuth flow for subscription-based providers
/logout         Drop credentials
/model          Switch the active model
/scoped-models  Mark which models cycle under Ctrl+P
/settings       Thinking level, theme, message delivery, transport
```

**Session control**

```
/new            Start a new session
/resume         Pick from previous sessions
/name <name>    Set the current session's display name
/session        Show session info
/quit           Quit
```

**Tree navigation (the interesting part)**

```
/tree           Jump to any point in the current session's tree
/fork           Create a NEW session file from a previous user message
/clone          Duplicate the current active branch into a new session file
/compact        Manually compact context (with optional prompt)
/reload         Reload keybindings, extensions, skills, prompts, context files
```

**Output and sharing**

```
/copy           Copy last assistant message to clipboard
/export [file]  Export session to HTML
/share          Upload as a private GitHub gist
/changelog      Show version history
/hotkeys        Show all keyboard shortcuts
```

**Extension surface**

```
/skill:<name>   Invoke a registered skill
/<templatename> Expand a prompt template
```

That last category is important: **anything not built in is reachable through the same `/` syntax**. Extensions register their own commands and templates and they appear here without ceremony.

---

## 5. System prompt assembly

Pi assembles its system prompt from layered files, with override and append semantics.

```
priority order (highest to lowest):

  .pi/SYSTEM.md                 project-level full replacement
  ~/.pi/agent/SYSTEM.md         global-level full replacement
  default system prompt         shipped in the binary

  + .pi/APPEND_SYSTEM.md        project-level appended after replacement target
  + ~/.pi/agent/APPEND_SYSTEM.md  global appended after replacement target

  + AGENTS.md / CLAUDE.md       walked from cwd upward to root, all concatenated
  + skill files                 all matching files concatenated
```

This is the prompt-assembly pattern from Lecture 21 (OpenClaw System Prompt Architecture) made even simpler: a small fixed default, replaceable, with append hooks for project-specific guidance, plus walk-up context files (`AGENTS.md`, `CLAUDE.md`) the way every recent agent has settled on.

The `CLAUDE.md` filename being honored alongside `AGENTS.md` is a deliberate compatibility move: a workspace already configured for Claude Code drops cleanly into Pi without renaming files.

---

## 6. The session model

Sessions are JSONL files stored under `~/.pi/agent/sessions/`, organized by working directory. The on-disk shape is the simplest possible event log:

```jsonl
{"id":"a","parentId":null,"role":"user","content":"refactor this fn"}
{"id":"b","parentId":"a","role":"assistant","content":"…"}
{"id":"c","parentId":"b","role":"toolResult","name":"read","data":"…"}
{"id":"d","parentId":"c","role":"assistant","content":"…"}
{"id":"e","parentId":"a","role":"user","content":"actually try a different approach"}
{"id":"f","parentId":"e","role":"assistant","content":"…"}
```

Each line carries an `id` and a `parentId`. Most lines extend the active branch (their `parentId` is the previous line's `id`). When the user takes a side-quest, a new line is written whose `parentId` is **not** the previous line — it is some earlier ancestor. The result is a tree, in one file:

```
                    a (user: refactor this fn)
                   / \
                  b   e (user: actually try a different approach)
                  |   |
                  c   f
                  |
                  d
```

Both branches `[a → b → c → d]` and `[a → e → f]` live in the same JSONL file. The active branch is determined by which leaf the runtime considers current.

The format is **Lecture 24b's "session is source of truth" principle** at minimum cost: one append-only file, one `parentId` field, no separate database. This is also why replay works trivially: re-read the file, fold events along whichever branch you select, derive the context window from there.

---

## 7. `/tree`, `/fork`, and `/clone` — three different things

The naming matters. From the README:

| Command | Effect | Files affected |
|---|---|---|
| `/tree` | Jump to any point in the current session's tree, switch branches in place | One file (the current session); just changes the active leaf |
| `/fork` | Create a **new session file** from a previous user message; copies the active path up to that point; selected prompt placed in the editor for modification | Two files (original untouched; new session created) |
| `/clone` | Duplicate the current active branch into a **new session file** at the current position; full active-path history kept; opens with empty editor | Two files (original untouched; new session created) |

The mental model:

- `/tree` is *navigation* within one session.
- `/fork` is *what-if from a past prompt*, with that prompt loaded for editing — you are saying "I want to ask this differently."
- `/clone` is *snapshot at the current state into a new session* — you are saying "I want to take this conversation somewhere else without polluting the original."

These three primitives cover the realistic side-quest design space:

- Try a different approach to an old prompt → `/fork` from that prompt.
- Go investigate something orthogonal without losing the main thread → `/clone`, work in the clone, come back.
- Move between branches that already exist → `/tree`.

A linear-only session log can do none of these without additional machinery.

---

## 8. The Extension API

Extensions live in two well-known locations:

```
~/.pi/agent/extensions/    global, available everywhere
.pi/extensions/            project-local
```

A third path is "pi packages" (npm packages discoverable through the normal Node resolution chain). Extensions can be disabled per-run with `--no-extensions` or explicitly loaded with `-e`.

Each extension is a TypeScript module with a default export: a function that takes an `ExtensionAPI` object and registers everything it wants:

```typescript
export default function (pi: ExtensionAPI) {
  pi.registerTool({
    name: "deploy",
    label: "Deploy",
    description: "Deploy the current branch to staging",
    parameters: Type.Object({
      target: Type.String({ description: "staging | production" })
    }),
    execute: async (toolCallId, params, signal, onUpdate) => {
      // ...
    }
  });

  pi.registerCommand("stats", {
    description: "Show cost and token usage for this session",
    handler: async (ctx) => { /* ... */ }
  });

  pi.on("tool_call", async (event, ctx) => {
    // observe every tool call, with full context
  });
}
```

What `ExtensionAPI` exposes (from the README):

- **`registerTool()`** — add an LLM-visible tool (Lecture 24 §2.1 dispatch surface)
- **`registerCommand()`** — add a slash command (out-of-context, user-visible)
- **`on(event, handler)`** — subscribe to runtime events such as `tool_call`
- Custom UI components, status lines, headers, footers, in-place editors
- Async extension factories (so extensions can do startup work)

Two structural notes on this API:

1. **The extension surface is the same as Lecture 24's two extension surfaces** (in-context LLM tools vs out-of-context TUI). `registerTool()` is the former; `registerCommand()` and the UI hooks are the latter. The decision rule from Lecture 24 §5.3 applies directly.

2. **Event handlers are first-class.** This is what makes "the agent extends itself" practical: an extension can observe tool calls, react to them, modify them, log them, gate them. The same hook surface that audit / security / telemetry would use is the one extensions see.

---

## 9. The Agent runtime — `pi-agent-core`

If the coding-agent CLI is the visible surface, `pi-agent-core` is the substrate. It is what OpenClaw, a Telegram bot, or your own front-end consumes when they want Pi-style agent behavior without the CLI.

The exposed shape:

```typescript
agent.state.systemPrompt  = "..."
agent.state.model         = getModel(...)
agent.state.tools         = [tool1, tool2]
agent.state.messages      = [...]    // top-level array; copied on assign

await agent.waitForIdle()
agent.abort()
agent.reset()

// read-only:
agent.isStreaming
agent.streamingMessage
agent.pendingToolCalls
agent.errorMessage
```

Tool definitions use a TypeBox-based interface:

```typescript
const readFileTool: AgentTool = {
  name:           "read_file",
  label:          "Read File",
  description:    "Read a file's contents",
  parameters:     Type.Object({ path: Type.String() }),
  executionMode:  "sequential",   // or default parallel
  execute: async (toolCallId, params, signal, onUpdate) => {
    // tool body
    // returns a result, or throws on failure
    // may include `terminate: true` to skip the next LLM call
  }
};
```

Two specifics that matter:

- **`executionMode: "sequential"`** — opt-out of parallel tool execution for tools that must not run concurrently. By default the runtime can execute non-conflicting tool calls in parallel.
- **`terminate: true`** — a tool result can flag the agent to skip the automatic follow-up LLM call. Useful for tools whose result is the answer (e.g., a `commit` tool that succeeded; nothing more to say).

Custom message types are added by **declaration merging** on a `CustomAgentMessages` interface, then filtered out of the LLM-bound subset by `convertToLlm()`. This is the "custom messages in the session log" pattern from §6 made operational at the type level.

---

## 10. Hot reload — what actually reloads

Pi's hot-reload story is more specific than "edit and save."

**`/reload`** is a manual command that re-reads:

- keybindings
- extensions
- skills
- prompts
- context files (`AGENTS.md` / `CLAUDE.md`)

**Themes hot-reload automatically** — modify the active theme file and the change applies immediately, no `/reload` needed.

**What does NOT reload at runtime:**

- the underlying agent runtime / TUI itself (requires process restart)
- already-running tool executions (sequential ones especially)
- model API state (cache prefixes etc.)

The structural lesson: hot reload in an agent runtime is *not* "every layer is hot-reloadable." It is specifically the *configuration and extension layers* that are hot-reloaded; the runtime core stays stable. This is exactly the right line to draw — it gives you the "agent extends itself" loop without inviting the bug class where a tool is half-reloaded mid-execution.

---

## 11. Configuration paths

| Path | Purpose | Scope |
|---|---|---|
| `~/.pi/agent/settings.json` | Settings (theme, thinking level, transport, etc.) | Global |
| `.pi/settings.json` | Settings overrides | Project |
| `~/.pi/agent/SYSTEM.md` | System prompt full replacement | Global |
| `.pi/SYSTEM.md` | System prompt full replacement | Project |
| `~/.pi/agent/APPEND_SYSTEM.md` | System prompt append | Global |
| `.pi/APPEND_SYSTEM.md` | System prompt append | Project |
| `~/.pi/agent/extensions/` | Extensions | Global |
| `.pi/extensions/` | Extensions | Project |
| `~/.pi/agent/sessions/` | Session JSONL files (organized by cwd) | Per-cwd within global |
| `~/.pi/agent/skills/` | Skills | Global |
| `.pi/skills/` | Skills | Project |
| `AGENTS.md`, `CLAUDE.md` | Context files | Walked up from cwd |

The whole config tree is overridable via the `PI_CODING_AGENT_DIR` environment variable, which is useful for testing and for running multiple isolated Pi instances.

The convention is the well-trodden one: a hidden dotted directory (`.pi`) in the project, a corresponding `~/.pi/agent/` for globals, and a single env var for everything-else cases. No surprises.

---

## 12. The "No MCP" stance, made concrete

The README is direct: *"No MCP. Build CLI tools with READMEs (see Skills), or build an extension that adds MCP support."*

The structural argument was previewed in Lecture 24 §2.1 and Lecture 27 §14.6. In Pi-specific terms:

- Pi expects to **mutate the tool surface mid-session** (extensions register tools; skills are loaded on demand; `/reload` re-reads everything).
- MCP, as deployed across most providers, expects the tool catalog to be **stable for the session** so it can sit in the cached prompt prefix.
- These two are in direct tension. Pi resolves it by not adopting MCP at the protocol level.

What you do instead:

1. **Build CLI tools and put a README on them.** Pi's `bash` tool reaches them by name; the README is what teaches the model how to use them. Skills make this idiomatic — a skill is a directory with a few markdown files and supporting scripts.
2. **Build an MCP-bridge extension** if you genuinely need MCP. The extension can spawn `mcporter` (or similar), translate calls, register the methods as Pi tools dynamically, and clean up on exit.

The lesson is broader than Pi: **a harness's tool-surface mutability and its protocol choice are coupled architectural decisions**. You cannot pick "tools defined as MCP, mounted at session start" and "agent extends itself by writing tools" without paying for the conflict somewhere.

---

## 13. Keyboard shortcuts as a UX primitive

Pi ships a keyboard-first TUI. The shortcuts are not decoration; they are the actual interaction model.

Selected from the README:

| Key | Action |
|---|---|
| `Ctrl+C` | Clear editor (single press) |
| `Ctrl+C` × 2 | Quit |
| `Escape` | Cancel / abort |
| `Escape` × 2 | Open `/tree` |
| `Ctrl+L` | Open model selector |
| `Ctrl+P` / `Shift+Ctrl+P` | Cycle scoped models forward / backward |
| `Shift+Tab` | Cycle thinking level |
| `Ctrl+O` | Collapse / expand tool output |
| `Ctrl+T` | Collapse / expand thinking blocks |
| `Shift+Enter` | Multi-line editor (`Ctrl+Enter` on Windows Terminal) |
| `Tab` | Path completion |
| `Ctrl+V` | Paste images (`Alt+V` on Windows) |
| `Enter` | Queue steering message (mid-stream) |
| `Alt+Enter` | Queue follow-up message |
| `Ctrl+G` | Open external editor |

Three of these are agent-specific innovations worth calling out:

- **`Enter` queues a steering message during a running stream.** You do not have to abort; you tell the agent something while it is working and it picks the message up at the next safe point.
- **`Alt+Enter` queues a follow-up.** Same idea but applies after the current turn completes rather than mid-stream.
- **`Escape` × 2 jumps to `/tree`.** Branch navigation is one keystroke away from anywhere.

These all reflect the same design choice: **the human is in the loop continuously**, not just at turn boundaries. The TUI gives you the affordances to act mid-stream, and the agent runtime is built to accept those signals without breaking.

---

## 14. The pi-mono ecosystem

The repo names two sister projects worth knowing:

- **[`badlogicgames/pi-share-hf`](https://github.com/badlogicgames/pi-share-hf)** — publish a Pi session to Hugging Face. The pattern is "session as artifact": once you have a tree-structured event log, sharing it is just publishing the file.
- **[`earendil-works/pi-chat`](https://github.com/earendil-works/pi-chat)** — Slack/chat automation workflows on top of Pi. Pi is the runtime; this is one of several front-ends that demonstrate `pi-agent-core` is meant to be embedded.

The branding domain is `pi.dev`.

The 44.9k-star count and 212 releases at v0.73.0 (May 2026) suggest a project that ships frequently and has reached a substantial user base. For a learner: **the version cadence is itself a signal** that the architectural choices in this lecture are not theoretical — they have to survive contact with users in production weekly.

---

## 15. Mapping every Pi decision back to Lectures 24 and 24b

Pi is the most concrete public instantiation of the general harness theory in this course. Walking the mapping:

| Pi decision | Lecture 24 / 24b principle |
|---|---|
| Four built-in tools (read, write, edit, bash) | §2.1 dispatch surface; §6.5 too-many-tools anti-pattern |
| `~/.pi/agent/sessions/*.jsonl` append-only | Lecture 24b §1 session as source of truth |
| `id` + `parentId` tree | Lecture 24b §2 event sourcing for cognition |
| Custom message types via declaration merging | Lecture 24b §3 schema; §10 anti-pattern eliminated |
| `/reload` for extensions, themes auto-reload | §2.6 extensibility; §5 stateless interpreter pattern |
| `.pi/SYSTEM.md` overrides + `APPEND_SYSTEM.md` | §2.3 context construction; Lecture 21 prompt assembly |
| `registerCommand` (TUI) vs `registerTool` (LLM) | §5.1 / §5.2 / §5.3 the decision rule |
| `executionMode: "sequential"` opt-out | §2.4 planning and recovery; tool-call ordering |
| `terminate: true` in tool result | §2.4 turn-loop control |
| Walk-up `AGENTS.md` / `CLAUDE.md` | §2.3 context construction |
| `PI_CODING_AGENT_DIR` env var | §2.5 policy and permission isolation between instances |
| No MCP, with `bash`-and-CLI workaround | §2.1 dispatch boundary; §2.6 extensibility tradeoff |
| `/tree`, `/fork`, `/clone` | Lecture 24b §2 capabilities (branching is the unlocked one) |
| Mid-stream `Enter` to steer | §2.4 planning and recovery; human-in-the-loop |

This table is the answer to "should I copy Pi's design decisions into my own harness?" Yes, except the No-MCP one — that is contingent on whether you also adopt mid-session mutability. Adopt both or neither.

---

## 16. Hardware-track tie-in

For learners on the Jetson / edge AI track, Pi is structurally interesting in three specific ways:

**Minimal core, minimal cold start.** A four-tool harness has a smaller prompt-cache surface and a faster cold start than feature-bloated alternatives. On a Jetson AGX with limited unified memory (Lecture VLA Deployment on Edge GPUs §5), every ~10 KB of system prompt costs decode latency on the first turn. Pi's minimum is small enough that it does not dominate.

**Provider-agnostic SDK enables hybrid routing.** `pi-ai` abstracts model providers cleanly enough that one session can mix a remote Anthropic call with a local on-device model (vLLM, llama.cpp, ONNX Runtime, the VLA stack from `vla-deploy-jetson`). The session log captures provider metadata per turn so replay still works. This is the right primitive for the hybrid cloud-vs-edge deployment pattern.

**Tree-structured sessions as a multi-agent edge primitive.** Two robots branching from a shared parent session expresses coordination without forcing a master-slave ordering. A Pi-on-Jetson fleet has a natural way to do this without bolting orchestration on top.

The VLA deploy guide in Phase 4 / Track B / ML and AI / `vla-deploy-jetson` is the closest sibling: **same engineering posture (minimal substrate, hot-reloadable composition, edge-aware design), different target workload**.

---

## 17. Build it

Two concrete artifacts, in increasing difficulty.

**Beginner — write a Pi extension.** Pick a capability you actually want: a `/diff` command that shows the current uncommitted diff in a TUI overlay, a `/cost` command that surfaces session cost and tokens, a `/scratchpad` command that opens an external editor for a quick note appended to the session as a custom message. Register one slash command, one event handler, and (optionally) one LLM tool. Land it in `.pi/extensions/` in your own working tree.

**Intermediate — embed `pi-agent-core` in a non-CLI front-end.** Build a small Discord bot, Slack bot, or web chat that uses `@mariozechner/pi-ai` and `@mariozechner/pi-agent-core` directly. Implement the four-tool default (or a subset). Persist sessions to JSONL with the same `id` / `parentId` shape. The point: prove to yourself that the substrate is consumable independent of the CLI.

**Advanced — write your own minimal harness in a different language**, applying every principle from the §15 mapping table. Same four built-in tools, same JSONL session format with `parentId` tree, same two extension surfaces. You will know you have understood the architecture when your harness can host an extension written by a model, hot-reload it, and continue the same session afterward.

---

## Key takeaways

- pi-mono is a TypeScript monorepo of five packages: `pi-ai`, `pi-agent-core`, `pi-coding-agent`, `pi-tui`, `pi-web-ui`. The CLI is one product; the runtime is meant to be embedded.
- The default tool set is exactly four: `read`, `write`, `edit`, `bash`. Three more (`grep`, `find`, `ls`) are CLI-toggleable.
- Sessions are JSONL files keyed by `id` and `parentId` — a tree in one file. `/tree` navigates, `/fork` branches from a past prompt, `/clone` snapshots the active branch into a new session.
- Extensions live in `~/.pi/agent/extensions/` or `.pi/extensions/` and register through an `ExtensionAPI` that exposes both LLM-tool registration and TUI / slash-command registration. Event handlers (`pi.on("tool_call", ...)`) make extensions first-class observers.
- Hot reload is targeted, not universal: `/reload` re-reads keybindings, extensions, skills, prompts, and context files; themes hot-reload automatically; the runtime core does not reload.
- "No MCP" is a structural choice driven by Pi's mid-session mutability requirement, not a roadmap gap. Use CLI tools with READMEs, or write an extension to bridge MCP, or use `bash` to invoke `mcporter`.
- Custom message types (declaration-merged into `CustomAgentMessages`) are how extensions persist state into the same append-only session log. This is Lecture 24b's principle made operational.
- Mid-stream steering (`Enter` queues a message; `Alt+Enter` queues follow-up; `Escape × 2` opens `/tree`) is a UX primitive built on the assumption that the human is in the loop continuously, not only at turn boundaries.
- Every major Pi design decision maps back to a section of Lecture 24 (harness concerns) and Lecture 24b (event-sourced session). Pi is the cleanest public instance of those principles in shipping code.
- For hardware-track learners: minimal cores, provider-agnostic SDKs, and tree-structured sessions are exactly the substrate properties edge AI deployments need.

---

## References

### Primary sources

- pi-mono repository — [https://github.com/badlogic/pi-mono](https://github.com/badlogic/pi-mono)
- `packages/coding-agent` README — [https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/README.md](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/README.md)
- `packages/agent` README — [https://github.com/badlogic/pi-mono/tree/main/packages/agent](https://github.com/badlogic/pi-mono/tree/main/packages/agent)
- npm: `@mariozechner/pi-coding-agent` — [https://www.npmjs.com/package/@mariozechner/pi-coding-agent](https://www.npmjs.com/package/@mariozechner/pi-coding-agent)
- npm: `@mariozechner/pi-agent-core` — [https://www.npmjs.com/package/@mariozechner/pi-agent-core](https://www.npmjs.com/package/@mariozechner/pi-agent-core)
- npm: `@mariozechner/pi-ai` — [https://www.npmjs.com/package/@mariozechner/pi-ai](https://www.npmjs.com/package/@mariozechner/pi-ai)
- npm: `@mariozechner/pi-tui` — [https://www.npmjs.com/package/@mariozechner/pi-tui](https://www.npmjs.com/package/@mariozechner/pi-tui)

### Sister and consumer projects

- `badlogicgames/pi-share-hf` — session publishing to Hugging Face: [https://github.com/badlogicgames/pi-share-hf](https://github.com/badlogicgames/pi-share-hf)
- `earendil-works/pi-chat` — Slack / chat workflows on Pi: [https://github.com/earendil-works/pi-chat](https://github.com/earendil-works/pi-chat)
- OpenClaw — multi-channel agent platform consuming Pi as a runtime: [https://github.com/openclaw/openclaw](https://github.com/openclaw/openclaw)
- pi.dev — the project's primary domain.

### Context

- Armin Ronacher, *Pi: The Minimal Agent Within OpenClaw* (Jan 31, 2026) — the framing essay that introduced this lecture's subject.

### Curriculum cross-references

- [Lecture 21 - OpenClaw System Prompt Architecture](Lecture-21.md)
- [Lecture 24 - What Is an AI Agent Harness?](Lecture-24.md)
- [Lecture 24b - Session as Source of Truth](Lecture-24b.md)
- [Lecture 25 - OpenCoven Local Harness Substrate](Lecture-25.md)
- [Lecture 26 - OpenKnots Trustworthy Agent Interfaces](Lecture-26.md)
- [Lecture 27 - AI Agent Security Engineer Roadmap](Lecture-27.md)
- [Phase 4 / Track B / VLA Deployment on Edge GPUs](../../../../Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/5.%20Application%20Development/5.%20ML%20and%20AI/vla-deploy-jetson/Guide.md) — sibling minimal-substrate engineering posture for VLA workloads.

---

*Next: [Lecture 29 - Agent Skills: Workflow Discipline for Reliable Coding Agents](Lecture-29.md)*
