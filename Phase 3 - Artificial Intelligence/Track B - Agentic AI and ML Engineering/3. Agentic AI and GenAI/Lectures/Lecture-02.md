# Lecture 02 - What Is an AI Agent Harness? The Runtime Around the Model

**Course:** [AI Agent Development 2026](../Guide.md) | **Previous:** [Lecture 01](Lecture-01.md) | **Next:** [Lecture 03](Lecture-03.md)

---

A large language model on its own is a **stateless function**:

```text
prompt + tools spec  ->  text + tool calls
```

That is not an agent.

An **agent** appears when something around the model:

- decides which tools the model is allowed to see
- calls those tools when the model asks
- feeds the results back into the next turn
- decides when to stop, summarize, or hand off
- keeps a workspace, files, identities, and budgets straight
- enforces what the model is and is not allowed to touch

That "something around the model" is the **harness**.

This lecture defines the harness, lists what it owns, walks through three concrete production harnesses, and explains why hardware-track engineers should care.

---

## Learning objectives

By the end of this lecture, you should be able to:

1. Define what an AI agent harness is and why it is separate from the model.
2. List the six responsibilities a harness must own.
3. Explain why each responsibility cannot be left to the model.
4. Recognize the harness layer in Claude Code, Cursor, and OpenAI Codex.
5. Read a transcript and identify which actions came from the model and which came from the harness.
6. Reason about throughput, batching, and locality from a hardware-aware perspective when a harness drives an inference engine.
7. Identify common harness anti-patterns: the "everything in the prompt" trap, unsupervised tool use, context bloat, and hidden state.
8. Sketch a minimal harness for a project of your own.

---

## 1. Mental model: model is a CPU, harness is the OS

A model alone is closer to a **CPU** than to a computer.

A CPU executes instructions but cannot, by itself:

- decide which programs to load
- arbitrate access to disk, network, or GPU
- swap context when memory runs out
- enforce permissions
- recover from a fault
- keep state across reboots

An operating system does those things.

The same gap exists between a model and a useful agent:

```text
+----------------------------------------------------+
|                   user / product                   |
+----------------------------------------------------+
|                       harness                      |   <- this lecture
|   tools | memory | context | planning | policy ... |
+----------------------------------------------------+
|                       model                        |
+----------------------------------------------------+
```

The model reasons.
The harness runs.

If your product behavior is unreliable, the cause is **almost always in the harness**, not in the model weights.

---

## 2. The six things a harness owns

A serious harness owns six concerns. Skip any of them and the system stops being usable in production.

```text
1. Tool dispatch          (the device-driver layer)
2. State and memory       (RAM, files, sessions)
3. Context construction   (what fits in the prompt this turn)
4. Planning and recovery  (turn loop, retries, sub-agents)
5. Policy and permission  (what tools, paths, networks are allowed)
6. Extensibility          (skills, MCP servers, plugins, channels)
```

Each one shows up as code you have to write or buy.

### 2.1 Tool dispatch

The model emits a structured tool call. The harness must:

- validate the schema
- resolve which implementation to run
- execute it (in-process, subprocess, MCP server, remote RPC)
- capture stdout, stderr, exit codes, return values
- truncate noisy output without losing the signal
- return a normalized tool result the model can read

Without this layer the model can ask for tools but **nothing happens**.

### 2.2 State and memory

Three time horizons need separate machinery:

- **Turn state.** The current tool call queue, partial outputs, locks.
- **Session state.** Conversation history for this run, plus working memory of files touched and decisions made.
- **Cross-session memory.** Persistent facts the agent carries to the next conversation: user profile, project conventions, prior decisions.

Models do not have memory; the **harness fakes it** by stuffing prior state into the next prompt or by exposing it as a tool.

### 2.3 Context construction

Every turn the harness assembles a fresh prompt:

- system prompt and identity
- tool catalog (full, compact, or none)
- bootstrap files and project context
- skill descriptions
- memory entries judged relevant
- conversation transcript, possibly compacted
- provider-specific overlays (cache markers, beta headers)

This is the **most context-window-sensitive job** in the whole system.

A harness that always sends the full transcript will **bankrupt you and degrade output**. A harness that compacts blindly will silently drop load-bearing detail.

See [Lecture 37 - System Prompt Architecture](Lecture-37.md) for an in-depth look at one production approach.

### 2.4 Planning and recovery

The harness runs the loop:

```text
loop:
  build prompt
  call model
  if model returns tool calls -> dispatch, capture results, continue
  if model returns text       -> stream to user, decide if turn is done
  if error                    -> classify, retry or surface
  if budget exceeded          -> stop with partial result
```

It also decides:

- whether to spawn a sub-agent for parallel work
- when to require human approval
- how to recover from a malformed tool call
- when to abandon a plan and replan

### 2.5 Policy and permission

The model has **no conscience and no awareness of impact**.

The harness enforces:

- which tools are even visible
- which file paths are readable, writable, or denied
- which network destinations are reachable
- which commands need user confirmation before running
- which secrets are reachable and which are masked
- which actions get audit-logged

This must be **runtime enforcement**, not advisory text in the system prompt. Anything you only ask the model to do, the model will eventually skip.

See [Lecture 24 - Runtime Discipline & AI Runtime Security](Lecture-24.md).

### 2.6 Extensibility

**Real agents grow.** Users add skills, organizations add MCP servers, products add channels.

The harness needs a stable plug-in surface so:

- new tools land without rewriting the loop
- new skills become discoverable to the model
- new transports (chat UI, terminal, IDE, voice) reuse the same core

If extensions can only be added by editing the core, the harness will **calcify within months**.

---

## 3. Three real-world harnesses, side by side

The clearest way to see what a harness is is to look at three of them.

### 3.1 Claude Code

A **terminal harness** wrapping the Anthropic Messages API.

Owns:

- `Read`, `Edit`, `Write`, `Glob`, `Grep`, `Bash`, sub-`Agent` tools
- a permission system that prompts on first use of risky shell commands
- context compaction once the conversation approaches the model limit
- skills and MCP servers as the extensibility layer
- a project-scoped CLAUDE.md auto-loaded as bootstrap context
- background tasks, scheduled tasks, and hooks

The model **never opens a file or runs a process directly**. The Claude Code harness does.

### 3.2 Cursor

An **IDE harness** wrapping multiple model providers.

Owns:

- editor-aware tools (multi-file edits, codebase search, lint integration)
- `.cursor/rules/` files as runtime-injected guidance
- a Skills system for repeatable domain workflows
- MCP for external tool servers
- inline diffs and an apply/revert loop tied to the editor's UI

The harness here is the **editor itself**. Strip away the editor and there is no agent.

### 3.3 OpenAI Codex (CLI)

A **coding-task harness** wrapping OpenAI models.

Owns:

- repo indexing for large codebases
- a sandboxed shell for command execution
- patch-style edits applied to the working tree
- approval modes for risky actions
- a periodic context-cleanup pass

Same shape, different defaults: same six concerns, tuned for non-interactive coding tasks.

### 3.4 What is the same across all three?

```text
              Claude Code     Cursor          Codex CLI
tools         shell + files   editor + tools  shell + patches
memory        CLAUDE.md +     rules + chat    repo index +
              session         history         scratch
context       compaction +    rule injection  cleanup pass
              skills
planning      sub-agents      single loop +   approval modes
              + hooks         apply
policy        per-tool        rule files +    approval modes
              prompts                         sandbox
extension     MCP + skills +  MCP + rules +   plugins
              hooks           skills
```

Different surface, **same six responsibilities**.

---

## 4. Why hardware-track engineers must care

This roadmap is about hardware. So why a lecture on software harnesses?

Because **the harness is what hits your hardware**.

When you build:

- a Jetson-hosted edge inference service
- an FPGA accelerator under a CPU shim
- a private vLLM cluster on H100s
- a CUDA kernel optimized for batched decode

your customer is **almost certainly a harness**, not a human typing.

Things only a harness can tell you, but that change your hardware design:

- **Batch shape.** A harness that fans out parallel sub-agents creates large concurrent batches. A single-loop harness sends one request at a time. Your scheduler and KV-cache layout depend on this.
- **Prompt cache reuse.** Harnesses that keep system prompts stable across turns can use prompt caching for 5-10x throughput. Harnesses that mutate the system prompt every turn cannot.
- **Tool latency budget.** The harness picks how long it will wait for a tool before timing out. That decides whether your hardware tool back-end has 200 ms or 30 s of headroom.
- **Streaming vs full-response.** A harness driving a chat UI streams; a harness driving a CI job buffers. Memory pressure on your inference server is different in each case.
- **Locality.** A "local harness substrate" (see [Lecture 03](Lecture-03.md)) wants its model on the same machine. A gateway harness multiplexes many users across a cluster. Edge vs datacenter design diverges from this point.

If you only think about FLOPs and bytes, you will **optimize for the wrong workload**.

---

## 5. A minimal harness in pseudocode

Strip away the production concerns and a harness fits in roughly 80 lines:

```python
class MinimalHarness:
    def __init__(self, model, tools, policy, memory):
        self.model = model
        self.tools = {t.name: t for t in tools}
        self.policy = policy
        self.memory = memory

    def run(self, user_input, max_turns=20, token_budget=200_000):
        history = self.memory.load_session()
        history.append({"role": "user", "content": user_input})

        for turn in range(max_turns):
            prompt = self.build_prompt(history)
            if self.token_count(prompt) > token_budget:
                history = self.compact(history)
                prompt = self.build_prompt(history)

            response = self.model.call(prompt, tools=self.visible_tools())

            if response.tool_calls:
                results = []
                for call in response.tool_calls:
                    if not self.policy.allow(call):
                        results.append(self.deny_result(call))
                        continue
                    results.append(self.dispatch(call))
                history.append({"role": "assistant", "content": response})
                history.append({"role": "tool", "content": results})
                continue

            history.append({"role": "assistant", "content": response.text})
            self.memory.save_session(history)
            return response.text

        raise RuntimeError("turn budget exceeded")

    def visible_tools(self):
        return [t.spec for t in self.tools.values() if self.policy.visible(t)]

    def dispatch(self, call):
        tool = self.tools[call.name]
        try:
            return {"ok": True, "data": tool(**call.args)}
        except Exception as e:
            return {"ok": False, "error": str(e)}
```

Notice what is **not** in the model:

- the loop itself
- tool dispatch
- token budgeting and compaction
- policy checks
- session persistence

All of that is the **harness**. The model only handles "given this prompt, produce text or tool calls."

---

## 6. Common harness mistakes

These appear in every team's first agent system.

### 6.1 Putting policy in the prompt

```text
"Never run rm -rf without asking the user first."
```

The model will obey 99 times. The **100th time it will not**.

Policy belongs in **dispatch**, not in prose.

### 6.2 Letting the transcript grow forever

Without compaction or summarization, the prompt **grows linearly with turn count**. Latency, cost, and degradation all rise together. After a few hours the agent becomes unusable and nobody knows why.

Build compaction in from day one, even a naive one.

### 6.3 Hidden state

If the harness mutates files, environment variables, or external services without recording the change in memory, the next turn's model will reason from a **stale view of the world**. It will then be "wrong" in confusing ways.

**Every side effect** should appear in the next prompt or be retrievable on demand.

### 6.4 No replay

A harness with no log of `(prompt, model output, tool calls, tool results)` per turn is **impossible to debug**. Treat the trace as a **first-class artifact**, not an afterthought.

### 6.5 Too many tools

Every tool spec **costs tokens and confuses tool selection**. A harness that exposes 80 tools at once will produce worse output than the same harness exposing 8 contextually relevant ones.

Skill systems exist to solve this: **load tools on demand**.

---

## 7. Build it: read your own harness

Pick the harness you use day to day (Claude Code, Cursor, Codex, Continue, Aider, your own). Find the answers to these questions before you build your own:

1. Where is the main turn loop? Trace one iteration.
2. How does it detect that the model wants to call a tool?
3. How does it dispatch the tool?
4. Where does it record the result?
5. What triggers context compaction, and what gets dropped?
6. Where are the permission checks? Are they advisory or enforced?
7. How is a session persisted across restarts?
8. What is the extension surface (MCP, plugins, skills, rules)?

If you cannot answer one of these for a harness you use every day, that is the gap to read source code into.

---

## 8. Ship it

Artifact: a one-page architecture sketch of a harness you have used or designed. It must label:

- the model boundary
- the tool dispatch path
- the memory store
- the context-construction step
- the policy enforcement points
- the extensibility surface

A reviewer should be able to point at any user-visible behavior of the agent and say which box was responsible. If they cannot, the diagram is incomplete.

---

## Key takeaways

- A model is a function. An agent is a model plus a harness.
- A harness owns six things: tool dispatch, memory, context, planning, policy, extensibility.
- Skip any one of them and the system fails in production.
- Claude Code, Cursor, and Codex are different surfaces over the same six responsibilities.
- Policy must be enforced at dispatch, not asked for in the prompt.
- Context construction is the most context-window-sensitive code in the system.
- Hardware engineers should care because the harness, not the user, is the actual workload that hits inference hardware.
- A minimal but correct harness is small. A production harness is mostly the things this lecture lists, written carefully.

---

## References

- bswen — *What Is an AI Agent Harness? The Operating System for Autonomous Coding Agents*: [https://docs.bswen.com/blog/2026-03-25-ai-agent-harness-explained/](https://docs.bswen.com/blog/2026-03-25-ai-agent-harness-explained/)
- Anthropic — Claude Code documentation: [https://docs.claude.com/en/docs/claude-code](https://docs.claude.com/en/docs/claude-code)
- Cursor — Rules and Skills documentation: [https://docs.cursor.com/](https://docs.cursor.com/)
- OpenAI — Codex CLI: [https://github.com/openai/codex](https://github.com/openai/codex)
- Model Context Protocol (MCP) specification: [https://modelcontextprotocol.io/](https://modelcontextprotocol.io/)
- [Lecture 24 - Runtime Discipline & AI Runtime Security](Lecture-24.md)
- [Lecture 37 - OpenClaw System Prompt Architecture](Lecture-37.md)
- [Lecture 03 - OpenCoven: Local Harness Substrate](Lecture-03.md)

---

*Next: [Lecture 03](Lecture-03.md)*
