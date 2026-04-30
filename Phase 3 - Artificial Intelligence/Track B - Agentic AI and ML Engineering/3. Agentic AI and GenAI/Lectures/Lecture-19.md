# Lecture 19 - OpenClaw Case Study: The Agent Loop

**Course:** [Agentic AI & GenAI](../Guide.md) | **Previous:** [Lecture 18](Lecture-18.md) | **Next:** [Lecture 20](Lecture-20.md)

---

## Why this lecture exists

The agent loop is the core execution engine of an agent system.

Once you understand the loop, the surrounding pieces become easier to reason about:

- cron decides **when** to run
- hooks decide **where to intercept**
- tools decide **what actions are available**
- sessions decide **what state is visible**
- the agent loop decides **how one complete run executes**

This lecture uses OpenClaw's agent loop as the case study.

The short definition:

> an agent loop is one complete, controlled execution of an AI agent from input to actions to final output, while preserving consistency, safety, streaming, and session state

It is not only an LLM call.

It is a stateful, tool-using, streaming workflow pipeline.

---

## Learning objectives

By the end of this lecture you will be able to:

1. Explain the full lifecycle of an agent run.
2. Understand why OpenClaw serializes runs per session.
3. Describe the role of session locks, queues, streaming, tools, hooks, and persistence.
4. Explain how `agent`, `agent.wait`, CLI runs, cron, and hooks connect to the same core runtime.
5. Identify where failures, timeouts, compaction, and early exits happen.
6. Design an agent loop for your own production assistant.

---

## 1. What an agent loop really is

At a high level:

```text
input -> validate -> prepare context -> run model -> call tools -> stream output -> finalize -> persist
```

That looks simple, but each step hides real engineering concerns.

The loop must answer:

- which session is this run using?
- what model and auth profile should execute it?
- what tools are allowed?
- who can write to the transcript?
- how are partial replies streamed?
- what happens if a tool fails?
- what happens if context is too large?
- what final output should the user see?
- what gets saved for the next turn?

So a more accurate definition is:

> the agent loop is a deterministic, serialized, observable execution pipeline for AI agents with tools and memory

---

## 2. The full lifecycle

OpenClaw's loop can be understood as this sequence:

```text
1. Intake request
2. Validate parameters
3. Resolve session
4. Queue by session lane
5. Prepare workspace, skills, and bootstrap context
6. Acquire session write lock
7. Assemble prompt
8. Resolve model and auth
9. Run model with streaming
10. Execute tools when requested
11. Shape final reply
12. Persist transcript and metadata
13. Emit lifecycle end or error
```

The key point:

> the loop has a beginning, middle, and end that the system can observe

That is why OpenClaw can support live UI updates, `agent.wait`, cron run history, hooks, and debugging.

---

## 3. Entry points

The same loop can start from several places.

| Entry point | Example | Meaning |
|---|---|---|
| Gateway RPC | `agent` | enqueue an agent run and return quickly |
| Gateway RPC wait | `agent.wait` | wait for lifecycle completion of a specific run |
| CLI | `openclaw agent ...` | local command-line invocation |
| Cron | `openclaw cron run <job-id>` | scheduled job triggers an agent run |
| Hook | webhook, Gmail, message hook | event-driven trigger starts or modifies a run |

The important design choice is that these entry points should converge into one runtime path.

That gives the product consistent behavior:

- same tool policy
- same session semantics
- same logging
- same streaming events
- same failure handling

---

## 4. Intake and validation

At the boundary, the gateway validates the request and resolves run metadata.

It needs to determine:

- target agent
- session key or session id
- message body
- model and thinking overrides
- trace/verbose settings
- delivery or caller context
- timeout behavior

Then it can return an accepted response such as:

```json
{
  "runId": "run_...",
  "acceptedAt": "2026-04-29T12:00:00Z"
}
```

This reflects an async-first design:

> accepting a run is not the same as finishing a run

The Gateway can accept work, queue it, stream progress, and let clients wait separately.

---

## 5. Queueing and serialization

One of the most important OpenClaw design rules:

> only one agent loop should run per session lane at a time

Why this matters:

- prevents two runs from writing the same transcript concurrently
- avoids interleaved tool results
- keeps conversation history deterministic
- avoids duplicate or contradictory final replies

The mental model:

```text
session A: run 1 -> run 2 -> run 3
session B: run 1 -> run 2
session C: run 1
```

Each session lane is serial.

Different session lanes can still make progress independently, subject to global concurrency controls.

This explains why isolated cron and subagent sessions matter: they let background work proceed without blocking the user's main session lane.

---

## 6. Session write locks

Queueing handles logical order.

Locks handle actual file/state mutation.

OpenClaw protects transcript writes with a process-aware session write lock.

That matters because multiple processes may exist:

- Gateway process
- CLI process
- maintenance or doctor commands
- test workers
- automation scripts

The rule:

> any transcript write, rewrite, compaction, or truncation must acquire the same session write lock

By default, the lock should not be reentrant. Code that intentionally nests the same lock must opt in explicitly.

This is a strong production lesson:

> session state is shared mutable state, so it needs a real concurrency boundary

---

## 7. Session and workspace preparation

Before the model is called, the loop prepares the run environment.

Typical preparation includes:

- resolving workspace
- creating workspace if needed
- applying sandbox workspace root when sandboxed
- loading a skills snapshot
- resolving bootstrap context files
- preparing environment variables
- preparing the session manager

This is where "agent behavior" becomes more than a prompt.

The model sees the world through:

- workspace
- tools
- skills
- bootstrap context
- session history
- policy and runtime settings

Bad preparation leads to confusing agent behavior later.

---

## 8. Prompt assembly

The model does not see one string called "the prompt."

It sees assembled context.

OpenClaw-style prompt material includes:

```text
base system prompt
+ skills prompt
+ bootstrap context
+ session history
+ per-run overrides
+ hook-injected context
```

The runtime must also account for:

- model-specific token limits
- compaction reserve tokens
- truncation behavior
- tool schemas
- reasoning or thinking configuration

The important rule:

> prompt assembly is runtime behavior, not just prompt writing

A production system should be able to answer:

- what model saw which instructions?
- which skill snapshot was active?
- which bootstrap files were injected?
- which hook changed the prompt?
- why was context compacted?

---

## 9. Model execution

In OpenClaw's architecture, the embedded agent runner handles model execution.

Conceptually, this phase does:

- resolve provider and model
- resolve auth profile
- start the model request
- subscribe to model and tool events
- enforce abort and timeout behavior
- return final payloads and usage metadata

This is the "thinking" phase, but it is still runtime-controlled.

The model may:

- emit assistant text
- request tools
- stream reasoning chunks when supported
- hit idle timeout
- trigger model switch behavior
- fail with provider or network errors

The loop wraps that uncertainty in a stable contract.

---

## 10. Streaming events

OpenClaw streams more than final text.

A useful event model is:

| Stream | What it carries |
|---|---|
| `lifecycle` | `start`, `end`, `error` |
| `assistant` | text deltas, block replies, optional reasoning chunks |
| `tool` | tool start, update, result, end |

This is what makes live UI and channel updates possible.

For example:

```text
lifecycle:start
assistant:delta "I will check..."
tool:start read_file
tool:end read_file
assistant:delta "The file shows..."
lifecycle:end
```

Streaming matters because long-running agent work should not feel like a black box.

It also gives operators a way to debug where a run got stuck:

- no lifecycle start means intake/queue issue
- lifecycle start but no assistant deltas means model or prompt issue
- tool start with no tool end means tool execution issue
- lifecycle error means the runtime saw a terminal failure

---

## 11. Tool execution

When the model requests a tool, the loop becomes an action engine.

The basic tool path:

```text
model requests tool
-> emit tool start
-> validate tool call
-> run tool
-> sanitize result
-> emit tool result
-> feed result back to model
```

Tool results should be:

- size-limited
- sanitized
- safe to persist
- safe to stream
- connected to the originating call

Some tools can send messages directly to users.

That introduces a reply-shaping issue:

if a messaging tool already sent the useful answer, the final assistant confirmation may be redundant.

OpenClaw tracks messaging tool sends so duplicate confirmations can be suppressed.

---

## 12. Hooks as interception points

Hooks let the product intercept the loop.

OpenClaw has internal Gateway hooks and plugin hooks. The important concept is that hooks can run at different lifecycle points.

Useful hook categories:

| Hook area | Example | Purpose |
|---|---|---|
| Model selection | `before_model_resolve` | choose or override provider/model |
| Prompt assembly | `before_prompt_build` | inject context or system prompt material |
| Reply control | `before_agent_reply` | claim, override, or silence a turn |
| Tool policy | `before_tool_call` | block or modify a tool call |
| Tool output | `after_tool_call`, `tool_result_persist` | audit or transform tool results |
| Messaging | `message_received`, `message_sending`, `message_sent` | route, cancel, or audit messages |
| Lifecycle | `agent_end`, `gateway_start`, `gateway_stop` | observe system state |
| Compaction | `before_compaction`, `after_compaction` | inspect summarization behavior |

Terminal hook decisions are important.

Examples:

```json
{ "block": true }
```

or:

```json
{ "cancel": true }
```

These stop lower-priority handling in their hook chain.

The production lesson:

> hooks are powerful because they can change runtime behavior, so their ordering and terminal semantics must be explicit

---

## 13. Reply shaping

At the end of a run, the runtime decides what is user-visible.

Final payload assembly may include:

- assistant text
- block replies
- tool summaries when verbose and allowed
- error messages when needed

Then the runtime applies cleanup rules:

- suppress exact silent tokens such as `NO_REPLY` or `no_reply`
- remove messaging-tool duplicate confirmations
- emit a fallback tool error reply if no renderable output remains and the user has not already seen a reply
- avoid replaying stale acknowledgement-only text when a better descendant result exists

This phase exists because models often produce text that is not the right final user output.

The agent loop must shape output into product behavior.

---

## 14. Persistence

After the run, the system writes:

- user message
- assistant message
- tool calls and tool results
- metadata
- usage information
- lifecycle state

Persistence must happen under the session write lock.

This protects the transcript from races and gives future turns a coherent history.

The practical rule:

> if it affects future context, persist it carefully

This is also why streaming and persistence are different concerns. A user can see partial output before the final transcript is fully committed.

---

## 15. `agent.wait`

The `agent` RPC starts a run.

The `agent.wait` RPC waits for a run to reach lifecycle `end` or `error`.

A wait result looks conceptually like:

```json
{
  "status": "ok",
  "startedAt": "2026-04-29T12:00:00Z",
  "endedAt": "2026-04-29T12:00:10Z"
}
```

or:

```json
{
  "status": "timeout"
}
```

Important distinction:

> `agent.wait` timeout does not necessarily stop the agent run

It only means the waiter stopped waiting.

This is the same concept as watching a background job: your terminal can time out while the job continues.

---

## 16. Compaction and retry

Agent context grows.

Eventually the transcript may become too large for the model context window or for the configured reserve budget.

When that happens, the loop may trigger compaction:

```text
detect context pressure
-> emit compaction event
-> summarize or rewrite context
-> retry the run
```

On retry, the runtime must reset in-memory buffers and tool summaries so output does not duplicate.

This matters because compaction is not just a storage task.

It affects:

- what the model sees
- what the user sees
- which messages remain detailed
- whether the rerun repeats old output

---

## 17. Timeouts and aborts

OpenClaw-style systems have multiple timeout layers.

| Timeout | What it affects |
|---|---|
| agent runtime timeout | maximum agent run duration |
| LLM idle timeout | aborts a model request when no chunks arrive |
| `agent.wait` timeout | how long the caller waits |
| cron outer timeout | scheduled-job-level control |

OpenClaw's documented defaults include:

- agent runtime default can be configured through `agents.defaults.timeoutSeconds`, with documented examples around long 48-hour runs
- `agent.wait` defaults to a short wait window
- LLM idle timeout can be configured separately
- cron-triggered runs may rely on outer cron control when no explicit LLM/agent timeout is set

The lesson:

> timeout semantics must say what gets cancelled and what only stops waiting

Without that distinction, operators misread system behavior.

---

## 18. Where runs can end early

An agent loop may end early because of:

- agent runtime timeout
- abort signal
- gateway disconnect
- RPC wait timeout
- model failure
- tool failure
- hook block or cancel decision
- compaction failure

Good runtime design still attempts to emit lifecycle events:

```text
lifecycle:error
```

That gives clients a final state and helps `agent.wait`, UI, cron, and logs agree about what happened.

---

## 19. How this connects to cron

Now the previous lecture on cron becomes easier.

Cron does not do the agent work itself.

Cron schedules the work:

```text
cron schedule
-> due job
-> agent RPC
-> agent loop
-> delivery/logging
```

So:

- cron answers **when**
- the agent loop answers **how**
- tools answer **what actions**
- hooks answer **where policy intervenes**
- sessions answer **what state**

This is the architecture pattern behind serious persistent agents.

---

## 20. Example: one OpenClaw-style run

Imagine a user sends:

> Check the repo and summarize today's failing tests.

The loop might behave like this:

```text
1. Gateway receives message.
2. Gateway resolves session key.
3. Run enters that session lane queue.
4. Session lock is acquired.
5. Workspace and skills are prepared.
6. Prompt is assembled with history and bootstrap context.
7. Model starts streaming.
8. Model calls a shell/test-inspection tool.
9. Tool result is sanitized and streamed.
10. Model writes a final summary.
11. Duplicate tool-send confirmations are suppressed.
12. Transcript and metadata are persisted.
13. lifecycle:end is emitted.
14. The chat channel sends the final response.
```

The user experiences one reply.

The system executed a controlled transaction-like runtime path.

---

## 21. Design exercise

Design an agent loop for a local engineering assistant.

Fill in this table:

| Area | Your design |
|---|---|
| Entry points | CLI, Web UI, Slack, cron |
| Session lane rule | one run at a time per session |
| Global queue | max 2 concurrent model runs |
| Write lock | required for transcript writes and compaction |
| Prompt inputs | base prompt, skills, repo context, session history |
| Tools | read, search, test runner, issue lookup |
| Tool policy | write and deploy require approval |
| Hooks | block deploy from unpaired channels |
| Streaming | lifecycle, assistant, tool |
| Final reply shaping | suppress `NO_REPLY`, remove duplicate confirmations |
| Timeout model | wait timeout separate from runtime timeout |
| Persistence | transcript, tool outputs, usage, run metadata |

The value of this exercise is that it forces you to treat the agent as infrastructure, not as a single API call.

---

## Key takeaways

- The agent loop is the execution engine of an agent product.
- It is a serialized, observable pipeline from input to persisted final state.
- Per-session queueing prevents transcript and tool races.
- Session write locks protect durable state across processes.
- Prompt assembly, model execution, tool execution, streaming, reply shaping, and persistence are separate runtime concerns.
- Hooks are policy and extension points inside the loop.
- `agent.wait` waits for lifecycle completion; it does not define the entire run.
- Cron triggers agent loops, but cron is not the agent loop.

---

## References

- OpenClaw agent loop: [https://openclaw.knidal.com/agent-loop](https://openclaw.knidal.com/agent-loop)
- Case-study source repo: [OpenClaw](https://github.com/openclaw/openclaw)
- OpenClaw concepts:
  - `docs/automation/cron-jobs.md`
  - `docs/cli/cron.md`
  - `docs/tools/subagents.md`
  - `docs/concepts/session.md`
  - `docs/reference/session-management-compaction.md`

---

*Next: [Lecture 20 - OpenClaw Case Study: Cron, Scheduled Agent Runs, and Automation Reliability](Lecture-20.md)*
