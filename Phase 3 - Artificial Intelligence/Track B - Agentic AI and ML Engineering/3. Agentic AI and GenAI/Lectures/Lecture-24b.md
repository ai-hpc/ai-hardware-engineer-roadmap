# Lecture 24b - Session as Source of Truth: Event-Sourced Agent State

**Course:** [Agentic AI & GenAI](../Guide.md) | **Previous:** [Lecture 24](Lecture-24.md) | **Next:** [Lecture 25](Lecture-25.md)

---

[Lecture 24](Lecture-24.md) listed "state and memory" as one of the six things a harness owns. That was a placeholder. This lecture takes that single concern and makes it the lecture, because almost every reliability bug in agent systems traces back to one mistake:

> treating the **context window** as if it were the **session**

A model context is a 200K-token sliding view over a conversation. A session is the durable record of what actually happened. They are not the same thing, and confusing them is the root of:

- generation lost on crash
- "the agent forgot what it was doing"
- impossible-to-replay incidents
- branching that silently loses state
- resume-after-restart that doesn't actually resume

This lecture defines the split, gives you an event schema, and walks through a `wake(sessionId)` protocol that lets a stateless harness pick up exactly where the previous instance died.

---

## Learning objectives

By the end of this lecture, you should be able to:

1. State the difference between a session and a context window in one sentence.
2. Explain why event sourcing is the correct architecture for agent state.
3. Design an append-only event schema for an agent runtime.
4. Implement a `buildContext(session, strategy)` function that derives the context window from the session log.
5. Sketch a `wake(sessionId)` recovery protocol.
6. Identify hidden in-runtime state that breaks crash recovery.
7. Persist a streaming generation as an event stream so partial output survives a gateway crash.
8. Reason about replay, branching, and audit when designing a session store.

---

## 1. The mental model: database vs query result

Two layers, very different lifetimes.

```text
+----------------------------------------------------+
|     SESSION   (ground truth, durable, append-only) |
|     -> message received                            |
|     -> tool invoked                                |
|     -> stream chunk                                |
|     -> generation complete                         |
+----------------------------------------------------+
                       |
                       v  buildContext(session, strategy)
+----------------------------------------------------+
|     CONTEXT   (ephemeral view, fits in 200K)       |
|     -> system prompt                               |
|     -> tool catalog                                |
|     -> compacted transcript                        |
|     -> retrieved memory snippets                   |
+----------------------------------------------------+
```

| Layer | Analogy | Lifetime | Source of truth? |
|-------|---------|----------|------------------|
| Session | Database table | Indefinite | Yes |
| Context | `SELECT` result | One model call | No |

The model only ever sees the context. The harness only ever writes to the session.

Once you internalize this, the rest of the lecture is mostly mechanical.

---

## 2. Why this is event sourcing

State is not stored. State is *derived* from a log of events.

This is exactly the pattern called **event sourcing** in the database community: the canonical record is an append-only stream of facts, and any view (current balance, current cart, current agent memory) is a fold over that stream.

```text
session_log = [event_0, event_1, event_2, ..., event_n]

current_state(t) = fold(session_log[0..t], reducer)
```

Apply that to an agent and you get:

- The session log is the canonical record.
- The context window is one possible projection of that log.
- A summary memory is another projection.
- A user-facing transcript is a third projection.
- Three different projections, one source of truth.

This unlocks four capabilities you cannot reasonably get any other way:

| Capability | Why event sourcing gives it to you |
|------------|------------------------------------|
| Replay | Re-run reducer on the log offline |
| Branching | Fork the log at event N, run two timelines |
| Resumability | After a crash, fold the log → derive state → continue |
| Auditing | The log is already the audit trail; nothing to bolt on |

If your agent gives you none of those today, you have a context-as-memory problem.

---

## 3. The event schema

A workable starting schema is six event types. Keep them small and stable; you cannot rewrite events later without breaking replay.

```jsonl
{"ts":"2026-05-05T01:00:00Z","type":"MESSAGE_RECEIVED","id":"evt_001","payload":{"role":"user","text":"summarize this PR"}}
{"ts":"2026-05-05T01:00:01Z","type":"LLM_CALLED","id":"evt_002","payload":{"model":"your-agent-model-id","prompt_tokens":12450,"context_strategy":"sliding_window_50"}}
{"ts":"2026-05-05T01:00:02Z","type":"GEN_START","id":"evt_003","payload":{"msg_id":"msg_42","stream":true}}
{"ts":"2026-05-05T01:00:02Z","type":"GEN_CHUNK","id":"evt_004","payload":{"msg_id":"msg_42","seq":0,"delta":"Looking at"}}
{"ts":"2026-05-05T01:00:02Z","type":"GEN_CHUNK","id":"evt_005","payload":{"msg_id":"msg_42","seq":1,"delta":" the diff"}}
{"ts":"2026-05-05T01:00:03Z","type":"TOOL_INVOKED","id":"evt_006","payload":{"call_id":"tc_7","name":"git_diff","args":{"ref":"HEAD~1"}}}
{"ts":"2026-05-05T01:00:04Z","type":"TOOL_RESULT","id":"evt_007","payload":{"call_id":"tc_7","ok":true,"data_ref":"blob://abc123","bytes":4200}}
{"ts":"2026-05-05T01:00:05Z","type":"GEN_COMPLETE","id":"evt_008","payload":{"msg_id":"msg_42","reason":"stop"}}
{"ts":"2026-05-05T01:00:05Z","type":"GEN_SENT","id":"evt_009","payload":{"msg_id":"msg_42","channel":"chat-ui"}}
```

Notes on this schema, all of which matter:

- **Append-only.** Once written, an event is never modified.
- **Monotonic ids.** Either ULIDs or a per-session counter. Required for ordering after a crash.
- **`ts` is wall-clock; ordering is by id.** Wall-clocks lie under load.
- **Large payloads go to a blob store, not the log.** Tool results, screenshots, audio: store the bytes elsewhere and put a `data_ref` in the event. The log stays small enough to scan and replay.
- **Two-phase send** for outputs: `GEN_COMPLETE` (model finished) is separate from `GEN_SENT` (user / channel actually received). Without this split you cannot tell, after a crash, whether you owe the user a duplicate or nothing.

JSONL on disk is fine to start. Move to a database when you need queries across sessions or when scale demands it. The schema does not change; only the storage does.

---

## 4. The harness becomes a stateless interpreter

Once events are durable, the harness collapses to a small loop:

```python
def step(session_id):
    log = session_store.load(session_id)
    if is_terminated(log):
        return

    context = build_context(log, strategy="sliding_window_50")
    response = model.call(context, tools=visible_tools(log))

    session_store.append(session_id, {"type": "LLM_CALLED", ...})

    if response.tool_calls:
        for call in response.tool_calls:
            session_store.append(session_id, {"type": "TOOL_INVOKED", ...})
            result = dispatch(call)
            session_store.append(session_id, {"type": "TOOL_RESULT", ...})
        return  # caller decides whether to step again

    # streaming text generation
    session_store.append(session_id, {"type": "GEN_START", ...})
    for delta in response.stream:
        session_store.append(session_id, {"type": "GEN_CHUNK", ...})
        channel.send(delta)
    session_store.append(session_id, {"type": "GEN_COMPLETE", ...})
    channel.flush()
    session_store.append(session_id, {"type": "GEN_SENT", ...})
```

Things to notice:

- The function takes a session id, not a state object. **Nothing is held in process memory between turns.**
- Every observable side effect is preceded or followed by an event write.
- After a crash, this exact code can be re-entered with the same session id and recover.

This is what people mean by "stateless harness over an event log."

---

## 5. The `wake(sessionId)` protocol

`wake` is the recovery procedure run by every harness instance on startup, and by any cron / scheduler / external trigger that wants to resume a session.

```text
wake(sessionId):
  1. log    = session_store.load(sessionId)
  2. last   = last_event(log)
  3. switch on last.type:
       MESSAGE_RECEIVED        -> step(sessionId)               # never got to model
       LLM_CALLED              -> step(sessionId)               # crashed before generation
       TOOL_INVOKED            -> reissue_tool_or_fail(last)    # tool may have run
       TOOL_RESULT             -> step(sessionId)               # safe to continue
       GEN_START / GEN_CHUNK   -> resume_or_replace(last)       # see below
       GEN_COMPLETE            -> redeliver_if_unsent(last)
       GEN_SENT                -> done; idle
       SESSION_TERMINATED      -> noop
```

The two interesting branches are `TOOL_INVOKED` and `GEN_*`.

### 5.1 Tool-call recovery

If the harness died after writing `TOOL_INVOKED` but before `TOOL_RESULT`, you do not know whether the tool ran. Three correct options, in order of preference:

1. **Idempotent tools.** Every tool call carries a `call_id`. Tool implementation either re-runs safely or detects the duplicate and returns the prior result. This is the only option that scales.
2. **Compensating action.** If the tool was non-idempotent (sent a message, charged a card), record `TOOL_FAILED_UNCERTAIN` and surface it to the user.
3. **Refuse to recover automatically.** Mark the session needs-human-attention.

Never silently retry a non-idempotent tool. Once is better than twice.

### 5.2 Streaming-generation recovery (the OpenClaw #40712 case)

The user asked the model. The model started streaming. Three chunks landed in the log. Then the gateway crashed. What now?

Two strategies:

- **Resume.** Re-call the model with the chunks-so-far as a prefilled assistant turn and ask it to continue. Works when the provider supports prefill / continuation. Cheaper, faster, but the continuation may diverge stylistically from what the user already saw.
- **Replace.** Discard the partial chunks, mark them invalidated, regenerate from scratch. Always works. Costs more tokens. The user sees the response start over.

Either is correct. Whichever you pick, the choice is visible in the log:

```jsonl
{"type":"GEN_RESUMED","payload":{"msg_id":"msg_42","strategy":"resume","prior_seq":2}}
```

The wrong strategy is to do nothing — leave a half-emitted message in the channel and never finish it. That is what "context as memory" produces.

---

## 6. Two anti-patterns that this architecture eliminates

### 6.1 Context as memory

```python
class BadHarness:
    def __init__(self):
        self.history = []   # this is the bug
    def step(self, msg):
        self.history.append({"role":"user","content":msg})
        r = model.call(self.history)
        self.history.append({"role":"assistant","content":r})
        return r
```

`self.history` lives in process memory. Restart the process and it is gone. Run two replicas behind a load balancer and they disagree. Hot-reload code and you lose the conversation. Every long-lived agent system that starts this way eventually gets paged for it.

The fix is not "persist `self.history` to disk." The fix is to **not have `self.history`**. Make the harness load from the session log on every step.

### 6.2 Hidden in-runtime state

If the harness mutates anything that is not in the session log — a counter, a feature flag, a tool's stateful client — the next instance reasons from a stale view of the world.

The rule: every fact the model will reason about on the *next* turn must be either in the session log or derivable from a tool the model can call. Anything else is hidden state and will eventually surprise you.

---

## 7. Build it

Pick one of these and finish it. The point is to feel the architecture, not to build a product.

**Beginner.** Modify a minimal harness (the one in [Lecture 24, §5](Lecture-24.md)) to write a JSONL log of the six event types to disk. Kill the process mid-conversation. Restart it pointed at the same log file. Confirm the conversation continues as if nothing happened.

**Intermediate.** Add streaming. Kill the process between `GEN_START` and `GEN_COMPLETE`. Implement `wake(sessionId)` with the **replace** strategy. Verify the user sees a clean restart of the response, not a half-message.

**Advanced.** Implement tool-call recovery with idempotent tools. The tool's first action is to look up its own `call_id` in a side log; if found, return the prior result without re-running. Kill the process between `TOOL_INVOKED` and `TOOL_RESULT` and confirm the tool runs exactly once.

---

## 8. Ship it

Artifact: a session log of one real conversation, plus a one-page write-up answering:

1. Which event types did your harness emit?
2. Which payloads went inline and which went to a blob store?
3. What `buildContext` strategy did you use? Show the function.
4. What happens to the log when the user starts a "new chat"? (Hint: it should not delete anything.)
5. Did you handle tool idempotency? How?
6. Show one timestamped sequence where the harness crashed and `wake(sessionId)` recovered.

A reviewer should be able to take your log, replay it through your `step` function, and reproduce the user's experience byte-for-byte. If they cannot, the harness is still hiding state somewhere.

---

## 9. Hardware-track tie-in

Why this lecture lives in a hardware roadmap:

- **Inference batching loves stateless harnesses.** A stateless step function can be invoked from any worker in a pool, which lets your inference back-end batch across sessions instead of serializing within one. KV-cache reuse becomes a per-request decision, not a per-process one.
- **Edge resumability matters.** A Jetson at a remote site that loses power overnight should resume mid-task on boot. Without an event-sourced session that is impossible; with one it is a `wake(sessionId)` on the systemd unit.
- **Audit beats observability for safety-critical work.** When an agent acts on hardware (a robot arm, a power switch, a vehicle), you want the canonical record to be the event log, not OpenTelemetry traces of a now-dead process.

---

## Key takeaways

- Session and context window are different layers. Session is the database; context is a query result.
- The session is append-only. State is derived, not stored.
- This is event sourcing applied to cognition. The same engineering discipline that gives you replayable banking systems gives you resumable agents.
- A workable schema starts at six events: `MESSAGE_RECEIVED`, `LLM_CALLED`, `TOOL_INVOKED`, `TOOL_RESULT`, `GEN_START` / `GEN_CHUNK`, `GEN_COMPLETE`, `GEN_SENT`. Large payloads go to a blob store with a reference in the event.
- `GEN_COMPLETE` and `GEN_SENT` must be separate events. Without that split, crash recovery cannot tell whether the user already saw the response.
- The harness becomes a stateless interpreter. Every step loads the log, derives the context, runs the model, and appends new events.
- `wake(sessionId)` is the universal recovery and resume entry point. Cron, retries, and crash recovery all use it.
- The "streaming output lost on gateway crash" bug is not a streaming bug. It is a context-as-memory bug. Persist generation as events and the bug disappears.
- Tool-call recovery requires idempotent tools or human escalation. Never silently retry a non-idempotent tool.
- Two anti-patterns to root out: in-process history, and any side effect not represented in the log.

---

## References

- bswen — *What Is an AI Agent Harness?*: [https://docs.bswen.com/blog/2026-03-25-ai-agent-harness-explained/](https://docs.bswen.com/blog/2026-03-25-ai-agent-harness-explained/)
- Martin Fowler — *Event Sourcing*: [https://martinfowler.com/eaaDev/EventSourcing.html](https://martinfowler.com/eaaDev/EventSourcing.html)
- Greg Young — *CQRS and Event Sourcing* (talk): [https://www.youtube.com/watch?v=JHGkaShoyNs](https://www.youtube.com/watch?v=JHGkaShoyNs)
- Anthropic — Messages API and streaming: [https://docs.claude.com/en/api/messages-streaming](https://docs.claude.com/en/api/messages-streaming)
- Model Context Protocol — Tool spec: [https://modelcontextprotocol.io/specification](https://modelcontextprotocol.io/specification)
- [Lecture 13 - Runtime Discipline & AI Runtime Security](Lecture-13.md)
- [Lecture 14 - Deterministic Startup for AI Agent Systems](Lecture-14.md)
- [Lecture 19 - OpenClaw Agent Loop](Lecture-19.md)
- [Lecture 24 - What Is an AI Agent Harness?](Lecture-24.md)
- [Lecture 25 - OpenCoven: Local Harness Substrate](Lecture-25.md)

---

*Next: [Lecture 25 - OpenCoven Case Study: Agent-Native Workspace and Local Harness Substrate](Lecture-25.md)*
