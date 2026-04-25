# Lecture 17 - OpenClaw Case Study: Multi-Agent Isolation, Workspaces, and Memory

**Track B - Agentic AI & GenAI** | [<- Lecture 16](Lecture-16.md) | [Next -> Lecture 18](Lecture-18.md)

---

## Why this lecture exists

As soon as you run more than one agent in one product, isolation becomes a core design problem.

You need to decide:

- what each agent can see
- where each agent stores state
- which files belong to which agent
- whether memories can cross between agents
- whether credentials are shared or separated

OpenClaw is a strong case study because it treats an agent as a **scoped brain** with its own:

- workspace
- state directory
- auth profiles
- session store

This lecture uses OpenClaw to teach an important production lesson:

> multi-agent systems are not only about orchestration. They are also about boundaries.

---

## Learning objectives

By the end of this lecture you will be able to:

1. Explain what an "agent" means in a real multi-agent runtime.
2. Understand why workspace isolation matters.
3. Explain the difference between workspace, state, and sessions.
4. Understand how memory becomes a boundary problem.
5. Design a clean multi-agent layout for a serious product.

---

## 1. What is one agent, really?

OpenClaw's multi-agent docs give a very practical answer.

One agent is not merely:

- one prompt
- one model
- one role string

One agent is a full scoped unit with its own:

- workspace
- rules and persona files
- auth profile
- state directory
- session history
- memory context

That is a much stronger definition than most tutorials use.

This matters because once you accept that definition, you stop designing agents as loose prompt templates.

You start designing them as isolated application units.

---

## 2. Workspace is not just a folder

OpenClaw's workspace docs are useful here.

The workspace is:

- the default working directory
- the place for agent bootstrap files
- the place for agent personality and instructions
- part of the agent's memory and identity

Important lesson:

> The workspace is not only storage. It is part of the agent's operating context.

Typical workspace files include:

- `AGENTS.md`
- `SOUL.md`
- `USER.md`
- `IDENTITY.md`
- `TOOLS.md`
- memory files

This is a useful pattern even outside OpenClaw.

It teaches that agent behavior should not live only in code. It should also have structured operating files.

---

## 3. Workspace is not the same as sandbox

This is one of the most practical lessons from OpenClaw.

The workspace is the default cwd.

It is **not automatically a hard sandbox**.

That means:

- relative paths may stay inside the workspace
- absolute paths may still escape unless sandboxing is enabled

This is exactly the kind of nuance students miss when they think:

> "I gave the agent a workspace, so it is isolated."

No.

Isolation requires explicit sandboxing or equivalent runtime controls.

This is a powerful teaching example because it forces students to distinguish:

- convenience boundary
- security boundary

Those are not the same thing.

---

## 4. State directory vs session store vs workspace

These three are easy to confuse.

### Workspace

The agent's home and operating context.

### State directory

The place for agent-specific runtime data such as auth profiles and configuration state.

### Session store

The place where conversation history and routing state live.

This separation is good engineering because it prevents everything from collapsing into one unclear folder.

When students build their own agents, they should copy this separation:

| Area | Purpose |
|---|---|
| workspace | human-editable operating files and agent context |
| state | runtime config, provider state, credentials references |
| sessions | transcripts, history, routing state |

That alone makes systems easier to debug and back up.

---

## 5. Multi-agent isolation is about avoiding collisions

OpenClaw warns against reusing the same `agentDir` across agents.

That warning teaches an important general lesson:

> if two agents share too much state, they stop being separate agents

Collisions can happen in:

- auth profiles
- sessions
- workspaces
- memory collections
- tool configuration
- file outputs

If you want separate agents, give them separate homes.

Example bad design:

```text
agent A and agent B
  -> same workspace
  -> same auth file
  -> same session store
```

At that point, you mostly have one agent with confusing labels.

Example better design:

```text
agent A
  -> workspace A
  -> auth A
  -> sessions A

agent B
  -> workspace B
  -> auth B
  -> sessions B
```

Now the system is actually structured.

---

## 6. Memory as a boundary problem

OpenClaw is especially useful here because it shows several memory ideas:

- normal session history
- long-term memory files
- active memory plugin
- cross-session search
- cross-agent search when explicitly configured

This is exactly how modern agent systems behave:

memory is not one thing.

There is:

- working memory
- session transcript memory
- long-term memory
- retrieved memory
- optional cross-agent memory

The production lesson is:

> every memory path should be explicit

If cross-agent memory is allowed, it should be configured intentionally.

If it is not intentional, it should not happen by accident.

---

## 7. Active memory as a design pattern

OpenClaw's `active-memory` concept is very educational.

The idea is simple:

Instead of waiting for the main agent to decide when to search memory, a bounded memory sub-agent runs before the main reply and surfaces relevant memory.

This teaches two important lessons.

### Lesson 1: memory can be proactive

Most students think memory means:

> save messages and search them later

Active memory shows a different pattern:

> memory can be a separate runtime component that improves the next reply before the main generation happens

### Lesson 2: helper agents should be bounded

OpenClaw's memory helper is not just "another free agent."

It is:

- optional
- scoped
- timed
- bounded

That is good design.

If you add helper agents, they should have tight scope and purpose.

---

## 8. Example: two agents, one host

Suppose you run two agents on one machine:

- `coding`
- `family`

Good design:

| Agent | Workspace | Sessions | Auth | Memory |
|---|---|---|---|---|
| coding | `workspace-coding` | `sessions-coding` | coding auth profile | coding memory only |
| family | `workspace-family` | `sessions-family` | family auth profile | family memory only |

Why this is good:

- coding tasks do not pollute personal memory
- personal messages do not appear in engineering sessions
- tool permissions can differ
- model providers can differ
- backups are easier

This is exactly the kind of layout serious products need.

---

## 9. Example config pattern

Here is a simple OpenClaw-style pattern:

```json5
{
  agents: {
    list: [
      {
        id: "coding",
        workspace: "~/.openclaw/workspace-coding"
      },
      {
        id: "family",
        workspace: "~/.openclaw/workspace-family"
      }
    ]
  }
}
```

Again, the point is not memorizing config syntax.

The point is understanding the design:

- each agent gets its own operating home
- each agent gets its own sessions
- each agent can have different auth and memory behavior

That is how you build multiple real agents in one runtime.

---

## 10. Isolation checklist

If you are building a multi-agent product, ask these questions.

### Workspace

- Does each agent have its own workspace?
- Are agent instruction files separate?

### Credentials

- Does each agent have its own auth profile?
- Are sensitive capabilities separated?

### Sessions

- Does each agent have its own session store?
- Can one agent accidentally read another agent's transcript?

### Memory

- Is cross-agent memory access explicit?
- Is it disabled by default?

### Tools

- Can each agent access only the tools it should use?
- Are dangerous tools excluded from low-trust agents?

This is what real multi-agent engineering looks like.

---

## 11. Design exercise

Design a three-agent system for a small startup:

- `support`
- `research`
- `ops`

For each one, define:

| Agent | Workspace | Tools | Memory rule | Risk note |
|---|---|---|---|---|
| support | customer support workspace | docs search, ticketing | no cross-agent memory | must avoid exposing internal ops data |
| research | analyst workspace | web search, notes | may read shared research library | external information risk |
| ops | automation workspace | deployment tools, logs | isolated | high-risk action space |

This exercise should make one thing clear:

multi-agent design is mostly about boundaries and responsibilities.

---

## Key takeaways

- In a real runtime, one agent is a scoped unit with its own workspace, state, auth, and sessions.
- The workspace is an operating context, not automatically a security sandbox.
- Workspace, state, and session storage should stay conceptually separate.
- Memory is a boundary problem, not only a retrieval feature.
- Cross-agent memory access should be explicit, not accidental.
- OpenClaw is a strong case study for how to structure serious multi-agent systems.

---

## References

- Case-study source repo: [OpenClaw](https://github.com/openclaw/openclaw)
- OpenClaw concepts:
  - `docs/concepts/multi-agent.md`
  - `docs/concepts/agent-workspace.md`
  - `docs/concepts/active-memory.md`
  - `docs/concepts/session.md`

---

*Next: [Lecture 18 - OpenClaw Case Study: Operating and Securing a Persistent Agent System](Lecture-18.md)*
