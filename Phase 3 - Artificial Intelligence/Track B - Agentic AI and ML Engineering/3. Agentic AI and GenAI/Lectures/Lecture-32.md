# Lecture 32 - OpenClaw Case Study: Channels, Routing, and Session Design

**Course:** [AI Agent Development 2026](../Guide.md) | **Previous:** [Lecture 31](Lecture-31.md) | **Next:** [Lecture 33](Lecture-33.md)

---

## Why this lecture exists

Once you move beyond a single chat window, agent design becomes a **routing problem**.

You must decide:

- which inbound message goes to which agent
- which session should hold the context
- when two messages should share state
- when two users must be isolated
- how replies return to the correct channel

OpenClaw is a strong example because it makes these decisions explicit.

This lecture uses OpenClaw to teach one of the most important practical lessons in agent systems:

> session design is product design

If session boundaries are wrong, the whole agent experience becomes **unsafe or confusing**.

---

## Learning objectives

By the end of this lecture you will be able to:

1. Explain the difference between channel, account, agent, and session.
2. Understand why routing rules are a first-class part of agent design.
3. Explain the session-key idea in simple terms.
4. Decide when DMs should share context and when they must be isolated.
5. Design a routing policy for a multi-surface agent product.

---

## 1. Four things students often mix up

When reading a real agent system, students often mix up:

- channel
- account
- agent
- session

OpenClaw is useful because it separates them clearly.

### Channel

A **channel** is the communication surface.

Examples:

- Telegram
- WhatsApp
- Slack
- Discord
- WebChat

### Account

An **account** is a specific identity on a channel.

Examples:

- one Telegram bot token
- one WhatsApp number
- one Slack app installation

You can have multiple accounts on the same channel type.

### Agent

An **agent** is the isolated brain:

- workspace
- instructions
- tools
- sessions
- auth profiles

One gateway can host multiple agents.

### Session

A **session** is the context bucket for a conversation or workflow.

It decides which messages **share memory** and transcript history.

This distinction is essential.

One user may contact the same agent through many channels, but whether those conversations share context is a **design choice**.

---

## 2. The routing problem

An inbound message is not just "text for the model."

It first raises a routing question:

> Which agent and which session should own this message?

That question depends on:

- channel
- sender
- account
- group or room
- thread
- configured bindings

This is why OpenClaw uses **explicit bindings**.

Instead of letting the model decide, the **host configuration** decides.

That is the right design.

The model should not choose:

- which human it is serving
- which workspace it is using
- which account should answer

Those are **control-plane decisions**.

---

## 3. Session keys in simple language

OpenClaw documents the idea of a **session key**.

The easiest explanation is:

> a session key is the label on the conversation bucket

Messages with the same bucket label **share context**.

Messages with different bucket labels **stay isolated**.

Examples from the OpenClaw model:

- one direct-message session
- one session per group chat
- one session per room
- one session per thread

This is extremely practical.

It means the system can say:

- all messages in this Slack thread belong together
- this Telegram group must not mix with that Discord room
- these DMs should share one private session
- these two users must never share context

---

## 4. DM isolation is not optional in multi-user systems

OpenClaw's session docs are very clear here:

If many people can message the bot, default shared-DM behavior can **leak context**.

That is a big teaching point.

A beginner might think:

> one assistant should have one big memory

But in a multi-user product, that is often wrong.

Example of a bad design:

```text
Alice messages the assistant privately.
Bob messages the same assistant privately.
Both share one DM session.
The assistant now carries Alice's context into Bob's chat.
```

This is not just awkward. It can be a **privacy issue**.

That is why OpenClaw supports DM scope settings like:

- one shared main session
- per-peer isolation
- per-channel-peer isolation

For a serious product, session scoping is a **security and UX feature**.

---

## 5. Channel routing as a product decision

OpenClaw's routing rules show that agent assignment can depend on:

- exact peer
- account
- channel
- group
- team
- guild
- role

That means routing is not merely **technical plumbing**.

It is **product behavior**.

Example:

```text
Telegram support bot -> support agent
Slack engineering room -> engineering agent
WhatsApp family group -> family assistant agent
Discord moderator room -> moderation agent
```

These are different products running inside one gateway.

So routing design is how you turn one runtime into many useful behaviors.

---

## 6. Example: the same agent across multiple surfaces

Suppose you want one personal assistant to exist in:

- Telegram DM
- WebChat
- mobile node voice input

You now have a design choice:

### Option A - one shared session

Pros:

- continuity across surfaces
- the agent remembers context everywhere

Cons:

- context can become messy
- one accidental message on one surface affects the others

### Option B - isolated sessions per surface

Pros:

- cleaner history
- easier debugging
- less accidental cross-surface contamination

Cons:

- weaker continuity

This is why session design is a **product tradeoff**, not a default you should ignore.

---

## 7. Example config pattern

Here is a simplified OpenClaw-style pattern:

```json5
{
  agents: {
    list: [
      { id: "support", workspace: "~/.openclaw/workspace-support" },
      { id: "personal", workspace: "~/.openclaw/workspace-personal" }
    ]
  },
  bindings: [
    { match: { channel: "slack", teamId: "T123" }, agentId: "support" },
    { match: { channel: "telegram", peer: { kind: "direct", id: "user_42" } }, agentId: "personal" }
  ],
  session: {
    dmScope: "per-channel-peer"
  }
}
```

You do not need to memorize the exact config.

The lesson is:

- agents are explicit
- bindings are explicit
- session policy is explicit

That is good architecture.

---

## 8. Groups, threads, and rooms

A serious agent product must understand that:

- a direct message is not the same as a group
- a group is not the same as a thread
- a room is not the same as a one-to-one chat

OpenClaw models this by keeping many of these session types isolated.

That is the right default for collaborative systems.

Why?

Because threads often represent **separate sub-conversations**.

If they all collapse into one bucket, the agent becomes **noisy and unreliable**.

This is the same lesson you should apply when building:

- support agents
- research agents
- team copilots
- personal assistants

---

## 9. Reply routing should be deterministic

OpenClaw's channel-routing docs make an important point:

> the model does not choose the channel

That choice belongs to the **host system**.

This is a good professional rule.

The model should help decide:

- what to say
- which tool to use
- how to summarize

The control plane should decide:

- where to reply
- which account to use
- which session to mutate
- which agent is in scope

This reduces a whole class of failure where the model **invents the wrong operational path**.

---

## 10. Design exercise

Design routing for this product:

> One family assistant and one engineering assistant run on the same host.

Requirements:

- Telegram DMs from family members go to the family assistant.
- Slack messages in the engineering workspace go to the engineering assistant.
- WebChat should talk only to the engineering assistant.
- DMs must not share context across users.

Fill this table:

| Decision area | Your answer |
|---|---|
| Agents | `family`, `engineering` |
| Channels | Telegram, Slack, WebChat |
| DM scope | `per-channel-peer` |
| Support binding | Slack workspace -> `engineering` |
| Family binding | Telegram direct peers -> `family` |
| Web binding | WebChat -> `engineering` |

This is a better system design exercise than "write a prompt for a helpful assistant."

---

## Key takeaways

- Channel, account, agent, and session are different concepts and should stay separate in your mind.
- Session design decides who shares context with whom.
- Routing is a product feature, not just backend plumbing.
- DMs should usually be isolated in multi-user systems.
- Reply routing should be deterministic and host-controlled, not model-chosen.
- OpenClaw is a strong example of how real agent products treat routing and session state as first-class concerns.

---

## References

- Case-study source repo: [OpenClaw](https://github.com/openclaw/openclaw)
- OpenClaw concepts:
  - `docs/channels/channel-routing.md`
  - `docs/concepts/session.md`
  - `docs/concepts/multi-agent.md`
  - `docs/channels/pairing.md`

---

*Next: [Lecture 33](Lecture-33.md)*
