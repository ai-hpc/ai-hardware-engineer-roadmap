# Lecture 21 - OpenClaw Case Study: System Prompt Architecture

**Track B - Agentic AI & GenAI** | [<- Lecture 20](Lecture-20.md) | [Next -> Lecture 22](Lecture-22.md)

---

Modern agent systems are not controlled by one short prompt.

They are controlled by a **prompt assembly system**.

That system decides:

- what identity the agent receives
- what tools it knows how to use
- what workspace context is injected
- what safety guidance is present
- what provider-specific tuning is added
- what should stay stable for prompt caching
- what should stay out of the prompt and be enforced by runtime policy

This lecture uses OpenClaw's system prompt design as the case study.

The important lesson is:

> a production agent should not depend on a random provider default prompt; it should have an owned, inspectable, versioned prompt assembled by the application runtime

---

## Learning objectives

By the end of this lecture, you should be able to:

1. Explain why OpenClaw owns the system prompt instead of using provider defaults.
2. Describe the major sections of an OpenClaw-style system prompt.
3. Understand full, minimal, and none prompt modes.
4. Explain how bootstrap files become Project Context.
5. Understand why skills are listed compactly and loaded on demand.
6. Separate advisory prompt safety from hard runtime enforcement.
7. Design a cache-aware provider overlay.
8. Inspect prompt size and context injection when debugging an agent.

---

## 1. Simple mental model

Think of a serious agent as a field engineer.

Before the engineer starts work, you do not just say:

```text
Be helpful.
```

You give them an operating binder:

```text
Who you are.
How to talk to users.
How to use tools.
Which workspace you are in.
What files contain project context.
What policies you must follow.
When to ask for help.
How to report completion.
```

In OpenClaw, the **system prompt** is that operating binder.

It is assembled before each agent run.

```text
agent request
  -> resolve agent/session/workspace
  -> assemble OpenClaw-owned system prompt
  -> inject bootstrap context and skills list
  -> apply provider-specific small overlays
  -> run model
```

The model does not invent the operating binder.

OpenClaw builds it.

---

## 2. Why OpenClaw owns the system prompt

A provider default prompt is useful for a generic chat product.

It is not enough for a product that has:

- tools
- sessions
- workspaces
- sub-agents
- cron jobs
- long-running processes
- provider plugins
- local docs
- memory files
- sandbox behavior
- gateway commands
- user-specific persona files

OpenClaw owns the prompt so that behavior is consistent across models and providers.

The system can say:

- "This is how OpenClaw agents use tools."
- "This is where local docs live."
- "This is the workspace."
- "This is how sub-agents should be used."
- "This is what bootstrap context was injected."
- "This is what safety means in this runtime."

The provider still matters, but the product owns the agent contract.

---

## 3. The fixed prompt spine

OpenClaw uses compact, structured sections.

The exact wording may evolve, but the architecture is stable.

An OpenClaw-style prompt looks like this:

```text
Identity
Tooling
Execution Bias
Safety
Skills
OpenClaw Self-Update
Workspace
Documentation
Project Context
Sandbox
Current Date & Time
Reply Tags
Heartbeats
Runtime
Reasoning
Provider additions
```

The point is not to make the prompt long.

The point is to make it **predictable**.

Each section has one job.

---

## 4. Tooling section

The Tooling section teaches the model how work should be done inside this runtime.

It covers patterns such as:

- use structured tools instead of pretending to act
- prefer `cron` for future follow-ups instead of sleep loops
- use process tools for long-running commands and logs
- use sub-agents for larger parallel work
- do not poll sub-agents in tight loops
- use `update_plan` only when it is enabled and useful

This section is important because tool-using agents often fail in boring ways:

- they say they did something but did not call the tool
- they start a long process and lose the logs
- they sleep inside a command instead of scheduling future work
- they spawn too many agents for simple work
- they update a plan repeatedly without doing work

The Tooling section converts product expectations into model-facing habits.

---

## 5. Execution Bias section

Execution Bias is the "finish the work" section.

It gives compact follow-through guidance:

- act within the current turn when the request is actionable
- continue until done or genuinely blocked
- recover when a tool result is weak
- check mutable state live instead of assuming
- verify before finalizing

This section exists because many models default to advice.

A production engineering agent often needs action.

Compare:

```text
Weak behavior:
"You could run the tests."

OpenClaw-style behavior:
"Run the tests, inspect the failure, patch the issue, rerun verification, then summarize."
```

Execution Bias does not mean reckless automation.

It means the agent should carry work through the runtime loop when it has enough permission and context.

---

## 6. Safety section

The Safety section is intentionally brief.

It tells the model not to bypass oversight, escalate privileges improperly, hide risky behavior, or seek control outside the user intent.

But this is a key production lesson:

> prompt safety is advisory; runtime enforcement is mandatory

The system prompt can ask the model to behave safely.

Hard enforcement must come from:

- tool policy
- exec approvals
- sandboxing
- filesystem boundaries
- channel allowlists
- identity and permission checks
- audit logs

If a command must never run, do not merely write "do not run it" in a prompt.

Block it in the tool layer.

---

## 7. Skills section

OpenClaw can inject a compact list of available skills.

The prompt does not paste every skill into the context.

It lists:

- skill name
- short description
- location

Then the model is instructed to read the relevant `SKILL.md` only when needed.

That is the correct architecture.

Bad pattern:

```text
Paste every skill file into every run.
```

Good pattern:

```text
List available skills compactly.
Load the matching skill on demand.
```

This avoids token bloat and keeps unrelated skills from polluting the run.

Skills have their own sizing budget:

- global default: `skills.limits.maxSkillsPromptChars`
- per-agent override: `agents.list[].skillsLimits.maxSkillsPromptChars`

This is separate from other runtime context limits.

That separation matters because a skills list and a workspace bootstrap file are different kinds of context.

---

## 8. OpenClaw Self-Update section

OpenClaw can expose tools for safely inspecting and changing its own configuration.

The prompt teaches a controlled path:

```text
config.schema.lookup
config.patch
config.apply
update.run
```

The model should inspect the schema before changing config.

It should patch narrowly.

It should apply through the supported gateway tool.

It should not rewrite protected execution policy keys casually.

This is a useful pattern for any self-modifying agent system:

> self-update should be schema-driven, narrow, logged, and guarded

Do not give an agent raw config file mutation and hope the prompt keeps it safe.

---

## 9. Workspace and documentation sections

The Workspace section tells the agent where it is operating.

It usually reflects:

```text
agents.defaults.workspace
```

The Documentation section tells the agent where local OpenClaw docs live and encourages consulting them first.

This is a subtle but important production pattern.

For a tool-rich local assistant, local docs are often more accurate than memory.

The prompt should point the model to the local source of truth:

```text
If you need OpenClaw behavior, inspect local docs first.
Then act.
```

---

## 10. Project Context and bootstrap files

OpenClaw appends selected workspace files under Project Context.

These are bootstrap files such as:

- `AGENTS.md`
- `SOUL.md`
- `TOOLS.md`
- `IDENTITY.md`
- `USER.md`
- `HEARTBEAT.md`
- `BOOTSTRAP.md`
- `MEMORY.md`

The goal is simple:

> important identity and project context should be present without requiring the model to remember to read it

Example:

```text
Project Context
  AGENTS.md -> project rules
  SOUL.md -> personality and voice
  TOOLS.md -> custom workspace tool guidance
  USER.md -> user preferences
  MEMORY.md -> durable compact memory
```

This is powerful, but it has a cost.

Large bootstrap files increase:

- prompt size
- compaction pressure
- latency
- cache invalidation risk
- irrelevant context exposure

So OpenClaw trims injection.

Documented defaults include:

- per-file max: `agents.defaults.bootstrapMaxChars`, default `12000`
- total injected max: `agents.defaults.bootstrapTotalMaxChars`, default `60000`
- truncation warning: `agents.defaults.bootstrapPromptTruncationWarning`, default `once`

The practical rule:

> bootstrap files should be concise operating context, not a dumping ground

---

## 11. Memory files

`MEMORY.md` can be injected as bootstrap context.

Daily memory files under `memory/*.md` are different.

They are not normally injected by default.

They are accessed on demand through memory tools such as:

```text
memory_search
memory_get
```

This keeps normal runs small.

Recent daily memory may be prepended once in special bare `/new` or `/reset` turns, but it is not the default for every run.

This is the right tradeoff:

- durable compact memory can be prompt context
- large daily logs should be retrieved only when relevant

---

## 12. Prompt modes

OpenClaw supports multiple prompt modes.

### Full mode

Full mode is the default.

It includes the main sections:

- Tooling
- Execution Bias
- Safety
- Skills
- OpenClaw Self-Update
- Workspace
- Documentation
- Project Context
- Sandbox
- Current Date and Time
- Reply Tags
- Heartbeats
- Runtime
- Reasoning

Use full mode for the main interactive agent.

### Minimal mode

Minimal mode is used for sub-agents.

It keeps the essential runtime contract but omits context that would bloat or confuse a delegated worker.

It omits sections like:

- Skills
- Memory Recall
- OpenClaw Self-Update
- Model Aliases
- User Identity
- Reply Tags
- Messaging
- Silent Replies
- Heartbeats

It keeps sections like:

- Tooling
- Safety
- Workspace
- Sandbox
- Current Date and Time
- Runtime
- injected context

In minimal mode, injected prompts are labeled **Subagent Context** instead of **Group Chat Context**.

The idea is:

> a sub-agent needs enough context to do its bounded task, not the entire personality and memory system of the main assistant

### None mode

None mode returns only the base identity line.

Use it rarely.

It is useful for tests or cases where the caller wants almost no runtime prompt material.

---

## 13. Sub-agent bootstrap behavior

Sub-agent sessions inject less context.

OpenClaw limits sub-agent bootstrap to:

- `AGENTS.md`
- `TOOLS.md`

This is intentional.

A sub-agent should know project rules and tool rules.

It usually does not need the full user profile, memory, heartbeat behavior, or self-update instructions.

This keeps delegated work cheaper and less noisy.

---

## 14. Time handling and prompt-cache stability

The Current Date and Time section is designed carefully.

It includes timezone and time-format guidance.

It avoids injecting a live clock into every prompt.

Why?

Because a live timestamp changes every run.

Changing every run can hurt prompt caching.

So OpenClaw keeps the prompt-cache boundary stable and tells the model how to get exact time when needed.

When the exact timestamp matters, use:

```text
session_status
```

Relevant config keys:

- `agents.defaults.userTimezone`
- `agents.defaults.timeFormat`

This is a strong production lesson:

> do not put high-churn values into the stable prompt unless the model truly needs them every turn

---

## 15. Provider tuning and cache-aware overlays

Provider plugins are allowed to contribute small additions.

They should not replace the whole OpenClaw system prompt.

Provider plugins can:

- replace named core sections such as `interaction_style`
- replace `tool_call_style`
- replace `execution_bias`
- inject a stable prefix above the prompt-cache boundary
- inject a dynamic suffix below the prompt-cache boundary

This gives model-family tuning without losing product control.

Example:

```text
OpenClaw core prompt:
  stable runtime behavior

Provider overlay:
  small model-family guidance for GPT-5, Claude, local models, etc.
```

The OpenAI GPT-5 family overlay is described as keeping core execution rules small while adding guidance such as:

- persona latching
- concise output
- tool discipline
- parallel lookup
- deliverable coverage
- verification
- missing context handling
- terminal-tool hygiene

The architecture rule is:

> provider tuning should be a small overlay, not a hostile takeover of the product prompt

---

## 16. Legacy hooks versus provider contributions

OpenClaw still supports a legacy `before_prompt_build` hook.

It can inject or mutate prompt material globally.

But for model-family behavior, provider contributions are preferred.

Why?

Because provider contributions can be cache-aware and scoped to the model family.

Use the right layer:

| Need | Better mechanism |
|---|---|
| Workspace-specific context | Bootstrap files |
| Global prompt mutation | `before_prompt_build` hook |
| Model-family tuning | Provider contribution |
| Runtime security | Tool policy and sandbox |
| User personality | `SOUL.md` |

---

## 17. Reply Tags and Heartbeats

Reply Tags are optional provider-specific syntax guidance.

They help models format replies in a way the runtime can parse or route.

Heartbeats are also optional.

When enabled, OpenClaw can include heartbeat prompt and acknowledgement behavior.

But heartbeats are omitted from normal runs when:

- heartbeats are disabled for the default agent
- `agents.defaults.heartbeat.includeSystemPromptSection` is false

This keeps the prompt clean when heartbeat behavior is not active.

---

## 18. Runtime and Reasoning sections

The Runtime section provides a compact one-line summary of the execution environment.

It can include:

- host
- OS
- Node.js version
- selected model
- repo root if detected
- thinking level

The Reasoning section explains visibility level and can hint at a `/reasoning` toggle.

This is not meant to be verbose.

It is a compact runtime signal so the model knows where it is operating.

---

## 19. Diagnostics: inspecting context

When an agent behaves strangely, inspect what it actually received.

OpenClaw supports commands such as:

```text
/context list
/context detail
```

Use these to inspect:

- which files were injected
- raw size versus injected size
- whether truncation happened
- tool schema overhead
- which context source dominates the prompt

This is critical for debugging.

Many "model problems" are actually context problems:

- a bootstrap file is too large
- a stale memory file is injected
- a project rule is missing
- a provider overlay is too strong
- a sub-agent received full context instead of minimal context

Inspect the prompt assembly before blaming the model.

---

## 20. Example: smart speaker engineering agent

Imagine an OpenClaw-powered smart speaker assistant for a Jetson-based product lab.

The agent may need to:

- inspect hardware notes
- schedule follow-up tests
- run shell commands
- spawn a sub-agent to research codecs
- remember user preferences
- avoid unsafe GPIO or power commands
- summarize logs

An OpenClaw-style system prompt might assemble this:

```text
Tooling:
  Use process tools for long-running audio tests.
  Use cron for future lab reminders.
  Spawn sub-agents for isolated research.

Execution Bias:
  Run available checks before answering.
  Verify file paths and hardware state live.

Safety:
  Do not bypass exec policy.
  Do not change protected power or network settings without approval.

Workspace:
  /home/lab/smart-speaker

Project Context:
  AGENTS.md: lab rules
  TOOLS.md: audio test commands
  USER.md: preferred report format
  MEMORY.md: concise project memory

Runtime:
  Jetson Orin Nano, Linux, local model provider, reasoning medium
```

Notice what is **not** in the prompt:

- every past lab log
- every available skill file
- every daily memory file
- raw policy implementation
- secrets

The prompt gives the model the operating contract.

The runtime enforces the dangerous boundaries.

---

## 21. Common design mistakes

### Mistake 1: putting everything in the system prompt

Large prompts feel powerful, but they become slow and noisy.

Use retrieval, skills, memory search, and local docs instead.

### Mistake 2: relying on prompt safety alone

If a tool action is dangerous, gate it in the tool layer.

Prompt wording is not a permission system.

### Mistake 3: giving sub-agents full context

Sub-agents should receive bounded context for bounded work.

Full identity and memory can distract them.

### Mistake 4: putting live timestamps above the cache boundary

High-churn values reduce cache stability.

Expose exact time through tools when needed.

### Mistake 5: letting provider plugins replace the product prompt

Provider overlays should tune.

They should not own the product contract.

---

## 22. Design exercise

Design a system prompt strategy for a local AI hardware lab assistant.

The assistant can:

- answer questions
- inspect local Markdown docs
- run safe shell commands
- schedule follow-ups
- use a sub-agent for research
- remember hardware inventory

Answer these:

1. Which sections belong in the full prompt?
2. Which sections should be omitted from sub-agent minimal prompts?
3. Which files should be bootstrap-injected?
4. Which information should be retrieved on demand instead of injected?
5. Which safety requirements must be enforced by tools rather than prompt text?
6. Which provider-specific guidance should be a small overlay?
7. How will you inspect prompt size and truncation?

If you cannot answer these, your agent system is not yet operationally mature.

---

## Key takeaways

- OpenClaw owns and assembles the system prompt for each agent run.
- The system prompt is a runtime contract, not a generic chat instruction.
- Provider plugins should add small cache-aware overlays, not replace the full prompt.
- Full mode is for main agents; minimal mode is for bounded sub-agents; none mode is for rare low-prompt cases.
- Bootstrap files provide Project Context, but they must stay concise.
- Skills should be listed compactly and loaded on demand.
- Exact time should come from tools when needed so the stable prompt remains cache-friendly.
- Prompt safety is advisory; hard safety belongs in runtime controls.
- Context inspection is a first-class debugging skill.

---

## References

- OpenClaw system prompt: [https://openclaw.knidal.com/system-prompt](https://openclaw.knidal.com/system-prompt)
- Case-study source repo: [OpenClaw](https://github.com/openclaw/openclaw)
- Related OpenClaw concepts:
  - agent loop
  - cron jobs
  - sessions and workspaces
  - skills
  - runtime configuration

---

*Next: [Lecture 22 - OpenClaw Case Study: App SDK Dogfooding and Typed Gateway RPCs](Lecture-22.md)*
