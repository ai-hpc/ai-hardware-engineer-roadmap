# Lecture 29 - Agent Skills: Workflow Discipline for Reliable Coding Agents

**Course:** [Agentic AI & GenAI](../Guide.md) | **Previous:** [Lecture 28](Lecture-28.md) | **Next:** [Lecture 30](Lecture-30.md)

---

Modern coding agents can generate code quickly.

That is not the same as doing engineering work correctly.

The useful mental model:

```text
Agents optimize for "done."
Senior engineers optimize for correct, reviewable, and safe.
Agent skills encode the missing senior-engineering process.
```

This lecture uses Addy Osmani's Agent Skills work as a reference pattern and adapts it to OpenClaw-style harnesses, local-first agents, on-device AI, and hardware bring-up workflows.

---

## Learning objectives

By the end of this lecture, you should be able to:

1. Explain why agent skills are workflows, not knowledge dumps.
2. Design a skill with triggers, checkpoints, evidence, and exit criteria.
3. Use anti-rationalization tables to prevent shortcut behavior.
4. Apply progressive disclosure so agents load only relevant workflows.
5. Separate soft skill guidance from hard runtime enforcement.
6. Map skills into OpenClaw-style prompts, hooks, tools, sessions, and artifacts.
7. Write skills for coding, hardware bring-up, and on-device AI work.

---

## 1. Why agents fail in practice

Agents often fail because they skip invisible engineering work:

| Missing discipline | Failure mode |
|---|---|
| Spec | the agent solves the wrong problem |
| Constraints | the agent changes files or behavior outside scope |
| Tests | the agent declares success without proof |
| Reviewability | the final diff is too broad to trust |
| Runtime evidence | code compiles but fails in the real environment |
| Safety boundary | the prompt says "be careful" but tools still allow damage |

This resembles a fast junior engineer:

```text
can produce output
but may skip assumptions, tests, and review shape
```

Agent skills exist to make the missing process explicit.

---

## 2. What an agent skill actually is

A useful skill is not:

```text
"Follow best practices."
```

A useful skill is:

```text
a small workflow
with specific steps
and a concrete completion signal
```

Weak instruction:

```text
Use TDD where appropriate.
```

Skill-shaped instruction:

```text
1. Identify the behavior contract.
2. Write the smallest failing test.
3. Run it and capture the failure.
4. Implement the smallest fix.
5. Run the targeted test and capture the pass.
6. Run the relevant broader check.
7. Finalize only with evidence.
```

The first version gives advice.

The second version creates a loop.

---

## 3. Where skills sit in the agent stack

Skills are one layer in the harness:

```text
Model
  -> system prompt
  -> skill router
  -> active skill workflow
  -> tools
  -> hooks and policy
  -> logs and artifacts
  -> final answer
```

In OpenClaw language:

```text
Gateway
  -> agent loop
  -> prompt assembly
  -> skills / bootstrap context
  -> tool execution
  -> hooks and approvals
  -> session log
  -> artifacts and delivery
```

Important distinction:

```text
Skill = workflow instruction
Hook = deterministic interception
Tool policy = authority boundary
Artifact = durable evidence
```

Do not ask a skill to do the job of policy.

A skill can say "ask before deleting files."

The runtime should still deny unsafe delete tools unless policy allows them.

---

## 4. Process over prose

Agents can summarize rules without applying them.

So a skill should prefer action steps over essays.

Weak:

```text
Be careful with production changes.
```

Better:

```text
Before editing production code:
1. Identify the production boundary.
2. Identify rollback path.
3. List files allowed to change.
4. List tests or runtime checks required.
5. Stop if required evidence cannot be produced.
```

Skill design rule:

```text
If the agent cannot act on it, it is reference material, not a skill.
```

---

## 5. Anti-rationalization tables

Agents are good at plausible excuses.

Examples:

| Shortcut claim | Required rebuttal |
|---|---|
| "This is too small for a spec." | Small changes still need acceptance criteria. Write the smallest possible spec. |
| "I will add tests later." | Later usually means never. Add the minimal verification now. |
| "The code compiles, so it works." | Compilation is one signal, not behavior proof. Run the relevant check. |
| "This nearby refactor is useful." | Useful is not requested. Keep the diff scoped unless scope expansion is approved. |
| "The tool output is probably good enough." | Mutable state must be checked live before finalizing. |
| "This is local only, so security does not matter." | Local agents often hold secrets and filesystem authority. Apply least privilege. |

This is cheap and effective.

The goal is to pre-write the response to the shortcuts the model is likely to take.

---

## 6. Verification is mandatory

A skill should end with evidence.

Evidence examples:

| Task type | Evidence |
|---|---|
| Code change | test output, lint output, build output |
| UI change | screenshot, visual diff, responsive check |
| API change | schema diff, contract test, compatibility note |
| Runtime change | health check, log excerpt, smoke test |
| Security change | denied-path test, permission audit, policy check |
| Documentation change | docs build, link check, rendered preview |
| Hardware bring-up | kernel log, bus scan, command output, waveform capture |

Rule:

```text
No evidence, no completion.
```

This matters more for long-running agents.

Small shortcuts compound over long sessions.

---

## 7. Progressive disclosure

Do not load every workflow into every run.

That creates:

- token bloat
- weaker attention
- slower inference
- more compaction pressure
- irrelevant instruction conflicts

Better pattern:

```text
small router
  -> load only relevant skill
  -> load deeper references only when needed
```

Example:

```text
Bug fix request
  -> load test-driven-bugfix
  -> maybe load runtime-debug
  -> do not load deployment, frontend, and release skills unless needed
```

This is especially important for on-device AI where context, latency, memory, and thermal budget matter.

---

## 8. Scope discipline

A reliable coding agent must keep changes reviewable.

Before editing:

```text
- list intended files
- list non-goals
- identify protected areas
- ask before broadening scope
```

Before final answer:

```text
- list files changed
- state whether scope expanded
- explain why any expansion was necessary
- provide verification evidence
```

Reviewability is not a cosmetic concern.

It is how humans keep authority over generated work.

---

## 9. Skill anatomy

A practical `SKILL.md` should be short and structured:

```markdown
---
name: test-driven-bugfix
description: Use for bug fixes where behavior must be proven with tests or runtime evidence.
---

# Test-Driven Bug Fix

## When to use

Use when fixing a bug, regression, failing test, or runtime error.

## Workflow

1. Reproduce the bug or failing behavior.
2. Record the exact failure output.
3. Identify the smallest behavior contract.
4. Add or update the minimal failing test.
5. Run the test and confirm failure.
6. Implement the smallest fix.
7. Run the targeted test and confirm pass.
8. Run the relevant broader check.
9. Review the diff for unrelated changes.

## Anti-rationalization

| Claim | Response |
|---|---|
| "This is obvious." | Obvious fixes still need evidence. |
| "There is no test harness." | Use the smallest available runtime or command-level check. |
| "The failure is intermittent." | Capture logs and state what was and was not reproduced. |

## Exit criteria

- Failure was reproduced or explicitly marked unreproducible.
- Fix is scoped to the bug.
- Verification command and result are recorded.
- No unrelated files were changed.
```

This is compact enough to load and specific enough to audit.

---

## 10. Hardware bring-up skill

Agent skills are useful for embedded and hardware work because bring-up is full of mutable state.

Example:

```markdown
---
name: hardware-bringup-debug
description: Use for Jetson, ESP32, I2S, SPI, UART, kernel, driver, and device-tree debugging.
---

# Hardware Bring-Up Debug

## Workflow

1. Identify board, OS image, kernel version, and exact hardware path.
2. Record the expected signal or interface contract.
3. Capture current observable state.
4. Separate host, wiring, firmware, driver, and userspace hypotheses.
5. Test one hypothesis at a time.
6. Do not change kernel, device tree, firmware, and userspace simultaneously.
7. Preserve raw command outputs for evidence.
8. Summarize blocker and next physical or software check.

## Anti-rationalization

| Claim | Response |
|---|---|
| "It is probably wiring." | Prove host and software state before blaming wiring. |
| "It is probably software." | Check voltage, pinmux, and physical bus assumptions. |
| "Let's rebuild everything." | Change one layer at a time or the result is not diagnosable. |
```

This applies directly to:

- Jetson I2S microphone capture
- ESP32-C6 RCP/NCP bring-up
- OpenThread attach debugging
- Zigbee coordinator testing
- camera sensor bring-up
- audio codec device-tree work

The skill prevents the classic failure:

```text
change five variables, then no one knows which one mattered
```

---

## 11. On-device AI skill

On-device agents have additional constraints:

- memory pressure
- thermal budget
- local privacy
- smaller context windows
- intermittent network
- model fallback behavior
- hardware permissions

Example:

```markdown
---
name: on-device-agent-change
description: Use when modifying an agent that runs on a laptop, Jetson, phone, or local gateway.
---

# On-Device Agent Change

## Workflow

1. Identify target device and runtime limits.
2. Identify local-only data and privacy boundaries.
3. Check startup path and readiness gates.
4. Keep prompt/context additions minimal.
5. Prefer deterministic checks over model judgment.
6. Validate behavior with network unavailable if relevant.
7. Record CPU/GPU/memory impact when measurable.

## Exit criteria

- startup remains deterministic
- local permissions are unchanged or explicitly reviewed
- context growth is bounded
- fallback behavior is documented
- verification was run on or representative of the target device
```

This fits OpenClaw, Jetson, and local-first assistant systems.

---

## 12. Runtime enforcement pattern

Use two layers:

```text
1. Soft guidance: skill workflow
2. Hard enforcement: harness policy
```

Examples:

| Workflow requirement | Runtime enforcement |
|---|---|
| Run tests before finalizing | final-answer hook checks for test evidence |
| Do not edit outside scope | filesystem policy or diff checker |
| Ask before dangerous command | exec approval gate |
| Keep secrets out of logs | log redaction and denylisted paths |
| Use small context | prompt budget and context inspectors |
| Preserve evidence | artifact API or session attachment |

Prompts help behavior.

They do not enforce authority.

---

## 13. Evidence ledger

For production agents, keep a run-level evidence ledger:

```text
task id
skill used
files touched
tools called
approval decisions
tests run
logs captured
artifacts created
scope changes
known gaps
```

This supports:

- review
- debugging
- incident response
- auditability
- future skill improvement

OpenClaw-style systems can store this across:

- session transcript
- run events
- artifacts
- Gateway RPC task state
- external dashboards

Principle:

```text
If the agent claims success, the system should know why.
```

---

## 14. Practical implementation checklist

Start with five skills:

| Skill | Why it matters |
|---|---|
| `spec-first` | prevents wrong-target implementation |
| `small-plan` | forces reviewable chunks |
| `test-driven-bugfix` | creates behavior evidence |
| `runtime-safety-review` | catches tool, permission, and data risks |
| `hardware-bringup-debug` | prevents multi-variable debugging chaos |

For each skill, define:

```text
name
description
when to use
workflow
anti-rationalization table
exit criteria
evidence format
references, if needed
```

Then add:

- a small router
- a final-answer evidence check
- a diff-scope check
- a way to inspect which skill ran
- skill versioning for reproducibility

---

## Mini-lab

Create one local skill for a painful workflow.

Recommended choices:

- Jetson audio debug
- ESP32-C6 radio bring-up
- OpenClaw plugin debugging
- App SDK smoke test
- model runtime regression
- documentation build failure

Test it manually:

1. Give the agent a task that should trigger the skill.
2. Check whether it follows the workflow.
3. Check whether it produces evidence.
4. Check whether the final answer is reviewable.
5. Revise the skill where the agent skipped or rationalized.

---

## Key takeaways

- Agent skills turn senior-engineering discipline into reusable workflows.
- A useful skill is process, not prose.
- Skills need checkpoints, anti-rationalization, and exit criteria.
- Verification must produce evidence.
- Progressive disclosure keeps context small and relevant.
- Scope discipline makes agent output reviewable.
- Skills do not replace hooks, approvals, sandboxing, or tool policy.

---

## References

- Addy Osmani, "Agent Skills": [https://addyosmani.com/blog/agent-skills/](https://addyosmani.com/blog/agent-skills/)
- Agent Skills repository: [https://github.com/addyosmani/agent-skills](https://github.com/addyosmani/agent-skills)
- Lecture 19 - OpenClaw Agent Loop: [Lecture-19.md](Lecture-19.md)
- Lecture 21 - System Prompt Architecture: [Lecture-21.md](Lecture-21.md)
- Lecture 28 - Pi: [Lecture-28.md](Lecture-28.md)

---

*Next: [Lecture 30 - Agentic SDLC: Explore Fast, Ship Safely](Lecture-30.md)*
