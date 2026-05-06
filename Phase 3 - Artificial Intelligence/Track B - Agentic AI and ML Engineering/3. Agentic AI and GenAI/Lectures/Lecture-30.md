# Lecture 30 - Agentic SDLC: Explore Fast, Ship Safely

**Course:** [Agentic AI & GenAI](../Guide.md) | **Previous:** [Lecture 29](Lecture-29.md) | **Next:** [Lab 01](Lab-01-Research-Agent.md)

---

Lecture 29 focused on agent skills:

```text
agents skip discipline
  -> encode senior-engineering workflows
  -> require checkpoints, evidence, tests, and scope control
```

This lecture starts from the opposite side:

```text
code is cheaper now
  -> use implementation as exploration
  -> preserve what matters: tests, intent, specs, security, and taste
```

The tension is the point:

```text
Explore fast.
Ship safely.
```

Strong agent systems support both.

---

## Learning objectives

By the end of this lecture, you should be able to:

1. Explain why cheap code changes software-process economics.
2. Separate exploration code from shipping code.
3. Treat tests and intent as persistent assets.
4. Explain why end-to-end behavior tests matter more when agents can rewrite internals quickly.
5. Keep specs synchronized with implementation instead of freezing them upfront.
6. Identify which work should be automated and which work still requires human taste.
7. Design a dual-mode agent workflow: explore mode and stabilize mode.
8. Apply this workflow to OpenClaw, on-device AI, and hardware engineering.

---

## 1. The core shift

Traditional software economics assumed:

| Old world | Practical effect |
|---|---|
| Writing code is expensive | plan carefully before coding |
| Rewriting is expensive | avoid large experiments |
| Tests feel like overhead | test after implementation pressure allows it |
| Specs are upfront artifacts | write once, then implement |

Agentic coding shifts the cost structure:

| Agentic world | Practical effect |
|---|---|
| Writing code is cheap | implement to learn |
| Rebuilding is cheaper | try parallel designs |
| Tests become the asset | behavior contracts let internals change |
| Specs are continuous | update intent as learning happens |

The bottleneck moves from typing code to judging code:

```text
Do we know what is worth building?
Can we tell when it is correct?
Can we maintain what we generated?
Can we keep it safe?
```

---

## 2. Code as exploration

"Implement to learn" is the key idea.

Sometimes you do not know the right design until you build a rough version.

This is especially true for:

- UI workflows
- agent loops
- streaming event protocols
- retrieval quality
- latency paths
- hardware bring-up scripts
- deployment automation
- developer experience

Prototype code becomes a probe:

```text
implementation -> feedback -> updated intent
```

You build a slice to discover:

- missing requirements
- hidden state
- bad abstractions
- UX friction
- testability problems
- performance bottlenecks
- security assumptions

Then you decide what to keep.

---

## 3. Cheap code still has expensive consequences

Cheap generation does not make software free.

It moves cost into:

- review
- verification
- maintenance
- support
- security
- incident response
- user trust
- documentation
- operational ownership

The practical rule:

```text
Treat exploratory code as disposable.
Treat tests and intent as assets.
```

This is why "vibe coding" without contracts breaks down quickly.

The agent can generate a feature in minutes.

The team still owns the bugs for months.

---

## 4. The synthesis with Agent Skills

Lecture 29 and this lecture fit together like this:

| Concern | Agentic SDLC | Agent Skills |
|---|---|---|
| Exploration | implement to learn, rebuild often | not the main focus |
| Discipline | maintenance is real | workflow checkpoints |
| Tests | persistent behavioral contracts | mandatory exit criteria |
| Specs | continuously synchronized | structured entry point |
| Safety | cheap code does not remove risk | anti-rationalization and policy gates |
| Human role | taste and experience become bottlenecks | review, scope, and verification discipline |

Combined loop:

```text
EXPLORE
  -> build cheap prototypes
  -> learn from behavior
  -> update intent

LOCK IN
  -> turn useful behavior into tests
  -> update specs
  -> define constraints

STABILIZE
  -> apply skills
  -> verify
  -> review diff

SHIP
  -> release with evidence
  -> monitor and maintain
```

This is the agentic SDLC.

---

## 5. Explore mode vs stabilize mode

Do not use the same rules for every phase.

### Explore mode

| Property | Rule |
|---|---|
| Goal | learn quickly |
| Code quality | rough is acceptable |
| Scope | broader experiments allowed |
| Tests | lightweight probes or golden examples |
| Output | notes, screenshots, traces, candidate designs |
| Human review | frequent direction checks |

### Stabilize mode

| Property | Rule |
|---|---|
| Goal | make selected behavior safe to ship |
| Code quality | maintainable and reviewable |
| Scope | narrow, explicit, approved |
| Tests | required behavior contracts |
| Output | small diff, evidence, risk note |
| Human review | final engineering review |

Example:

```text
"Try three ways to implement local voice activity detection."
  -> explore mode

"Make the selected VAD implementation production-ready."
  -> stabilize mode
```

---

## 6. Tests as the stability layer

When code is easy to rewrite, tests become more important.

Reason:

```text
tests preserve behavior while agents rewrite implementation
```

Useful agentic tests are often behavior-level:

- user journey tests
- API contract tests
- CLI smoke tests
- event-stream contract tests
- artifact-shape tests
- model-independent harness tests
- hardware observable-state tests

For OpenClaw-style systems:

| Area | Useful contract |
|---|---|
| Gateway RPC | request/response schema and event ordering |
| App SDK | normalized event shapes and wait/cancel behavior |
| cron | invalid schedules rejected before job creation |
| node transport | node command must be declared and allowed |
| tool policy | denied tools fail closed |
| system prompt | expected sections present without leaking secrets |

The test should answer:

```text
What must remain true if the implementation changes?
```

---

## 7. Intent documentation

Tests say what works.

Code says how it works.

Specs say what the system should do.

Intent explains why.

Agents need intent because they do not have durable product judgment unless you write it down.

Good intent docs include:

- why this design exists
- alternatives rejected
- tradeoffs accepted
- what must not be optimized away
- what future work is intentionally deferred

Example:

```markdown
# Intent: Gateway RPC Event Normalization

We normalize raw Gateway frames in the App SDK because external apps need a
stable event contract. Apps should not parse internal runtime frames directly.

Rejected alternative:
- expose raw frames only

Reason:
- raw frames create fragile UI integrations and make runtime changes risky

Must preserve:
- unknown raw frames remain available for advanced users
- stable event envelope stays versioned
```

This is high-value context for future agents.

---

## 8. Specs must evolve

A static spec is often wrong after implementation begins.

Agentic development reveals:

- API edge cases
- missing permission states
- testability constraints
- model behavior issues
- UI states not considered
- hardware timing problems

Continuous spec rule:

```text
Every meaningful implementation discovery should update:
- acceptance criteria
- non-goals
- constraints
- test plan
- open risks
```

This is not bureaucracy.

It preserves learning.

---

## 9. Human taste becomes the limiter

When code arrives faster than external feedback, judgment becomes the bottleneck.

Taste means knowing:

- what good looks like
- which complexity is not worth it
- when a prototype is lying
- when UX is awkward
- when an abstraction is premature
- when a test is too brittle
- when security risk is being hand-waved

Agents amplify taste.

They do not replace it.

Better engineers get more from agents because they:

- frame tasks precisely
- constrain the search space
- detect weak answers faster
- recognize accidental complexity
- identify missing verification

---

## 10. Automate the easy stuff

Good automation targets:

- formatting
- linting
- test selection
- smoke test execution
- dependency checks
- docs build checks
- API schema generation
- screenshot capture
- event fixture replay
- log summarization

Repeated lessons should become:

```text
habit -> checklist -> skill -> hook -> CI gate
```

Example:

```text
Agent repeatedly forgets to run mkdocs build.
  -> add docs-build skill
  -> add final-answer evidence check
  -> add CI gate
```

---

## 11. Dual-mode agent design

A practical coding agent should support two explicit modes.

### Explore mode

Purpose:

```text
learn quickly, compare options, surface hidden constraints
```

Allowed behavior:

- build throwaway prototypes
- compare approaches
- run quick probes
- produce notes and tradeoff tables
- ask for human direction before stabilizing

Required output:

```text
what was tried
what was learned
which option is recommended
what evidence supports it
what should be discarded
```

### Stabilize mode

Purpose:

```text
turn selected behavior into reviewable, maintainable code
```

Required behavior:

- update spec
- add or update tests
- keep diff scoped
- run verification
- document remaining risk
- produce review-ready summary

Required output:

```text
files changed
tests run
evidence captured
scope changes
known risks
next action
```

---

## 12. OpenClaw mapping

In an OpenClaw-style runtime:

| SDLC concern | Runtime primitive |
|---|---|
| Explore mode | isolated session or sandbox workspace |
| Stabilize mode | main project session with stricter tools |
| Tests as contracts | tool execution plus captured run output |
| Intent docs | workspace bootstrap files or project docs |
| Spec sync | session memory and project markdown updates |
| Scope discipline | file policy, diff review, approval hook |
| Evidence | artifacts, logs, screenshots, run events |
| Human taste | approval UI, dashboard, review surfaces |
| Long-running work | cron, sessions, task ledger |

Useful command vocabulary:

```text
/explore "Try three possible implementations"
/choose "Select option B and explain why"
/stabilize "Make option B production-ready"
/verify "Run the contract checks"
/review "Inspect the diff and risks"
```

The runtime should record mode in run metadata.

Reviewers need to know whether they are looking at experiment output or ship-ready output.

---

## 13. On-device AI example

Task:

```text
Improve wake-word responsiveness without increasing false positives.
```

Explore mode:

```text
1. Try three VAD/wake-word pipeline variants.
2. Measure latency on short sample clips.
3. Track CPU/GPU usage.
4. Record false-positive behavior on noisy clips.
5. Recommend one candidate.
```

Stabilize mode:

```text
1. Update the selected pipeline only.
2. Add regression clips.
3. Add latency threshold test.
4. Add false-positive check.
5. Document runtime limits.
6. Run on target Jetson or representative device.
```

Exploration discovers behavior.

Tests turn discoveries into contracts.

Stabilization prevents prototype debt.

---

## 14. Hardware bring-up example

Task:

```text
Get ESP32-C6 Zigbee NCP talking to Jetson over UART.
```

Explore mode:

```text
1. Confirm serial device candidates.
2. Try baud rates and flow-control assumptions.
3. Capture logs for each attempt.
4. Compare host-side and firmware-side symptoms.
5. Stop before changing firmware and kernel settings together.
```

Stabilize mode:

```text
1. Document working wiring and serial config.
2. Add a bring-up checklist.
3. Add a smoke command.
4. Save known-good logs.
5. Add troubleshooting table for common failure states.
```

The agent skills from Lecture 29 prevent multi-variable chaos.

This SDLC lets you explore enough to learn.

---

## 15. Minimal artifact set

For serious projects, preserve:

```text
SPEC.md
INTENT.md
TEST_PLAN.md
DECISIONS.md
RISKS.md
RUNBOOK.md
```

Minimal version:

```text
SPEC.md      what should be true
INTENT.md    why decisions were made
TESTS        executable behavior contracts
```

If the agent can read only three things before changing code, give it:

```text
current spec
relevant tests
current intent
```

---

## 16. Failure modes

| Failure | What happened | Fix |
|---|---|---|
| Prototype shipped | exploration code went to production | require stabilize mode before merge |
| Spec drift | implementation taught new facts, docs stayed old | update spec during work |
| Test theater | tests assert implementation details only | write behavior contracts |
| Infinite exploration | agent keeps trying ideas without converging | timebox and force recommendation |
| Over-process | agent writes bureaucracy for tiny tasks | scale process to risk |
| Weak taste | agent optimizes local code but worsens product | human review for UX/architecture/security |
| Hidden maintenance | generated code owns long-term support burden | record owner, risks, and rollback path |
| Security blind spot | code is cheap, exploit cleanup is not | enforce policy and review threat paths |

Dangerous confusion:

```text
fast generation != low total cost
```

---

## Mini-lab

Add two commands or skills to your agent workspace:

```text
/explore
/stabilize
```

`/explore` output:

```text
- options tried
- evidence gathered
- recommendation
- discarded ideas
- follow-up questions
```

`/stabilize` output:

```text
- updated spec/intent
- tests added or updated
- verification command output
- scoped diff summary
- risks and rollback
```

Test with:

```text
Explore three ways to improve OpenClaw App SDK event replay.
Then stabilize the best one.
```

---

## Key takeaways

- Cheap code changes software process, but it does not remove engineering cost.
- Implementation can be an exploration tool.
- Tests and intent are durable assets.
- Specs should evolve as implementation reveals reality.
- Human taste and domain experience become more important when code arrives faster.
- Agent skills provide the stabilization discipline that exploration alone lacks.
- The useful pattern is dual-mode: explore fast, then stabilize with evidence.

---

## References

- Drew Breunig, "10 Lessons for Agentic Coding": [https://www.dbreunig.com/2026/05/04/10-lessons-for-agentic-coding.html](https://www.dbreunig.com/2026/05/04/10-lessons-for-agentic-coding.html)
- Addy Osmani, "Agent Skills": [https://addyosmani.com/blog/agent-skills/](https://addyosmani.com/blog/agent-skills/)
- Lecture 29 - Agent Skills: [Lecture-29.md](Lecture-29.md)
- Lecture 19 - OpenClaw Agent Loop: [Lecture-19.md](Lecture-19.md)
- Lecture 22 - OpenClaw App SDK: [Lecture-22.md](Lecture-22.md)

---

*Next: [Lab 01 - Research Agent](Lab-01-Research-Agent.md)*
