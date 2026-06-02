# Lecture 39 - Agent Skills Eval: Benchmarking SKILL.md Files

**Course:** [Agentic AI & GenAI](../Guide.md) | **Previous:** [Lecture 38](Lecture-38.md) | **Next:** [Lecture 40](Lecture-40.md)

---

Skills are **code-adjacent infrastructure**.

If a skill changes how an agent behaves, it **needs tests**.

Agent Skills gives agents portable, version-controlled workflows through folders such as:

```text
my-skill/
  SKILL.md
  references/
  scripts/
  assets/
  evals/
```

But a `SKILL.md` file can **look good and still fail** in practice.

The right question is not:

```text
Does this skill read well?
```

The right question is:

```text
Does this skill measurably improve the agent on the task it claims to support?
```

`agent-skills-eval` is a **test runner** for that question.

It runs the same eval prompt twice:

```text
with_skill
without_skill
```

Then a **judge model** grades both outputs against expected behavior and assertions.

That gives you **evidence-backed pass/fail** rather than subjective prompt review.

---

## Learning objectives

By the end of this lecture, you should be able to:

1. Explain why skills need regression tests.
2. Understand the `with_skill` versus `without_skill` baseline pattern.
3. Write basic `evals/evals.json` cases for a skill.
4. Interpret judge-graded pass/fail results.
5. Use deterministic assertions for tool-call skills.
6. Understand the artifact layout produced by `agent-skills-eval`.
7. Design CI gates for OpenClaw-style skill repositories.
8. Identify common failure modes in LLM-judged skill evaluation.

---

## 1. Why skill evaluation matters

Agent skills package **procedural knowledge**.

Examples:

- triage GitHub issues
- summarize weather data
- operate a CRM
- write release notes
- inspect GPU traces
- follow a security review checklist
- translate kernels between DSLs

This is powerful because skills are **reusable**.

It is dangerous because skills can **silently degrade**.

A bad skill can:

- add irrelevant context
- over-constrain the model
- cause wrong tool calls
- increase token cost without quality lift
- make the agent slower
- hide stale instructions
- make a task worse than baseline

Without evaluation, every skill PR becomes a **taste debate**.

With evaluation, the conversation becomes:

```text
This skill improved 7/9 evals.
It regressed the pagination case.
The judge evidence points to missing tool-call criteria.
The report includes both outputs and timing.
```

That is a **better review standard**.

---

## 2. The baseline pattern

The core design is simple:

```text
same prompt
  -> target model without skill
  -> target model with skill
  -> judge compares both against assertions
```

This matters because **absolute output quality is not enough**.

You want to know **skill lift**:

```text
output with skill passes
output without skill fails
  -> skill likely helps

both pass
  -> skill may be unnecessary for this eval

both fail
  -> skill or eval is insufficient

with skill fails, baseline passes
  -> skill regressed behavior
```

The baseline mode prevents a common mistake:

```text
The skill produced a good answer, therefore the skill is useful.
```

Maybe the model already produced the same answer **without the skill**.

The eval needs to **measure the delta**.

---

## 3. Quickstart

Run directly with `npx`:

```bash
npx agent-skills-eval ./skills \
  --target gpt-4o-mini \
  --judge gpt-4o-mini \
  --baseline \
  --strict
```

Install if you want it in a project:

```bash
npm install agent-skills-eval
```

The key flags:

```text
--target
  model being evaluated

--judge
  model grading outputs

--baseline
  run without_skill as comparison

--strict
  enforce skill/spec validation
```

For OpenClaw contributors, this is the useful mental model:

```text
skills are tested like code
eval artifacts are reviewed like logs
reports are attached to PRs
```

---

## 4. Skill layout

A minimal evaluated skill looks like:

```text
skills/
  weather-summary/
    SKILL.md
    evals/
      evals.json
```

Example `SKILL.md`:

```markdown
---
name: weather-summary
description: Summarize weather forecasts and call out operational risks.
license: MIT
compatibility: Works with text-capable chat models.
---

When summarizing weather, identify the location, time range, precipitation risk,
temperature extremes, wind risk, and one practical recommendation.
```

Example `evals/evals.json`:

```json
{
  "skill_name": "weather-summary",
  "evals": [
    {
      "id": "storm-risk",
      "name": "storm risk summary",
      "prompt": "Summarize this forecast for an outdoor robotics test: thunderstorms after 2pm, wind gusts to 35 mph, high 91F.",
      "expected_output": "The response should identify thunderstorm timing, wind risk, heat risk, and recommend moving the test earlier or indoors.",
      "assertions": [
        "The output mentions thunderstorm risk after 2pm.",
        "The output mentions wind gusts or wind risk.",
        "The output gives a practical scheduling or safety recommendation."
      ]
    }
  ]
}
```

The assertions are the **contract**.

If they are vague, the eval will be **vague**.

---

## 5. Artifact layout

A run creates a workspace similar to:

```text
agent-skills-workspace/
  iteration-1/
    meta.json
    benchmark.json
    eval-basic/
      with_skill/
      without_skill/
    report/
      index.html
```

Important artifacts:

- `benchmark.json`: rolled-up pass/fail results
- `with_skill/`: output, timing, grading
- `without_skill/`: baseline output, timing, grading
- `report/index.html`: static report for review
- JSON/JSONL events: useful for dashboards or CI history

This **artifact-first design** is important.

You can **diff runs over time**.

You can attach reports to pull requests.

You can detect regressions after changing `SKILL.md`.

---

## 6. Judge model grading

The judge sees:

- eval prompt
- expected output
- assertions
- target model output

Then it grades pass/fail with evidence.

This is useful, but it is **not perfect**.

LLM judges can:

- be inconsistent
- over-reward fluent answers
- miss subtle tool-call failures
- leak bias from expected output phrasing
- pass outputs that satisfy wording but not intent
- disagree across model versions

Mitigations:

- keep judge temperature at `0`
- use explicit assertions
- add deterministic checks where possible
- review failures and unexpected passes manually
- pin judge and target model versions when possible
- keep raw artifacts
- rerun flaky evals before blocking a release

The judge is a **tool, not an authority**.

---

## 7. Tool-call assertions

Many useful skills are **not pure text-generation skills**.

OpenClaw-style skills often affect tool behavior:

- call `gh` commands
- invoke gateway RPCs
- query weather APIs
- inspect logs
- read files
- create issues
- update docs

For those, **text-only grading is weak**.

You want **deterministic tool-call assertions**:

```text
Did the agent call the expected tool?
Did it use the expected method?
Did it pass safe arguments?
Did it avoid a destructive command?
Did it include the required idempotency key?
```

Use LLM judging for **semantic quality**.

Use deterministic assertions for **protocol behavior**.

That split is **critical**:

```text
semantic correctness:
  judge model

tool contract correctness:
  deterministic assertions
```

---

## 8. Config file for CI

For repeated runs, use a config file:

```yaml
# agent-skills-eval.yaml
root: ./skills
workspace: ./agent-skills-workspace
baseline: true
target: gpt-4o-mini
judge: gpt-4o-mini
baseUrl: https://api.openai.com/v1
apiKeyEnv: OPENAI_API_KEY
include:
  - "skills/**"
exclude:
  - "**/draft-*"
concurrency: 4
layout: iteration
strict: true
report:
  enabled: true
  title: Agent Skills Report
targetParams:
  temperature: 0
judgeParams:
  temperature: 0
```

Run:

```bash
OPENAI_API_KEY=... npx agent-skills-eval --config agent-skills-eval.yaml
```

In CI, do **not rely only on console output**.

Persist:

- `benchmark.json`
- judge grading files
- generated report
- JSONL event logs

Those artifacts are the **evidence**.

---

## 9. OpenClaw skill testing workflow

OpenClaw-style systems can use skills for:

- GitHub issue triage
- release note generation
- weather and schedule planning
- Gateway runbooks
- node troubleshooting
- app SDK testing
- security review
- GPU performance analysis

A practical workflow:

```text
1. Contributor edits SKILL.md.
2. Contributor adds or updates evals/evals.json.
3. CI runs agent-skills-eval with baseline.
4. Report is uploaded as an artifact.
5. PR review checks pass rate, regressions, and judge evidence.
6. Maintainer decides whether the behavior change is acceptable.
```

Suggested PR rule:

```text
No skill behavior change without at least one eval proving the intended behavior.
No regression accepted without an explicit note explaining why.
```

This mirrors **how code should be tested**.

---

## 10. Designing good skill evals

A good eval is **narrow**.

It tests **one behavior**.

Bad eval:

```text
Prompt: "Use the GitHub skill to manage issues well."
Assertion: "The answer is good."
```

Good eval:

```text
Prompt: "Given these three issue titles and labels, identify which one is a bug, which one is a feature request, and which one needs more information."
Assertions:
  - The output classifies all three issues.
  - The output asks for reproduction steps for the ambiguous bug report.
  - The output does not propose closing any issue without evidence.
```

Good skill evals should cover:

- happy path
- ambiguous input
- missing data
- adversarial instruction
- unsafe action request
- tool-call behavior
- regression case from a real bug

Do **not make the eval suite huge** at first.

Start with the three cases **most likely to break user trust**.

---

## 11. Common failure modes

### The skill adds no lift

Both `with_skill` and `without_skill` pass.

Interpretation:

```text
The model may already know this task,
or the eval is too easy.
```

Fix:

- make the eval more specific
- test domain-specific constraints
- test tool-call behavior
- test edge cases

### The skill makes output worse

Baseline passes, skill fails.

Interpretation:

```text
The skill is too broad, stale, misleading, or over-prescriptive.
```

Fix:

- shorten the skill
- remove stale rules
- improve trigger description
- add counterexamples
- split into smaller skills

### The judge is unreliable

Repeated runs disagree.

Fix:

- lower temperature
- sharpen assertions
- add deterministic checks
- use a stronger judge
- manually review artifacts

### The eval tests formatting instead of behavior

Fix:

- assert outcome, not prose style
- use schema checks for structured outputs
- separate style tests from correctness tests

---

## 12. How this connects to earlier lectures

Lecture 29 introduced skills as **workflow discipline**.

This lecture adds the **missing test loop**:

```text
skill design
  -> eval prompt
  -> with/without comparison
  -> judge + deterministic assertions
  -> artifact review
  -> skill revision
```

Lecture 30 argued that tests are the **durable asset** in agentic software development.

Skill evals are the **tests for agent behavior**.

Lecture 35 showed skills for GPU kernel translation.

`agent-skills-eval` is how you test whether those translation rules actually improve outputs.

Lecture 37 used traces as evidence for performance claims.

Skill evals are **evidence for prompt/workflow claims**.

Same principle:

```text
No evidence, no claim.
```

---

## Mini-lab: Add evals to one OpenClaw-style skill

Pick one skill:

- GitHub issue triage
- weather planning
- Gateway troubleshooting
- node pairing runbook
- app SDK testing
- GPU trace analysis

Create:

```text
SKILL.md
evals/evals.json
agent-skills-eval.yaml
```

Run:

```bash
npx agent-skills-eval ./skills \
  --target gpt-4o-mini \
  --judge gpt-4o-mini \
  --baseline \
  --strict
```

Then write a short report:

```text
Skill:
Eval count:
Pass rate with skill:
Pass rate without skill:
Cases improved:
Cases regressed:
Judge concerns:
Deterministic assertions needed:
Decision:
```

If the skill does **not beat baseline**, do not ship it as-is.

Either **improve the skill** or admit the skill is unnecessary.

---

## Key takeaways

- Skills need tests because they change agent behavior.
- `with_skill` versus `without_skill` is the core pattern for measuring skill lift.
- LLM judges are useful when paired with explicit assertions and stored artifacts.
- Deterministic assertions are required for tool-call and protocol behavior.
- Artifact output makes skill reviews repeatable and CI-friendly.
- OpenClaw contributors can use this pattern to validate skill changes before merge.
- Good evals test concrete behavior, edge cases, safety constraints, and regressions.
- A skill that does not beat baseline is not automatically worth carrying.

---

## References

- agent-skills-eval repository: [https://github.com/darkrishabh/agent-skills-eval](https://github.com/darkrishabh/agent-skills-eval)
- agent-skills-eval documentation: [https://darkrishabh.github.io/agent-skills-eval](https://darkrishabh.github.io/agent-skills-eval)
- Agent Skills overview: [https://agentskills.io/home](https://agentskills.io/home)
- Lecture 29 - Agent Skills: [Lecture-29.md](Lecture-29.md)
- Lecture 30 - Agentic SDLC: [Lecture-30.md](Lecture-30.md)
- Lecture 35 - Agent Skills for GPU Kernel Translation: [Lecture-35.md](Lecture-35.md)

---

*Next: [Lecture 40 - ZAYA1-8B](Lecture-40.md)*
