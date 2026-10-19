# Lecture 15 — Agent Patterns with Measurable Stop Conditions

**Course:** [AI Agent Development 2026](../Guide.md) | **Previous:** [Lecture 14](Lecture-14.md) | **Next:** [Lecture 16](Lecture-16.md)

**Outcome:** choose a control-flow pattern, enforce a finite budget, and decide success using evidence outside the generated prose.

## 1. Three patterns

| Pattern | What happens | Useful when | Main cost |
|---|---|---|---|
| ReAct-style tool loop | Model proposes an action; runtime executes an allowed tool and returns the observation | The next step depends on new evidence | Repeated model calls and changing context |
| Plan then execute | A proposed dependency graph is validated and executed | Dependencies and intermediate artifacts are useful to inspect | Plan validation and replanning |
| Critique then revise | A candidate is checked and a bounded revision is attempted | A reliable check can expose a repairable defect | Extra calls; correlated critic mistakes |

These describe application control flow. They do not require exposing a model's private reasoning. Request a short action rationale or testable plan where it helps the user, then inspect the actual actions and results.

## 2. Run a bounded tool loop first

Run the [provider contract lab](Provider-Contract-Lab.md). It uses real Anthropic and OpenAI SDKs with local fixture responses and a genuine bounded integer-addition tool. No API key, web search, GPU or arbitrary code execution is needed.

Trace one run:

```text
user: add 2 and 3
model fixture: call add_integers(a=2, b=3), call ID c1
runtime: validate name, exact keys, types, bounds and budget
runtime: execute Python addition -> {ok: true, sum: 5}
runtime: return result correlated with c1 using the provider's protocol
model fixture: final answer "5"
```

The tool really computes 5; the fixture supplies the final sentence. Therefore the test establishes dispatch/protocol behavior, not a model's ability to select the right tool. Change the arguments to `true` and `3`: Python considers booleans integer subclasses, so the validator deliberately requires exact `int` types.

Do not implement a “safe calculator” with `eval`, even with `__builtins__` removed. Do not call `exec` on generated Python and label it a sandbox. These are interpreter entry points, not containment. Real code execution needs a separate process/environment with resource limits and carefully scoped filesystem/network access.

## 3. Validate a plan as data

A model-generated plan is a proposal. Before execution, check a strict schema, unique bounded task IDs, allowed operation names, existing dependencies, absence of cycles, maximum node count and current authorization. Execute a ready node only after its dependencies have verified successful outcomes. A list's presentation order is not necessarily a topological order.

**Worked plan:**

```text
A = retrieve authorized manual passages
B = identify the applicable firmware revision; depends on A
C = produce a cited answer; depends on A and B
```

If A fails, B and C remain blocked. They are not successful merely because a model can produce text. If the plan is listed as C, A, B, a scheduler may still execute A, B, C after validating the graph. If A depends on C, the cycle must be rejected before any action runs.

Store each task's state (`pending`, `running`, `verified_success`, `failed`, `blocked`) and artifact references. Replanning must preserve action IDs already committed, so a retry cannot silently repeat an external write. Use the [executed local RAG/workflow lab](../Local-RAG-Workflow-Lab/Guide.md) for durable state and failure injection.

## 4. Critique is evidence only when the check is meaningful

A model critic can identify mistakes, but often shares the generator's blind spots. Separate its suggestion from an independent check: unit assertions for code, labeled answers for extraction, citation support for RAG, or a measured receipt for an action.

Never use `if "PASS" in critique`: “DO NOT PASS” satisfies it. An exact structured decision fixes parsing, but it still does not make the critic correct. Measure false acceptance/rejection against human or task-specific labels.

A bounded revision algorithm should retain each candidate and its measured score. If no attempt passes the required checks, return a failed/unknown status with the best *measured* candidate if that is useful. Returning the last attempt and calling it “best” is unjustified unless scores show that.

**Example:** attempt scores are 0.7, 0.9, 0.8 and the pass threshold is 0.95. None passes. Candidate 2 is best on this declared metric; candidate 3 is merely last. Choosing the maximum on a tiny development set can overfit, so evaluate the final revision policy on held-out tasks.

## 5. Budgets must be enforced where work happens

| Limit | Where to enforce | Common ineffective version |
|---|---|---|
| Model turns | Before every model request | Counting only calls to an outer wrapper |
| Tool calls | Admit the whole proposed batch before dispatch | Checking after the first side effect |
| Wall time | Deadlines passed through model and tool transports | A timer that logs lateness but never cancels |
| Money | Reserve a conservative bound, then reconcile actual usage | A `total_cost = 0` field never updated |
| Output/state size | Provider limit plus response/parser/storage bounds | Truncating text after executing it |

A max-turn cap limits one source of work; it does not cap a slow tool, model output cost or asynchronous child work. The teaching adapters set transport timeouts and turn/call limits, but do not claim a hard global deadline or dollar budget. Keep this scope visible when extending them.

Retries can duplicate effects after a timeout. Correlation IDs match a response to a call; an idempotency key plus durable server-side deduplication protects a logical write across retries. These are different jobs.

## Practice and answers

1. Run the lab; add a fixture that asks for three tools with a two-call budget. **Expected:** reject the batch before any execution.
2. Validate the dependency cycle A → B → C → A. **Expected:** reject before running A; do not mark a skipped dependency as complete.
3. Change a critic response from `PASS` to `NOT PASS`. **Expected:** substring matching fails this test; structured exact validation still needs an independently meaningful criterion.
4. Compare a direct answer, one tool loop and a critique loop on the same frozen tasks. Report every attempt, success rate, latency and calls per successful task. **There is no predetermined winning pattern.**
5. Disconnect after a write reaches the server but before the reply arrives. **Expected design:** recover the action receipt using its stable idempotency key; do not infer failure from a missing reply.
