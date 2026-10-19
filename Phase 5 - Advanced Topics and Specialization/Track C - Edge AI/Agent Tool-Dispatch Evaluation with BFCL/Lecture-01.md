# Tool-call evaluation: learn from BFCL, then test your own dispatcher

**Reviewed 18 October 2026.** Allow 60–90 minutes. Prerequisites: JSON, Python functions, a basic LLM inference loop, and [inference representation changes](../Edge%20LLM%20Inference%20Internals/Lecture-01.md). The supplied lab needs Python 3.10+ only. It runs arithmetic and an in-memory light simulator; it does not contact a model API or operate devices.

Your deliverable is a case-level report that distinguishes malformed output, invalid arguments, wrong calls, and wrong simulated outcomes. The lesson includes deliberately broken responses so a green test cannot mean “the evaluator never rejected anything.”

## 1. Separate the model, parser, and application

A tool-using assistant has several independently testable stages:

```text
request + tool schemas → prompt/template → model output
→ parser → schema validation → dispatcher → result/state → next turn
```

A model can produce valid JSON with the wrong action. A parser can reject a semantically correct answer. A dispatcher can interpret a valid argument in the wrong units. An accurate single-turn call can still fail a task that depends on prior results. Record evidence at each boundary rather than inferring the cause from one aggregate score.

For example, the request “set the office light to 60 percent” expects:

```json
[{"name":"set_level","arguments":{"room":"office","level":60}}]
```

These are distinct failures:

| Response | Failure |
|---|---|
| `level: "60"` | string where the contract requires an integer |
| `level: true` | boolean where the contract requires an integer |
| `level: 6000` | value outside the declared 0–100 range |
| `room: "kitchen"` | valid schema but wrong requested action |
| `name: "unlock_door"` | tool absent from the catalog |
| correct call executed twice | duplicate action, potentially different side effects |

Python's `isinstance(True, int)` is true. A JSON integer validator that uses only that check will incorrectly accept booleans. The supplied validator uses exact type checks for this teaching schema.

## 2. What the current BFCL benchmark covers

The [official leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) identifies **BFCL V4** and, at this review, states that its displayed results use commit `f7cf735` / package `bfcl-eval==2025.12.17`. That is a reproduction pin for those results, not a claim that every newer package produces the same benchmark.

The benchmark evolved from structured function-call evaluation to broader tasks. V2's **live** cases are community-contributed functions; “live” does not mean “multi-turn” or “invoke arbitrary live services.” V3 adds multi-turn/multi-step interactions. V4 adds web search, memory, and format sensitivity. Consult the [V2 description](https://gorilla.cs.berkeley.edu/blogs/12_bfcl_v2_live.html), [V3 description](https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html), and [V4 methodology](https://gorilla.cs.berkeley.edu/blogs/15_bfcl_v4_web_search.html) for their versioned categories and aggregation rules.

Useful single-turn distinctions include one available function, selection among functions, several independent calls, and calls across different functions. Irrelevance cases test when no available function should be called. They do not establish a universal ordering of difficulty or which category quantization will damage first; those are empirical questions for the specific artifact and task distribution.

**The lab below is not BFCL.** It uses six original course fixtures and its own explicit matching rules. Do not label its percentage a BFCL score or compare it with the public leaderboard. To reproduce BFCL, use the official dataset, adapters, scorer revision, category weights, and settings.

## 3. Structural matching and execution answer different questions

An abstract syntax tree (AST) represents the structure of a parsed expression. BFCL's AST-based evaluation checks supported function-call structures and acceptable arguments under its rules; it is not simply raw-string equality. Parsing is part of that pipeline. A parser defect can affect an AST result too. See the [original methodology](https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html).

A simulator can instead compare return values and state transitions. Distinct valid calls may produce the same result, and a structurally matching call may produce a wrong state when the implementation is defective. The task determines whether exact call identity, acceptable alternatives, final state, or a sequence of states is the right metric.

An AST score of 88% and execution score of 71% **does not prove a dispatcher bug**. The subsets, matching rules, environment, model behavior, and implementation can differ. Start with paired cases and inspect their raw outputs and state traces. Equal scores likewise do not prove a model defect.

For parallel calls, ordering can be irrelevant only when the task permits it. Compare a **multiset** rather than a set: `[add(1,2), add(3,4), add(1,2)]` contains an extra call even though its set equals the two-call reference. Across dependent turns, retain order and state. Do not sort state-dependent actions merely to improve a score.

## 4. Run a complete offline evaluator

From the repository root:

```bash
cd 'Phase 5 - Advanced Topics and Specialization/Track C - Edge AI/Agent Tool-Dispatch Evaluation with BFCL'
python3 labs/dispatch_eval.py
python3 -m unittest discover -s labs -p 'test_*.py' -v
# Deliberately bad responses: prints a report and exits with status 1.
python3 labs/dispatch_eval.py --responses labs/responses-regression.json
```

The [implementation](labs/dispatch_eval.py), [fixtures](labs/fixtures.json), [good responses](labs/responses-good.json), [bad responses](labs/responses-regression.json), and [tests](labs/test_dispatch_eval.py) are supplied. A model is not required to understand or verify the evaluator.

The fixtures cover six situations: a simple arithmetic call, selecting a light-setting call, two independent arithmetic calls, mixed independent tools, no-tool relevance judgment, and a two-turn light-setting task. The response artifact contains one raw JSON string per turn plus a canonical catalog hash. The same typed catalog drives validation; checking only catalog length would miss renamed tools or changed argument schemas.

The evaluator performs these steps:

1. Require unique fixture IDs, an exact response-ID match, and the expected catalog hash. Missing outputs cannot quietly reduce the denominator.
2. Parse strict JSON. Reject duplicate object keys, nonfinite numbers, unknown tools, missing/extra arguments, wrong types, and range violations. It never uses `eval` on generated text.
3. Compare validated calls. The explicitly parallel fixtures allow reordering but preserve multiplicity; turn order remains fixed.
4. Execute only local simulated functions and compare results and state after each turn.
5. Emit per-case failures, category totals, accuracy, and Wilson intervals. The teaching gate requires all six cases to pass and exits nonzero otherwise.

Recorded results:

| Artifact | Passed/total | Gate | Evidence |
|---|---:|---|---|
| Supplied correct fixture responses | 6/6 | pass | [reference report](labs/reference-report.json) |
| Four injected regressions | 2/6 | fail | [regression report](labs/regression-report.json) |

The four rejected cases contain a boolean level, an extra duplicate call, an unnecessary call, and an incorrect second-turn level. Ten tests also check malformed JSON, unknown functions, stale catalogs, missing cases, and ordering. These results validate the **evaluation mechanism**. They are not measurements of an LLM, quantization quality, or agent deployment success.

## 5. Turn the fixture harness into a model experiment

Keep the fixture format and replace the supplied response strings with actual model outputs. Save checkpoint revision, tokenizer/processor, exact chat template, tool catalog hash, runtime revision, weight/KV formats, decoding settings, maximum output length, seed where supported, and raw responses. A native tool-calling API may return structured objects instead of text; define and test the adapter boundary explicitly.

Preserve both raw output and any adapter transformation. If production permits fenced JSON or another repair, implement that as a separate, versioned parser and report strict versus repaired success. Quietly removing malformed fragments can hide runtime regressions or change the intended action.

For multi-turn cases, feed each actual simulated result back into the model. The provided response files are a replay fixture: they test scoring and state progression, not whether a model uses its history. Extend them with missing parameters, unavailable tools, delayed results, retries, and repeated request IDs. Compare catalog contents, not just their number of entries.

Make one controlled change at a time, such as FP16→INT4 weights, a different KV format, or a template update. If the float model runs on a larger machine, use its output as a quality reference and label its hardware; that does not establish a same-device speedup.

## 6. Use paired results and honest uncertainty

Overall accuracy is `passed cases / all required cases`. Category accuracy uses that category's cases. A macro average weights categories equally; a micro average weights cases equally. State which one you use. Official BFCL aggregation must follow its pinned scorer rather than this toy gate.

For the same case IDs, record four counts: both pass, reference-only pass (regressions), candidate-only pass (improvements), and both fail. The supplied comparison has **2 both-pass, 4 regressions, 0 improvements, 0 both-fail**. A net score change alone can hide serious regressions offset by unrelated improvements.

The six-fixture set is intentionally tiny. A category with 1/1 passing has a roughly **20.7%–100% Wilson 95% interval** under an independent Bernoulli sampling model. Even that interval does not account for correlated prompts or a nonrepresentative hand-picked set. One passing example cannot establish near-perfect real-world reliability. Use an independently held-out, representative test set, retain every failed case, and report counts as well as percentages.

Set acceptance criteria before inspecting a candidate: required behaviors, allowable regressions, latency/memory limits, and sample requirements. A fixed “2 percentage-point loss is safe” rule has no general meaning without sample size and task consequences. For repeated stochastic generation, record repetitions and avoid treating dependent samples as independent users.

## Check your understanding

1. Why reject `true` for `level`? **The declared JSON contract requires an integer brightness; Python's bool/int inheritance must not widen it.**
2. Why can sorting calls be wrong? **Order can change state. Only explicitly independent calls may be compared without order.**
3. Why does a stale catalog hash fail before scoring? **The model and evaluator would otherwise be testing different tool contracts.**
4. Does 6/6 prove an edge model is accurate? **No model was run, and the fixtures are too small for that general claim.**
5. Can a lower execution score identify the failing component by itself? **No. Inspect the paired cases and stage-level evidence.**

Continue to [Qwen quantization](../Qwen%20Inference%20Optimization/Lecture-02.md) using an evaluator whose rejection behavior you have verified. For robot action schemas and timing, use the [VLA dataset/execution lab](../../Track%20D%20-%20Robotics/VLA%20Foundations%20to%20Frontier%202026/Dataset-and-Execution-Lab.md).
