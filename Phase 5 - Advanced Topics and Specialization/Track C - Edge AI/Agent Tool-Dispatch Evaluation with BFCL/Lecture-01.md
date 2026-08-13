# Lecture: Agent Tool-Dispatch Evaluation with BFCL

<div class="course-identity edge-ai" markdown="1">
<div class="course-identity__icon">BFCL</div>
<div markdown="1">
<p class="course-identity__eyebrow">Phase 5 · Edge AI · Lecture</p>
<p class="course-identity__title">Measure whether an edge LLM agent actually calls the right tool with the right arguments, and prove your quantization did not break it.</p>
<p class="course-identity__meta">Artifact: BFCL-style harness + category accuracy report · Measure: top-1 tool-call accuracy, argument-match rate, parity vs fp16 reference</p>
</div>
</div>

> *A 4-bit Qwen on a Jetson is only useful if it still picks the right tool.*

An on-device agent ships when two things are true: **it fits the device, and it calls tools correctly**. Phase 5's [Qwen Inference Optimization](../Qwen%20Inference%20Optimization/README.md) covers the first half — quantization, KV cache, decode optimization on Jetson. This lecture covers the second half — the **measurement framework** that decides whether each of those optimizations silently broke the agent.

The standard benchmark is **BFCL** — the **Berkeley Function-Calling Leaderboard**. It is to LLM tool dispatch what LIBERO is to VLA action parity: a fixed, public, category-broken-down evaluation that lets you compare a candidate model (or a candidate runtime configuration) against a reference on exactly one axis — *does the function call match the ground truth*.

This lecture treats BFCL as a **hardware-quality gate**, not a leaderboard chase: you build a tiny BFCL-style harness, run it against your edge model, and use the **per-category accuracy table** to gate your optimization ladder.

**Layer mapping:** L3-L5. Sits between the inference runtime (L3-L4) and the agent harness above it (L5).

**Role targets:** Edge AI Engineer · Edge Inference Optimization Engineer · Agent Harness / Runtime Engineer · Embedded LLM Eval Engineer.

**Prerequisites:**

* Phase 5 — Edge AI — [Edge LLM Inference Internals](../Edge%20LLM%20Inference%20Internals/Lecture-01.md) — you need to already know how a transformer dispatches.
* Phase 5 — Edge AI — [Qwen Inference Optimization](../Qwen%20Inference%20Optimization/README.md), particularly [Lecture 2 — Quantizing Qwen3-4B to Q4](../Qwen%20Inference%20Optimization/Lecture-02.md). BFCL is what tells you whether AWQ-INT4 cost you tool-call accuracy.
* Phase 4 Track B — [Jetson Real-Time Inference](../../../Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/1.%20Nvidia%20Jetson%20Platform/Orin-Nano-Real-Time-Inference/Guide.md) for the deployment target.

**What comes after:** the [VLA Action-Parity Harness](../../Track%20D%20-%20Robotics/VLA%20Optimization%20and%20Action-Parity%20Harness/Lecture-02.md) — the same idea applied to embodied policies, where the "action" is a 7-DoF command instead of a JSON tool call.

By the end of this lecture you should be able to:

* explain the five BFCL categories (simple, multiple, parallel, parallel_multiple, multi-turn) and which one fails first under edge quantization
* tell the difference between AST scoring and executable scoring, and pick the one your repo actually needs
* derive a tool-call argument-match rate that decomposes into the failures you can act on (wrong tool, wrong arg name, wrong arg type, wrong arg value, hallucinated tool, missing required arg)
* stand up a minimal BFCL-style harness against your own `ToolDispatcher` catalog and produce a category accuracy table
* define a parity budget for "the edge model is allowed to ship" instead of guessing
* tie a regression in one category back to a specific runtime change — quantization recipe, prompt template, KV-cache precision, decode sampler

---

## 1. The agent tool-dispatch loop, end to end

Strip the marketing away and an LLM agent tick is:

```text
user turn ──► prompt assembler ──► tokenizer ──► LLM forward pass
                  ▲                                     │
                  │                                     ▼
                  │                              sampling / decode
                  │                                     │
                  │                                     ▼
                  │                          raw output (text or JSON)
                  │                                     │
                  │                                     ▼
                  │                              tool-call parser
                  │                                     │
                  │                                     ▼
                  │                              argument validator
                  │                                     │
                  │                                     ▼
                  │                          dispatcher / ToolDef table
                  │                                     │
                  │                                     ▼
                  │                              tool invocation
                  │                                     │
                  └────────────── result + new turn ◄───┘
```

Everywhere a BFCL eval can catch a regression:

| Stage | Failure the eval should catch | BFCL signal |
|-------|-------------------------------|-------------|
| Prompt assembler | tool catalog drift — the prompt advertises a tool the dispatcher does not implement, or vice versa | category accuracy drops *and* an `unknown_tool` error appears in failure breakdown |
| Tokenizer | special-token handling for tool start/end markers regresses after a template change | AST parse failures spike with no model change |
| LLM forward pass | quantization or KV-cache precision loss biases the function-call logits | per-category accuracy drops, often `multiple` and `parallel` first |
| Sampling / decode | non-greedy sampler picks an alternate but valid-looking call | accuracy looks fine at temperature 0, drops at temperature > 0 |
| Tool-call parser | parser is brittle to whitespace / JSON quoting / trailing commas | AST score is healthy on the model output but executable score is not |
| Argument validator | the model emits args of the right shape but wrong types | argument-match rate falls below tool-name-match rate |
| Dispatcher | runtime tool catalog has drifted from the schema the model was prompted with | tool-name-match passes, argument-match fails on an enum field |

The point of running BFCL — or your own BFCL-shaped harness — is to be able to point at one of those rows and say "this is the row that moved."

---

## 2. What BFCL is, concretely

The [Berkeley Function-Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) is a **public benchmark** from the Gorilla project at UC Berkeley. Each test case is a `(user prompt, available functions, ground-truth call)` tuple. The model is shown the prompt and the function schemas (in the prompting style the model expects — system prompt, JSON, XML, etc.) and is graded on whether its emitted call matches the ground truth.

### 2.1 Categories

The categories are not equally hard on edge hardware, and they fail in characteristic orders under quantization. From easiest to hardest in practice on a 4B-class quantized model:

| Category | Definition | What it stresses | Fails first under |
|----------|------------|------------------|-------------------|
| `simple` | one user prompt → one function call from a single tool | basic tool-name + arg-name binding | nothing usually; a regression here is a sanity-check fail |
| `multiple` | one prompt, multiple tool schemas in the catalog, must pick one | tool selection precision | aggressive quant on the LLM backbone |
| `parallel` | one prompt → multiple calls to the same tool | argument enumeration and list-shape outputs | low-bit KV cache, short decode budgets |
| `parallel_multiple` | one prompt → multiple calls across multiple tools | both selection and enumeration at once | combined precision drops |
| `multi_turn` / `live` | multi-turn dialogues with tools that depend on prior turn state | prefix caching correctness, KV-cache eviction policy | paged-attention or prefix-cache bugs |
| `irrelevance` / `relevance` | the right answer is *do not call any tool* or *call exactly this one* | refusal behavior, tool-selection precision | over-aggressive prompt-compression that drops the catalog |

When you see "BFCL score dropped 4 pp on the new quantization" without a category breakdown, you are looking at a number that hides which row of §1 actually moved. Always report by category.

### 2.2 Scoring modes

BFCL has two scoring modes, and they catch different bugs:

* **AST scoring** — parse the model's emitted call as an abstract syntax tree, compare the function name and argument tree to the ground truth. Fast, deterministic, no live tools needed. Catches model errors but not parser errors. This is what you use in CI on every commit.
* **Executable scoring** — actually invoke the tool, compare side effects or return values. Catches parser errors, dispatcher drift, schema mismatches, and tool implementation bugs. Slower and stateful. This is what you run before promoting a checkpoint.

A model that scores 88% AST and 71% executable has a **parser/dispatcher drift problem**, not a model problem. A model that scores 71% AST and 71% executable has a **model problem**. Knowing which one you have is the difference between fixing it in an hour and chasing the wrong stage for a week.

### 2.3 Argument-match decomposition

A single AST mismatch is not actionable. Decompose it into the failure modes you can fix:

| Failure | Definition | Most common cause |
|---------|------------|-------------------|
| `wrong_tool` | called a tool that exists but is not the ground truth | tool selection collapsed under quantization, or two tools have near-identical names |
| `hallucinated_tool` | called a tool not in the catalog | prompt template lost the catalog section, or the model is regurgitating a tool from pretraining |
| `missing_required_arg` | a required arg is absent | aggressive prompt compression dropped the schema description |
| `extra_arg` | an arg not in the schema is present | model is hallucinating fields; often a sign of cross-tool confusion |
| `wrong_arg_type` | right name, wrong JSON type (e.g. `"true"` vs `true`) | usually parser-side; check executable score |
| `wrong_arg_value` | right type, wrong value (e.g. `"living_room"` vs `"livingroom"`) | enum normalization, or training-data drift |
| `wrong_arg_name` | right value, wrong key (e.g. `room` vs `entity`) | catalog drift between prompt and runtime — see §6 |

The deliverable artifact for this lecture is a CSV with one row per test case and these columns populated. Aggregate per category to a tiny table the maintainer can read in 30 seconds.

---

## 3. Why this lecture lives in Edge AI

BFCL is a model evaluation, but the action it gates is a *runtime decision*:

* "Can I ship Qwen3-4B at AWQ-INT4 on this Jetson, or does it need to be Q5_K_M?" → BFCL accuracy parity vs fp16 reference, per category.
* "Can I enable FP8 KV cache to fit a 4096-token context?" → BFCL `multi_turn` parity, because long prefixes are where KV-cache precision actually shows up.
* "Can I compress the system prompt by removing the per-tool examples?" → BFCL `multiple` and `irrelevance` parity, because that is what the examples were buying you.
* "Can I switch from the published prompt template to a tighter ChatML variant?" → BFCL AST parity, because parsers tend to be template-specific.

None of those questions are answerable from a perplexity number, MMLU score, or vibes. They are answerable from a BFCL-style category accuracy table run on the actual edge artifact.

---

## 4. A minimal BFCL-style harness

You do not need to reimplement Berkeley's harness to get the engineering value. A **200-line Python script** is enough to gate your own optimization ladder. The skeleton:

```text
harness/
├── manifest.yaml             # categories, seed list, tool catalog manifest, tolerance budgets
├── fixtures/
│   ├── simple/000.json       # {prompt, available_tools, ground_truth_call}
│   ├── multiple/...
│   ├── parallel/...
│   └── multi_turn/...
├── adapters/
│   ├── openai_compatible.py  # talks to vLLM, llama.cpp server, TRT-LLM, etc.
│   └── local_llama_cpp.py    # in-process for tiny models
├── parser/
│   └── ast.py                # tolerant JSON / function-call parser, returns ToolCall AST
├── score/
│   ├── ast_score.py          # ground-truth vs parsed AST → match + failure tag
│   └── exec_score.py         # invokes real ToolDef, compares side effects
├── report/
│   ├── per_case.csv          # one row per test, with all §2.3 failure tags
│   └── category_table.md     # the 30-second summary
└── gate.py                   # CI entry, exits non-zero if any category misses its budget
```

A useful first set of categories to ship is **simple, multiple, parallel, irrelevance, multi_turn**. Skip the live-tool categories until §5.

Five design choices worth defending up front:

1. **Adapters expose `complete(prompt, tool_schemas) -> raw_string`, nothing else.** This is what lets you grade vLLM, llama.cpp, and TensorRT-LLM with the same harness.
2. **The parser is tolerant on the *input* and strict on the *output*.** Accept the model's quirks (trailing commas, single quotes, code-fence wrappers); emit a strict `ToolCall` value the scorer can compare. Tag every leniency as a `parser_repair` so you can spot template regressions.
3. **The scorer never reaches into the model.** Logit-level checks are useful but belong in a different harness; here we grade emitted strings.
4. **Tolerance budgets live in `manifest.yaml`, per category.** Not a single overall number. A 2 pp drop in `simple` is a different failure from a 2 pp drop in `multi_turn`.
5. **The gate exits non-zero on failure.** Same contract as the [VLA action-parity gate](../../Track%20D%20-%20Robotics/VLA%20Optimization%20and%20Action-Parity%20Harness/Lecture-02.md#7-ci-gating) — a missing manifest entry or a stale reference SHA *fails* the gate; default for unknown is "do not ship."

---

## 5. Executable scoring against a real `ToolDef` catalog

AST scoring is enough to catch model regressions. Executable scoring catches the **silent integration bugs** that make production agents misbehave even when BFCL leaderboard numbers look fine.

The pattern, drawn from real edge-agent codebases:

1. The runtime owns a typed `ToolDef` table — one entry per tool, with a JSON schema for arguments and a Rust/Python implementation.
2. The BFCL prompt is generated *from* that table, not from a hand-maintained list. The harness asserts `len(prompt_catalog) == len(runtime_catalog)` and emits a `catalog_drift` failure if a tool advertised to the model has no implementation, or vice versa.
3. The dispatcher exposes a `simulate(call: ToolCall) -> side_effects` entry point. A successful executable test calls `simulate`, observes the side effects, and compares to a recorded reference trace.
4. A "fake" provider implementation (no real network, no real actuation) is mandatory for the home-automation-like tools, so the BFCL run is hermetic.

The regression this catches in practice: a contributor adds a new action to a typed tool's enum, updates the runtime, and forgets to update the catalog the model is prompted with. AST score is fine — the model never learns about the new action. Executable score drops on the test cases that need the new action because the runtime rejects calls that omit it. Same pattern in reverse — drop an action from the runtime, leave it in the prompt, and the model happily emits a call that the dispatcher refuses.

In real codebases this shows up as a recurring class of bug. The fix is the same in every codebase: **derive the BFCL catalog from the runtime, not from a hand-maintained constant, and write the regression test that locks the two together.** That single line of test is worth more than 5,000 BFCL cases.

---

## 6. Tolerance budgets for an edge ship

Same logic as the [VLA harness tolerance section](../../Track%20D%20-%20Robotics/VLA%20Optimization%20and%20Action-Parity%20Harness/Lecture-02.md#23-tolerance-budgets) — derive the budget from two floors and take the min:

1. **Reference-vs-reference floor.** Run the fp16 reference twice with different seeds. The difference is your noise floor — a candidate that exceeds it is statistically different, not just sampled differently.
2. **Product floor.** Ask the agent integrator what accuracy drop the product can absorb on each category. A 2 pp drop in `simple` may be a release blocker; a 2 pp drop in `parallel_multiple` may be fine if the product never uses it.

Example budget for a Jetson-class home-agent shipping at 4096-token context:

| Category | Reference fp16 (200 cases) | Candidate budget (absolute pp) | Why |
|----------|-----------------------------|-------------------------------|------|
| `simple` | 0.93 | -2 | one-call dispatch is the load-bearing path; any drop ships incidents |
| `multiple` | 0.86 | -3 | tool selection; modest drop tolerable if product has ≤8 tools |
| `parallel` | 0.78 | -4 | most products do not actually use parallel calls |
| `parallel_multiple` | 0.62 | -5 | aspirational; not required for v1 ship |
| `irrelevance` | 0.81 | -2 | refusal correctness; spurious tool calls are user-visible |
| `multi_turn` | 0.70 | -3 | long context — covered separately by KV-cache regression suite too |

The candidate ships when **all** rows are inside budget. A pass on the weighted average is not a pass — it hides the row that broke. Read by category, always.

---

## 7. Hardware-side stories the harness lets you tell

These are the optimization-ladder questions the BFCL harness was built to answer. Same set as in [Qwen Optimization Lecture 2](../Qwen%20Inference%20Optimization/Lecture-02.md), now with a measurement attached.

| Optimization | What you expect to move | What actually moves on a 4B-class model |
|--------------|-------------------------|-----------------------------------------|
| bf16 → fp16 weights | nothing | usually nothing; if `multiple` drops by >1 pp, suspect Inf/NaN-prone heads and check norm fusion |
| AWQ-INT4 backbone, LM head fp16 | small drop everywhere | typically -1 to -3 pp on `simple` and `multiple`, -3 to -5 pp on `parallel_multiple`; `multi_turn` rarely worse |
| INT3 backbone | larger drop | -4 to -8 pp on `simple`; not worth it for an agent product |
| FP8 KV cache | drop on long context only | `multi_turn` -2 to -5 pp depending on context length; `simple` unchanged |
| Speculative decoding, draft = 1B | none if accept-reject is correct | nothing on AST; latency improves; if AST drops, your verifier is wrong |
| Prompt-template change (ChatML → tight variant) | none if parser is template-aware | AST score moves before model — almost certainly a parser repair issue, fix the parser |
| Compress system prompt by dropping tool examples | drop on `multiple` and `irrelevance` | -3 to -7 pp on `irrelevance`; smaller on `multiple` |
| Reduce context window 8192 → 4096 | drop on `multi_turn` long cases | -5 to -10 pp on the long-prefix subset; flat on `simple` |
| Sampler temperature 0 → 0.3 | small randomness drop | -1 to -2 pp uniformly; if `simple` drops more, your model is over-confident-but-wrong on close ties |
| Replace published model with a distilled student | depends on student | re-run *all* categories; do not extrapolate |

The hardware engineer's job is to bring the candidate inside the budget at the lowest deployment cost. The harness is what lets you stop guessing.

---

## 8. Lab — Stand up a tiny BFCL gate

Suggested 1-2 day lab.

1. **Pick a target.** Default: Qwen3-4B-Instruct at fp16 (reference) and AWQ-INT4 (candidate), both served via vLLM or llama.cpp on your dev machine. If you have a Jetson AGX Orin available, do the candidate on the Jetson.
2. **Hand-write 30 fixtures.** Five per category from §2.1, plus the `irrelevance` set. Each fixture is a JSON file with `prompt`, `available_tools`, `ground_truth_call`. Use your own tool schemas — `home_control`, `set_timer`, `search`, etc. — not a public list. The point is to grade *your* agent, not Berkeley's.
3. **Implement the harness skeleton in §4.** Adapter, parser, AST scorer, per-case CSV, category table.
4. **Run reference vs candidate.** Produce two CSVs, one category table per model.
5. **Compute the failure decomposition.** For every miss, tag with one of the §2.3 failure modes. Aggregate to a per-category, per-mode table.
6. **Set tolerance budgets** in `manifest.yaml` derived from §6. Run `gate.py` and confirm it exits non-zero on at least one intentionally-bad candidate (e.g. drop the system prompt's tool catalog and watch `multiple` collapse).
7. **(Stretch)** Add executable scoring against a fake home provider. Show that a deliberately-drifted catalog (a tool in the runtime but not in the prompt, or vice versa) fails executable scoring while AST scoring passes.

Pass criterion for the lab: another engineer can clone your repo, run `python gate.py --candidate qwen3-4b-awq-int4 --manifest manifest.yaml`, and reproduce your pass/fail decision on the same hardware class. The artifact is the harness + the report, not a single number.

---

## 9. How this lecture connects to the rest of the roadmap

| What you finish | Where it goes next |
|-----------------|--------------------|
| Edge agent runs but tool-call accuracy is unknown | this lecture |
| You have a BFCL-style category table for fp16 vs INT4 | feed back into [Qwen Inference Optimization, Lecture 2](../Qwen%20Inference%20Optimization/Lecture-02.md) to pick the quantization that meets your budget |
| You have a BFCL-style harness | the same shape gates the [VLA action-parity harness](../../Track%20D%20-%20Robotics/VLA%20Optimization%20and%20Action-Parity%20Harness/Lecture-02.md) for embodied policies |
| You need to wire the harness into CI | see the [MLSys Stage 0 measurement discipline](../../Track%20G%20-%20ML%20Systems%20Engineering/Guide.md#stage-0-measurement-discipline) — same contract, different metrics |
| You need long-context multi-turn parity | combine with the [Long-Context MoE Foundation Training](../../Track%20A%20-%20GPU%20Infrastructure/Nvidia%20GPU/HPC%20Setup/Long-Context-MoE-Foundation-Training/07-Long-Context-Evaluation.md) evaluation patterns |

---

## Self-check

1. Your AWQ-INT4 candidate scores 89% overall vs the fp16 reference's 91%. The maintainer says "ship it." What is the single table you ask them to look at before agreeing, and what specifically would make you refuse even though the overall number passed?
2. A candidate's AST score drops by 5 pp after switching from the published Qwen prompt template to a custom ChatML variant. The weights did not change. What is the most likely stage of §1's table that moved, and what is the first thing you change in the harness — not the model — to confirm it?
3. Your `multi_turn` category drops 6 pp after enabling FP8 KV cache. `simple` is unchanged. Why is this consistent with FP8 KV being the culprit, and what one extra experiment in the harness pins it down without re-quantizing?
4. The executable score is 12 pp below the AST score on `multiple`. What kind of bug does that pattern point at, and which §2.3 failure tag would you expect to dominate the failure breakdown?
5. A contributor PR moves the BFCL overall number up by 2 pp but moves the `irrelevance` category down by 4 pp. Is this PR a win, a loss, or "depends" — and what specifically does it depend on?

---

## References

* Berkeley Function-Calling Leaderboard — [leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html), [Gorilla project](https://gorilla.cs.berkeley.edu/), [GitHub](https://github.com/ShishirPatil/gorilla)
* BFCL v1/v2/v3 dataset cards — [Hugging Face](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard)
* OpenAI function-calling format (the default schema many models train against) — [API docs](https://platform.openai.com/docs/guides/function-calling)
* Anthropic tool-use format — [API docs](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)
* "Gorilla: Large Language Model Connected with Massive APIs" — [paper](https://arxiv.org/abs/2305.15334) (origin of the dataset family)
* vLLM tool-call serving — [docs](https://docs.vllm.ai/en/latest/features/tool_calling.html)
* TensorRT-LLM tool-call paths — [docs](https://nvidia.github.io/TensorRT-LLM/)
* llama.cpp grammar-constrained sampling for tool calls — [GBNF grammars guide](https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md)
* For a real-world reference of "BFCL catalog derived from the runtime `ToolDef`" — see any production edge-agent repo that ships a typed dispatcher; the pattern is the same regardless of language.

---

## Next in this roadmap

* Previous in track: [Lecture — Edge LLM Inference Internals](../Edge%20LLM%20Inference%20Internals/Lecture-01.md)
* Sibling: [Qwen Inference Optimization](../Qwen%20Inference%20Optimization/README.md) — the optimization side of the same contract
* Cross-track: [VLA Optimization and Action-Parity Harness](../../Track%20D%20-%20Robotics/VLA%20Optimization%20and%20Action-Parity%20Harness/README.md) — the embodied analog
* Up: [Phase 5 — Edge AI Guide](../Guide.md)
