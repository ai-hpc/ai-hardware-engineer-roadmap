# Provider protocol lab: real SDKs and local tool-call fixtures

**Time:** 60–90 minutes. **Prerequisites:** Python functions, dictionaries, JSON and [Lecture 08](Lecture-08.md). **Outcome:** explain exactly what a tool loop sends and receives, then reject malformed/incomplete requests without confusing them with successful work.

This lab runs actual Anthropic and OpenAI Python SDKs through their HTTP request/response code. A local `MockTransport` supplies fictional provider replies. The tool really adds two validated integers. **No network request, real model, API account, GPU or paid API is used by the tests/demo.** A passing fixture test verifies integration logic, not the hosted model's intelligence or a live provider's acceptance of every fixture.

## 1. Start with the portable contract

From this directory (`Lectures/`):

```bash
python3 -m unittest discover -s labs -p 'test_*.py' -v
```

Expected: **8 tests pass** using only the standard library. Read [tool_contracts.py](labs/tool_contracts.py): the tool allowlist, exact argument keys, integer type/bounds, strict duplicate-key JSON parser and whole-batch call budget are separate checks.

Predict these results before looking at the tests:

| Input | Expected |
|---|---|
| `add_integers(a=2, b=3)` | `{"ok": true, "sum": 5}` |
| `a=true, b=3` | Invalid arguments, even though Python `bool` subclasses `int` |
| `a=2, b=3, command="..."` | Extra key rejected |
| Tool named `shell` | Unknown tool rejected |
| JSON repeats `"a"` | Rejected; no silently chosen first/last value |
| Two calls when only one remains | Whole batch rejected before dispatch |

## 2. Install the isolated SDK fixture environment

The lock was built and executed on **CPython 3.12, Linux x86-64**. It pins 20 packages with wheel hashes, including Anthropic 1.15.0, OpenAI 3.35.1 and Pydantic 2.14.1. The downloaded wheels totaled about 8 MB in the authoring environment. Other Python/platform combinations may require different wheels; do not delete hashes to hide that incompatibility.

```bash
python3.12 -m venv /tmp/course-provider-contracts-venv
/tmp/course-provider-contracts-venv/bin/python -m pip install \
  --require-hashes --only-binary=:all: -r labs/requirements-provider.lock
/tmp/course-provider-contracts-venv/bin/python -m unittest discover \
  -s labs/optional -p 'test_*.py' -v
/tmp/course-provider-contracts-venv/bin/python labs/demo_provider_contracts.py
```

Expected: **20 optional SDK tests pass**. The demo prints two runs, each with answer `5` and two captured request bodies. It prints no real credentials or HTTP authorization headers. The fictional string `fixture-model` is a fixture label, not a model to purchase or select in an API console.

The standard-library runner intentionally does not descend into `labs/optional`; dependency-bearing checks are a separate command. See [recorded validation](labs/evidence/validation.json) and [captured local run](labs/evidence/provider-demo.json).

## 3. Follow the same action through two protocols

Read [provider_loops.py](labs/provider_loops.py). Both loops enforce a turn count and a total unique call-ID budget. They retain provider continuation data rather than flattening everything to text.

| Step | Claude Messages | OpenAI Responses |
|---|---|---|
| Tool definition | `input_schema`, `strict` | `parameters`, `strict`, type `function` |
| Requested action | `tool_use` content block | `function_call` output item |
| Result association | `tool_use_id` matches block ID | `call_id` matches the call ID, not its output-item ID |
| Result container | User message with tool-result blocks | `function_call_output` input item |
| State carried forward | Complete assistant content blocks | Complete output items, including reasoning items |

The SDK fixture tests inspect the second outgoing request. This catches a common bug that a mocked `model.generate()` function would miss: sending the right answer in the wrong protocol shape.

Current Anthropic 1.15.0 requires an **`httpx2.Client`**, while OpenAI 3.35.1 uses **`httpx.Client`**. The test deliberately passes the wrong client and confirms the SDK rejects it. The Anthropic SDK also rejects the removed `temperature` argument before making a request. An old tutorial that imports successfully may still fail at construction or call time.

## 4. Completion and validation are different checks

A `max_tokens`/incomplete response is not a complete tool request. A refusal is not a successful answer. This minimal Claude loop stops on `pause_turn`; supporting server-side tool continuation is an explicit extension, not something to infer from text.

For Responses, an explicit boolean `end_turn: false` is treated as continuation within the same finite budget, including a text-only response. A malformed value such as the string `"false"` is rejected. This defensive behavior is based on the audited runtime case in [Lecture 35](Lecture-35.md); the fixtures do not establish which live models emit the field. An absent signal uses the conventional complete-response/no-function-call terminal rule. Built-in server tools are outside the adapter's supported item set.

The structured-output tests exercise `messages.parse(output_format=...)` and inspect the generated `output_config.format`. One supplies valid JSON with a negative bandwidth. Local Pydantic range validation rejects it. Schema-shaped output is not automatically physically plausible or factually supported.

These adapters are teaching implementations: no durable restart, streaming assembly, external writes, async fan-out, total dollar cap or global wall-clock deadline. Tool calls are bounded local arithmetic; transport timeouts alone are not a global deadline. Extend one responsibility at a time with a failure test.

## 5. Guided variations and worked answers

1. **Two calls:** change the first fixture to request `(2, 3)` and `(7, -2)`. **Answer:** two results with distinct matching IDs, each sum 5; one grouped Claude user result message. Do not return only the first tool result.
2. **Continuation state:** remove the reasoning item from the second Responses request. **Answer:** the preservation test fails; reducing state to final text and calls loses provider continuation material.
3. **Refusal plus call:** add a refusal message beside a function call. **Answer:** fail before dispatch under this adapter's policy; do not execute the call because a different item looked valid.
4. **Turn exhaustion:** emit `end_turn: false` for every response. **Answer:** finite turn-budget failure, never a fabricated final success.
5. **Units:** a source says 40 Gb/s. Is `bandwidth_gb_s=40` correct because it parses? **No:** it is 5 GB/s before any protocol overhead; preserve original units and provenance.
6. **Replay:** return a repeated call ID on the next model turn. **Answer:** reject it. For a remote write, durable idempotency would additionally be required across process restarts.

## 6. Independent task

Add a new pure tool `scale_length(value_mm, scale_num, scale_den)` with a bounded rational result, then test zero denominator, extra fields, boolean input, unknown tool name and two calls in one turn. Do not execute generated code. Keep the operation's units in its schema/result and show a worked `100 mm × 1/2 = 50 mm` trace.

Submit the tests, captured fictional request bodies, exact package lock and a short statement separating **tested SDK protocol behavior**, **computed tool results**, and **unmeasured model behavior**. Full credit requires an intentional failure test and correct explanation of why the two providers require different result envelopes.

Sources: [Claude tool-result protocol](https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls), [Claude structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs), [OpenAI function calling](https://developers.openai.com/api/docs/guides/function-calling). Source revisions and metadata are recorded in [the evidence note](Lecture-Source-Evidence.md).
