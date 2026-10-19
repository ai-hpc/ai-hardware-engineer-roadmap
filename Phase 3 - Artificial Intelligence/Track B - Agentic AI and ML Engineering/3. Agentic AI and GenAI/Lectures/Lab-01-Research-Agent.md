# Lab 01 — Build and test a bounded research workflow

**Track B · Agentic AI & GenAI** | [Index](README.md) | [Next: workflow orchestration](Lab-02-Multi-Agent-Pipeline.md)

**Time:** 60–90 minutes. **Outcome:** retrieve evidence, preserve citations, reject unsupported tool calls and report a failure. Start with the executed [local retrieval lab](../Local-RAG-Workflow-Lab/Guide.md), sections 1–3 and 8. Its deterministic baseline works without any account or model download.

The former lab silently substituted mock web search but still required an Anthropic model key. That was not an offline agent. This version makes the boundary explicit: local retrieval and quotation are executed; adding an LLM planner or live web search is an optional integration with its own credentials and evaluation.

## 1. Trace one concrete research task

Run the query `battery return threshold`. Follow it through the original fixture, trusted access/effective-date filter, FTS query, ranked result and quotation. Explain why the September 25-percent rule is selected and why the expired 10-percent rule and private note cannot enter this reader's index.

Write down the source ID, revision, quotation span and content hash. A citation that merely names a filename cannot detect that its content changed. The lab tests modified spans, invented quotations and stale revisions.

## 2. Separate a proposed tool call from execution

Read `dispatch` in [local_rag.py](../Local-RAG-Workflow-Lab/labs/local_rag.py). This accepted shape is data:

```json
{"name":"search_manual","arguments":{"query":"camera checkerboard","k":3}}
```

The application chooses the identity and available tool. A model cannot add an `operator` role, call a missing `write_file` capability, or expand `k` without validation. Test a boolean `true` for `k`: Python's boolean/integer relationship is a common accidental bypass, so the execution check requires an actual integer.

The hostile attachment contains a request to overwrite an approval record. It is corpus text, not a new tool definition. A future model adapter must keep that trust distinction; the deterministic baseline's behavior is not proof of LLM injection resistance.

## 3. Establish the baseline before adding a model

Run development queries, write down your intended change, then run the test split once. Report recall and abstention separately. Explain the retained `VIOLET-ONLY diagnostic` failure: a common query token retrieves an unrelated authorized paragraph. Correctly quoted text can still be an incorrect response.

The output is a source excerpt, not a generated literature review. Its limitation is useful: compare an eventual model against a baseline whose evidence and failure modes are visible.

## 4. Optional model-driven loop

Design a small adapter with two outputs: a structured tool proposal or a final answer with source references. Keep the dispatcher independent of the model client. Bound model turns, total tool calls, elapsed time and output size; a per-request timeout alone does not bound a multi-turn run.

For a Responses API integration, use the current [function-calling contract](https://developers.openai.com/api/docs/guides/function-calling): function definitions belong in `tools`, function outputs correlate with the returned `call_id`, and strict schemas require the documented object/required-field rules. Preserve relevant returned output items when continuing a reasoning turn. Validate arguments and authorization again in your executor. A schema-valid request can still ask for an unauthorized or stale action.

Use an explicitly configured accessible model and credentials. Missing configuration must fail clearly; never substitute a fake key or silently report fixture output as live research. This optional provider path has not been executed by the local lab.

For live web retrieval, add an explicit allowed network scope, bounded response size, redirect policy, content parsing and provenance. Do not let a model-generated URL turn a document reader into unrestricted access to local files, loopback services or cloud metadata. Keep source failures visible instead of replacing them with invented search results.

## 5. Completion and diagnostic answers

Submit a source-preserving answer, one rejected tool proposal, the test report and one failure explanation. If you add a model, include a separate trace and measured cost/latency; do not overwrite the deterministic baseline result.

- **Why not judge success by fluent prose?** Fluency does not establish source support or task success.
- **Why is the role outside tool arguments?** Identity and permission come from trusted application context.
- **Why preserve a failed case?** It tells you what to improve and prevents unsupported quality claims.
- **Why is a fixed workflow worth building first?** It isolates retrieval and execution errors from model decisions and supplies a measurable baseline.
