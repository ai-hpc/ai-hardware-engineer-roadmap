# Local retrieval and durable workflow lab

**Reviewed and executed 18 October 2026.** Build a small system that retrieves authorized evidence, quotes a source, pauses for review and survives a process failure. The core uses actual SQLite FTS5. Optional stages use actual LangGraph and FastAPI. No model, API account, GPU or external document download is needed.

This is a deterministic baseline for an AI application, **not an LLM reasoning benchmark**. It makes the surrounding machinery testable before adding a model. Follow it alongside [ingestion](../Lectures/Lecture-11.md), [retrieval](../Lectures/Lecture-12.md), [workflows](../Lectures/Lecture-16.md) and [deployment](../Lectures/Lecture-30.md).

**Prerequisites:** Python functions, dictionaries and basic SQL. **Time:** two 60-minute sessions plus a 45-minute framework extension. **Artifact:** a retrieval report, one failure explanation and a restarted workflow with exactly one local artifact row.

## 1. Know the fixture and the question

[The corpus](labs/corpus.json) contains eleven original, CC0 teaching documents about the **fictional Rowan simulator**. These are not specifications for a real robot. The documents include a superseded battery rule, a future charging rule, an operator-only note and an adversarial support attachment. Their access and validity metadata belong to the trusted fixture manifest, not to the document author's prose.

The default reader has only the `public` role and uses the date `2026-10-18`. The index contains only authorized, effective documents. A private document is excluded before indexing, so even its term frequencies do not affect this small reader's BM25 ranking. A production system needs an authenticated principal and a suitable isolation design; a role name supplied by a model is not authentication.

Each short paragraph is one chunk. The separate [question file](labs/queries.json) contains four development cases and eight fixed test cases with relevant document IDs. Tune on development cases; freeze your change before running the test split. Once you inspect test failures, treat subsequent tuning as development and create a new independent holdout. This tiny authored fixture does not estimate real-world generalization.

## 2. Run the actual retrieval baseline

From the repository root:

```bash
python3 'Phase 3 - Artificial Intelligence/Track B - Agentic AI and ML Engineering/3. Agentic AI and GenAI/Local-RAG-Workflow-Lab/labs/local_rag.py' \
  --query 'battery return threshold'
python3 'Phase 3 - Artificial Intelligence/Track B - Agentic AI and ML Engineering/3. Agentic AI and GenAI/Local-RAG-Workflow-Lab/labs/local_rag.py' \
  --split development --k 3
```

Python must include SQLite with FTS5; the tested interpreter is Python 3.12.3 with SQLite 3.45.1. The first command quotes the September rule: **25 percent**, with `battery-v2`, revision B, character span and a content hash. It must not quote the expired 10-percent rule or the private diagnostic marker.

[The implementation](labs/local_rag.py) tokenizes the query, constructs quoted FTS terms and binds the MATCH expression as a SQL parameter. It ranks actual FTS5 matches with `bm25`. SQLite's convention makes **lower scores better**; do not reuse a “higher cosine is better” threshold. A quoted source span establishes provenance, not relevance or completeness. [SQLite FTS5 ranking](https://www.sqlite.org/fts5.html#the_bm25_function).

The tool dispatcher permits only `search_manual` with exactly `query` and `k`. It rejects a proposed `write_file`, an added role argument, a boolean masquerading as an integer, and unbounded `k`. The prompt-injection attachment remains text; this baseline has no interpreter that could execute it. Passing this test does not prove an eventual LLM will resist every injection.

## 3. Measure retrieval and abstention separately

```bash
python3 'Phase 3 - Artificial Intelligence/Track B - Agentic AI and ML Engineering/3. Agentic AI and GenAI/Local-RAG-Workflow-Lab/labs/local_rag.py' \
  --split test --k 1
python3 'Phase 3 - Artificial Intelligence/Track B - Agentic AI and ML Engineering/3. Agentic AI and GenAI/Local-RAG-Workflow-Lab/labs/local_rag.py' \
  --split test --k 3
```

Actual recorded fixture results:

| Metric | k=1 | k=3 |
|---|---:|---:|
| Mean recall over 6 answerable questions |0.9167|1.0000|
| Mean reciprocal rank over those 6 |1.0000|1.0000|
| Correct abstention over 2 unanswerable questions |0.5000|0.5000|

[Top-1 report](labs/measured-test-k1-2026-10-18.json) · [Top-3 report](labs/measured-test-k3-2026-10-18.json).

For the two-document camera/audio question, top-1 finds one of two relevant sources: recall is `1/2`, while reciprocal rank is 1 because the first result is relevant. MRR therefore hides the missing second source. The baseline quotes only its first result even when retrieval finds more: perfect retrieval recall does not imply a complete answer.

The query `VIOLET-ONLY diagnostic` should abstain for a public reader. The tokenizer separates `ONLY`, and the OR search finds unrelated public paragraphs containing “only”. It returns a traceable but irrelevant quote. **This known failure is retained.** Private material remains excluded, but answerability is wrong. Adding every observed test word to a stop list would overfit the fixture. Propose a development-tested lexical coverage rule or explicit answerability stage, then evaluate new questions.

## 4. Install the optional framework extension

```bash
python3.12 -m venv /tmp/course-agent-workflow
/tmp/course-agent-workflow/bin/python -m pip install --require-hashes -r \
  'Phase 3 - Artificial Intelligence/Track B - Agentic AI and ML Engineering/3. Agentic AI and GenAI/Local-RAG-Workflow-Lab/labs/requirements-linux-cpu-py312.txt'
```

The full wheel/hash lock records LangGraph 1.2.16, SQLite checkpoint adapter 3.1.2, FastAPI 0.146.0, HTTPX 0.28.1 and text splitters 1.1.3, plus transitive dependencies. It targets the tested Linux x86_64/Python 3.12 environment. It downloads packages, not models. Framework telemetry is disabled for the workflow commands below.

## 5. Pause, restart and approve a concrete quotation

[The actual graph](labs/workflow_lab.py) is:

```text
retrieve → citation check ─┐
         → scope check ───┴→ review → local artifact save
```

The parallel checks return partial state updates. A list reducer combines their reports, and an explicit multi-source edge waits for both. Neither check establishes that the quote fully answers the question; the review payload shows the actual content.

```bash
LANGSMITH_TRACING=false /tmp/course-agent-workflow/bin/python \
  'Phase 3 - Artificial Intelligence/Track B - Agentic AI and ML Engineering/3. Agentic AI and GenAI/Local-RAG-Workflow-Lab/labs/workflow_lab.py' \
  start --directory /tmp/rowan-workflow --run-id example-1
```

The process exits with `pending_approval: true`. Read the proposed quotation. The next invocation is a **new process**:

```bash
LANGSMITH_TRACING=false /tmp/course-agent-workflow/bin/python \
  'Phase 3 - Artificial Intelligence/Track B - Agentic AI and ML Engineering/3. Agentic AI and GenAI/Local-RAG-Workflow-Lab/labs/workflow_lab.py' \
  approve --directory /tmp/rowan-workflow --run-id example-1
```

Expected status: `saved`. Use `reject` on another run to end without saving. A missing or reused run ID fails clearly. `retry` never supplies approval. Inside the graph, approval uses `interrupt()` and `Command(resume=True)`; the SQLite saver is opened with its context manager and kept alive through execution. [Interrupt semantics](https://docs.langchain.com/oss/python/langgraph/interrupts).

## 6. Inject the dangerous crash window

Start a fresh run, such as `--run-id crash-1`. Then approve with `--crash-after-save`. The command intentionally exits nonzero **after** the artifact database commits but **before** the graph checkpoints that node's success. Run `retry` in another process with the same directory/run ID.

Expected outcome: `saved`, with one artifact row for that ID. The local database uses a unique run ID and a content hash; retrying the same content reuses the row, while changing content under the same ID raises an error. This is a concrete local idempotency mechanism. It is not a claim of exactly-once execution across arbitrary external services. Such services need their own idempotency/transaction or reconciliation contract.

The review node performs no write before its interrupt because that node starts again from the top on resume. Checkpoints save graph state; they do not roll back filesystem changes, emails or robot motion.

## 7. Exercise the local API contract

[service_lab.py](labs/service_lab.py) exposes `/ask`, `/ask/stream`, `/healthz` and `/readyz` for the public synthetic corpus. Its tests use a real FastAPI application and an in-process HTTP test client. No network listener, TLS, production authentication or load test is claimed.

The POST streaming endpoint emits JSON-framed SSE `answer` and `done` events. This is source-quotation streaming, not token generation. Browser `EventSource` sends GET, so it cannot call this POST endpoint directly; use a POST streaming client and a proper SSE parser. The test proves GET receives 405 and POST receives the declared format.

Readiness is false until the index exists; shutdown closes it. Strict request validation rejects extra identity fields and unreasonable sizes. These are application contracts, not a complete production security boundary.

## 8. Run the tests and explain their limits

```bash
python3 -m unittest discover -s \
  'Phase 3 - Artificial Intelligence/Track B - Agentic AI and ML Engineering/3. Agentic AI and GenAI/Local-RAG-Workflow-Lab/labs' -p 'test_*.py' -v
LANGSMITH_TRACING=false /tmp/course-agent-workflow/bin/python -m unittest discover -s \
  'Phase 3 - Artificial Intelligence/Track B - Agentic AI and ML Engineering/3. Agentic AI and GenAI/Local-RAG-Workflow-Lab/labs/optional' -v
```

Seven portable tests and seven optional tests cover effective revisions, scoped indexing/cache identity, literal query handling, citation tampering, metric denominators, framework fan-in, actual separate-process restart, injected crash recovery, rejection, readiness, strict API schemas and SSE methods. A text-splitter test also demonstrates that `CharacterTextSplitter` can return an oversized unsplittable unit, whereas recursive splitting with a final empty-string separator enforces the character bound for that fixture.

## 9. Extend deliberately

1. **Change query wording.** Predict exact lexical failures before running. Add labels for relevance and abstention separately.
2. **Add a longer document.** Preserve document ID/revision, per-chunk ID and source spans. Do not use bare chunk text as a global identity.
3. **Replace the quote step with a model.** Keep the same retrieval, validation and evaluation boundaries. Record model revision, prompt, budget and failures; no result is assumed from this lab.
4. **Add a cache.** The supplied key includes principal/roles, effective date, corpus hash, retrieval/version policy and query. Preserve citation metadata, expiry and invalidation. Similar text is not automatically an interchangeable answer.
5. **Compare a second reviewer.** Measure whether it catches new errors, how much it costs and whether failures are correlated. Two copies of the same model do not create independent ground truth.

**Submit:** one original report, one failed case, the restart trace and a short statement of what was actually executed. The known abstention failure must remain visible in your report even though all implementation-contract tests pass.

## Reproduction record

[Executed checks and installed versions](labs/validation-2026-10-18.json) · [primary-source snapshot](labs/source-evidence-2026-10-18.json) · [fixture provenance and license](labs/DATA.md). The full lock and artifact hashes identify the CPU exercise; hosted provider and production deployment outcomes remain unexecuted.
