# Lab 03 — RAG evaluation and a small service contract

**Track B · Agentic AI & GenAI** | [Previous](Lab-02-Multi-Agent-Pipeline.md) | [Index](README.md)

**Time:** 90–120 minutes. Extend the executed [local retrieval/workflow lab](../Local-RAG-Workflow-Lab/Guide.md). This is a tested teaching application with a clear production extension path; it is not advertised as a production deployment merely because FastAPI starts.

## 1. Establish ingestion and access contracts

Use the original fictional corpus, its CC0 provenance and its explicit access/effective-date fields. Compute the corpus hash. Keep each document's ID/revision and each chunk's source span. Index only the reader's authorized current evidence in this small baseline.

Superseded facts and private notes are deliberately present in the corpus. Test that they cannot influence this reader's results. Filtering after a model has seen a document is too late. In a larger shared vector store, permissions must be enforced in every retrieval path, including sparse search, reranking, cache hits, exports and logs.

## 2. Measure what retrieval actually returns

Run the fixed development and test splits. At k=1, the measured test recall is 0.9167; at k=3 it is 1.0. Both have MRR 1.0 and abstention accuracy 0.5. These are eight synthetic questions, not a benchmark of a language model or deployed service.

The two-document question reveals why recall and MRR differ. The unanswerable diagnostic question reveals why a source can be genuine but irrelevant. Record retrieved IDs and source order, not only a concatenated context blob.

Compute each metric by hand for at least one case. Unanswerable questions have no relevant-document denominator, so they are evaluated separately rather than assigned an arbitrary recall score.

## 3. Add generation as a separately measured stage

The executed baseline quotes its first result. An optional LLM stage can synthesize several chunks, but it must preserve citations and handle insufficient evidence. Record:

- Retrieval recall/ranking against relevant source IDs.
- Answer correctness and completeness against reviewed references.
- Support for individual factual claims in retrieved evidence.
- Abstention behavior, citation validity and unauthorized-source exposure.
- Retrieval/model latency, actual usage and failed/unscored cases.

Use human-reviewed questions and references; generating a question and treating its automatically generated answer as ground truth creates correlated errors. A model judge is a fallible measurement instrument. Calibrate it against reviewed examples and preserve judge failures as unscored rows. Do not replace failures with zero or invent an “expected” results table. See [evaluation and observability](Lecture-23.md).

For RAGAS, preserve ordered per-chunk contexts and match the installed metric's required fields. Reference-based metrics need a reviewed reference; faithfulness alone cannot certify factual truth. The [official metric definitions](https://docs.ragas.io/en/stable/concepts/metrics/) and your recorded package/model revisions define the experiment.

## 4. Exercise the actual API

Run the optional test suite in the linked lab. It executes a real FastAPI application with an in-process HTTP test client. The service uses a public synthetic corpus and exposes POST `/ask`, POST `/ask/stream`, GET `/healthz` and GET `/readyz`.

Inspect [service_lab.py](../Local-RAG-Workflow-Lab/labs/service_lab.py). Startup creates the index before readiness becomes true. Shutdown closes it. Strict input validation rejects extra fields, empty/oversized queries, invalid `k` and caller-supplied roles. The POST SSE route JSON-encodes its event data and has an explicit completion event. A GET-only `EventSource` client will receive405; method agreement is part of the contract.

The tests do not exercise a public listener, TLS, client disconnects, reverse-proxy buffering, multiple workers or sustained load. Those remain explicit deployment experiments in [Lecture 30](Lecture-30.md), not inferred successes.

## 5. Make cache identity preserve meaning

The supplied cache-key function includes trusted subject/roles, effective date, corpus hash, retrieval version, answer policy, `k` and the original query. A cached result must retain its citations and be invalidated when relevant state changes. This lab provides the identity function but does not claim a measured Redis cache or cache-hit speedup.

Semantic similarity is not equivalence. “Stop at0.5m” and “stop at 5 m”, or “enable” and “do not enable”, may embed nearby but require different answers. Begin with exact scoped reuse. If adding semantic reuse, evaluate false reuse and stale/unauthorized answers, not just hit rate.

## 6. Complete one meaningful extension

Choose one:

1. Add chunking to a longer source while preserving ID/revision/spans; show a retrieval failure before and after.
2. Add a reviewed answerability rule using development cases, then evaluate fresh unanswerable questions.
3. Add an actual local/API model and compare answer completeness with the quotation baseline under a fixed budget.
4. Add a bounded cache and test corpus revision, access changes, expiry and citation preservation.

**Submit:** the corpus/version manifest, labelled cases, baseline and changed results, one failure analysis and a list of unexecuted deployment stages. Do not hide the baseline's known 50-percent abstention result behind its passing implementation tests.
