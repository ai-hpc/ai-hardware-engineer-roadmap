# Lecture 30 — Deploy an AI application with explicit service contracts

**Course:** [AI Agent Development](../Guide.md) | [Previous](Lecture-29.md) | [Next](Lecture-31.md)

**Reviewed 18 October 2026.** Start with the executed [local FastAPI exercise](../Local-RAG-Workflow-Lab/Guide.md#7-exercise-the-local-api-contract): FastAPI 0.146.0, HTTPX 0.28.1, a public synthetic corpus and an in-process client. Its tests establish API/lifecycle behavior, not a production deployment or an LLM quality result.

**Time:** 60–90 minutes. **Outcome:** distinguish a working endpoint from an operated service, define bounded work and preserve evidence when dependencies fail.

## 1. Decide what the endpoint promises

A request contract includes identity, input schema, allowed operations, deadline and output/error format. The server—not arbitrary client text—selects permitted models, tools, source scopes and budgets. An opaque `session_id` is a lookup key, not proof that the caller owns that session.

[The exercised service](../Local-RAG-Workflow-Lab/labs/service_lab.py) accepts a bounded query and integer `k`. It rejects extra fields, including caller-supplied roles. It serves only the public fixture and does not implement production authentication. A real service must map a verified principal to its resources and sessions before retrieval or tool execution.

## 2. Make startup and shutdown observable

Load required resources in FastAPI's lifespan context. Mark readiness only after required configuration, schemas and indexes are usable, and close clients/resources in `finally`. Do not substitute a fake API key or mark the service ready merely because an object constructor returned.

- Liveness asks whether the process can make progress; it should not restart every replica merely because one provider is temporarily down.
- Readiness asks whether this instance can serve its declared traffic.
- Startup checks allow initialization time without accepting work prematurely.

The lab tests 503 before lifespan startup,200 while ready, and resource closure on exit. It does not claim a Kubernetes rollout or downstream-provider health test. [FastAPI lifespan](https://fastapi.tiangolo.com/advanced/events/) and [Kubernetes probe semantics](https://kubernetes.io/docs/concepts/configuration/liveness-readiness-startup-probes/).

## 3. Keep blocking work out of the async event loop

An `async def` route does not make synchronous embedding, filesystem or Redis calls asynchronous. Use the appropriate async client, a bounded worker pool or a job queue for that work. Blocking the event loop can delay unrelated requests, cancellation and health checks.

The local service uses synchronous route functions for its small SQLite/quotation work; FastAPI runs ordinary route functions in a worker thread. The index serializes access to its shared connection. This is a small tested design, not a claim that SQLite or an unbounded thread pool supports any desired load. [FastAPI concurrency behavior](https://fastapi.tiangolo.com/async/).

Set an admission/concurrency limit and define what happens when full. A queue that grows without bound converts overload into rising latency and memory use. A queued robot-planning request may be obsolete before inference begins.

## 4. Match the streaming method and format

SSE is an event format/transport pattern, not a promise that a model generates one token per event. The lab POST route emits JSON-encoded `answer` and `done` events. Encoding data prevents embedded newlines from accidentally creating extra events.

Browser `EventSource` makes a GET request. It cannot directly call a POST-only endpoint with a JSON body or arbitrary authorization header. Use a suitable POST streaming client and an SSE parser, or design a GET subscription endpoint with an appropriate authentication mechanism. Do not silently mismatch the two.

For a provider stream, handle non-text events, empty deltas, completion, refusal, partial failure and missing usage explicitly. Detect client disconnects and cancel/close upstream work where supported. Do not leak raw exception strings or provider secrets in streamed errors. Test reverse-proxy buffering and timeouts on the actual deployment path. The local in-process tests establish method/framing behavior only. [SSE API](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events).

## 5. Configure model calls without invented defaults

Use a server-owned configuration with a deliberately selected accessible model and required credentials. Pin SDK versions and inspect the selected API/model's supported parameters. Do not unconditionally send a Chat Completions temperature parameter to every reasoning model or assume every returned output item is text.

For a Responses API adapter, the official [response contract](https://developers.openai.com/api/reference/resources/responses/methods/create) defines input/output items, tool calls, usage and optional controls. Treat `output_text` as a convenience view over output, not a substitute for handling tool calls or incomplete responses. Preserve the model/API errors needed for diagnosis without exposing credentials.

No provider inference was executed in the local lab. Add it as a separate experiment with actual model ID, latency, usage and a failure trace. Account access and current pricing cannot be inferred from a placeholder string.

## 6. Bound retries and distinguish failure outcomes

Use a monotonic total deadline across the operation, a maximum attempt count and jittered backoff. Respect provider retry guidance and account for retries already performed by the SDK. Nested five-attempt loops can multiply requests rather than improve reliability.

Retry transient failures only when the operation is safe to repeat. A timeout can occur after an effect succeeded. Application idempotency keys must bind identity and payload; a key reused with changed content is a conflict. The [workflow crash lab](../Local-RAG-Workflow-Lab/Guide.md#6-inject-the-dangerous-crash-window) demonstrates this boundary with a real local database.

A list of timestamps pruned to the last minute is a sliding-window limiter, not a token bucket. A true token bucket refills tokens over time up to a capacity. An in-memory limiter applies to one process and resets on restart; per-worker limits are not a global quota. Derive client identity from verified context, not an arbitrary `X-Client-Id` header.

## 7. Cache only within the same meaning and authority

Cache identity must include relevant principal/scope, source snapshot, model/prompt/policy, generation settings and query. Cache entries retain original citations and expiry. A changed source or revoked permission must not leave a reusable answer that bypasses current access checks.

Semantic similarity is not answer equivalence. Small changes in numbers, negation or dates can invalidate reuse. Begin with exact scoped caching. If adding semantic caching, evaluate wrong reuse, stale answers and scope violations as well as hit rate. A0.95cosine cutoff is not a universal correctness guarantee.

A Redis list of IDs that never expires can grow even when individual values have TTLs. Production storage needs bounded indexing/eviction and an atomic policy for related records. Do not report an unexecuted cache speedup table.

## 8. Evaluate routing and filters as fallible components

Query length or a keyword such as “why” is not a reliable measure of task difficulty. Compare routing policies on labelled tasks under a shared cost/latency budget. A second routing model has its own inference cost and failure modes.

Content moderation, PII detectors and prompt filters can help with specific risks; none authorizes a tool call, guarantees no secret leakage or proves physical safety. A detector can miss domain-specific identifiers or erase information required for the task. Evaluate its errors and apply resource/action controls in the executor.

For a robot application, keep the language-model service outside the deterministic actuator/safety loop. Validate action meaning, age and limits at the receiving system regardless of model confidence.

## 9. Verify the actual deployment boundary

Before claiming production readiness, perform the relevant experiments: authenticated cross-user/session tests, bounded overload, client disconnect/cancellation, dependency outage, malformed input, rolling restart, recovery after an ambiguous effect, and observability under the intended traffic pattern. Use a production process manager and deliberate network binding; `--reload` is a development feature, not a public deployment recipe.

Record successful, failed, cancelled and unscored requests. Report latency percentiles with sample count, workload and measurement boundary. Log request/run IDs, versioned configurations and action results while minimizing sensitive content. A process-health 200 response is not evidence that answers are correct.

**Completion:** run the local API tests, explain the POST/GET mismatch, write one cache-invalidation case and one retry-idempotency case, and identify which tests need a real deployment environment. Preserve unexecuted items as unexecuted.
