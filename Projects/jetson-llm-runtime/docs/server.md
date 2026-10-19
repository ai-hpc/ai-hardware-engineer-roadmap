# Serving exercise: an endpoint name is not an API contract

`src/server/http_server.cpp` is an experimental source-audit example. It is not a validated API-compatible server, and this checkout cannot generate with ordinary pretrained tokenizers. No server deployment or client-compatibility test was executed in the review.

## 1. Trace one request

A complete service has several boundaries: socket bytes, HTTP framing, JSON schema, model/chat formatting, admission, execution and response serialization. Each needs explicit accepted inputs and errors.

The current source handles one client at a time, reads once into an 8 KiB buffer, searches strings for the last `content` field and returns a response in a fixed buffer. This makes the code short enough to study, but does not implement a robust HTTP or JSON parser.

| Source behavior | Why it matters |
|---|---|
| One socket read | TCP may split headers or bodies across reads |
| String search for `content` | Escapes, nested objects and message roles are not parsed correctly |
| TODO for generation parameters | Requested temperature/max_tokens are not honored |
| One write per buffer | A write can be partial or interrupted |
| Fixed response buffer | A long completion can be truncated into invalid JSON |
| Static `status: ok` | Does not prove model readiness or successful recent inference |
| Bind to all interfaces | Local experiments may become reachable beyond the intended client |

These observations come from the checked-in source, not a claim that the interface was exercised successfully.

## 2. Understand stream framing

Imagine the bytes `Content-Length: 12` arrive in three fragments. A valid stream parser must retain partial input, find the complete header boundary, validate the length and read exactly the declared body under a deadline and size limit. A second request arriving in the same read must not be mistaken for body data.

TCP preserves byte order; it does not preserve application message boundaries. Similarly, writing a header and body requires progress tracking until all bytes are sent or an error/deadline occurs.

Before connecting inference, test a small local transport fixture with fragmented writes, early EOF, delayed body bytes, oversized input and a client disconnect. A parser must reject ambiguous framing consistently, rather than guessing which length field to trust.

## 3. Define a small honest schema

For a teaching service, a narrow versioned schema is easier to validate than claiming compatibility with a large external API. Specify required fields, types, limits, defaults and unsupported options. Use a JSON library that handles escaping and Unicode, then validate the decoded structure.

Example requirements for a future implementation:

- A request ID is distinct from model identity and cache identity.
- Prompt and output budgets are bounded before allocation.
- Unsupported streaming or tool options return an explicit error.
- Temperature and sampling fields reach the actual validated sampler.
- A cancellation stops work and releases request-owned state.
- A repeated request starts with the intended cache/reset policy.

These are acceptance criteria, not implemented features of the existing source.

## 4. Separate liveness, readiness and progress

Liveness asks whether the process can respond. Readiness asks whether the selected model/runtime can accept the defined workload. Progress asks whether an accepted request is advancing within its deadline. A single `ok` string cannot answer all three.

Sequential execution is a reasonable first design if its queue and admission policy are explicit. An unlimited queue converts overload into stale requests. For a robot, an answer produced after its observation has expired can be unusable even when inference succeeds.

Measure arrival time, processing start, first output and completion with a monotonic clock. Keep request cancellation, queue capacity and observation freshness visible to the caller.

## 5. Completion criteria

Choose a maintained HTTP/JSON stack, pin its version, and keep inference behind a small tested handler. Verify fragmented input, invalid JSON, unsupported fields, limits, disconnects, cancellation, readiness transitions and repeated requests with an actual localhost client. Only then test the declared client/API surface.

The [CPU lab](../labs/README.md) supplies validated parsing and sampling components to study now. The [testing guide](../TESTING.md) and [milestones](../ROADMAP.md) explain what remains before model serving. Naming a route `/v1/chat/completions` does not supply those missing contracts.
