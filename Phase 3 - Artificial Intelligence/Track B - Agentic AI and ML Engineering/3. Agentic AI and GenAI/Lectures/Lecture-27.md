# Lecture 27 - Deterministic Startup for AI Agent Systems

**Course:** [AI Agent Development 2026](../Guide.md) | **Previous:** [Lecture 26](Lecture-26.md) | **Next:** [Lecture 28](Lecture-28.md)

---

## Why this lecture exists

An AI agent system is not just a model call.

It is usually a stack of moving parts:

- configuration files
- environment variables
- model clients
- prompts
- tools
- memory stores
- vector indexes
- workflow graphs
- schedulers
- background workers
- auth providers
- observability
- runtime policies

If those parts start in a different order every time, the system becomes hard to debug.

If one dependency is half-ready, the agent may run with missing tools, stale memory, wrong prompts, broken retrieval, or unsafe permissions.

That is why production agent systems need **deterministic startup**.

Deterministic startup means:

> Given the same code, config, secrets, data snapshot, and environment, the agent system boots into the same known-good state every time.

This does not mean model outputs become deterministic. LLMs may still produce different text.

It means the **system around the model** starts **predictably**.

---

## Learning objectives

By the end of this lecture you will be able to:

1. Explain deterministic startup in simple terms.
2. Identify why agent systems fail when startup order is unclear.
3. Design a startup sequence with explicit phases.
4. Validate config, prompts, tools, model clients, memory, indexes, policies, and workers before serving traffic.
5. Build readiness checks that prevent half-started agents from accepting requests.
6. Separate startup, warmup, recovery, and normal serving.
7. Write a startup manifest for a real AI agent application.
8. Understand why deterministic startup matters for edge AI devices and always-on assistants.

---

## 1. The simple mental model

Think of an AI agent system like a smart factory.

Before the factory opens, you do not want workers randomly turning on machines in any order.

You want a checklist:

1. Power is stable.
2. Safety guards are installed.
3. Machines are calibrated.
4. Materials are loaded.
5. Operators are assigned.
6. Emergency stop works.
7. Quality checks pass.
8. Production starts.

Agent systems need the same discipline.

A bad startup looks like this:

```text
server starts
  -> accepts request
  -> model client is ready
  -> tool registry is still loading
  -> vector index is stale
  -> memory migration is incomplete
  -> policy engine has old rules
  -> agent acts incorrectly
```

A deterministic startup looks like this:

```text
load config
  -> validate schema
  -> connect dependencies
  -> register tools
  -> load prompts
  -> load policies
  -> hydrate memory
  -> verify indexes
  -> warm model paths
  -> run startup self-test
  -> mark service ready
  -> accept traffic
```

The system should not serve users until the full **startup contract** passes.

---

## 2. What deterministic startup is not

Deterministic startup does **not** mean:

- the model always returns the same answer
- temperature must always be `0`
- every request follows the same path
- the agent cannot adapt at runtime
- the system never fails

It means:

- startup order is explicit
- startup checks are repeatable
- configuration is validated
- tools are registered predictably
- prompts and policies have known versions
- dependencies are either ready or the service refuses traffic
- failures happen early instead of silently during user requests

In professional systems, startup should be boring.

If startup is surprising, production will be worse.

---

## 3. Why agent systems especially need this

Normal web services need deterministic startup too.

Agent systems need it more because the model can hide infrastructure problems behind fluent language.

Example:

```text
User:
What did customer ACME order last month?

Agent problem:
CRM tool did not register at startup.

Bad agent behavior:
"ACME likely ordered standard parts based on previous demand."
```

The answer sounds plausible, but it is wrong.

Another example:

```text
User:
Summarize the safety procedure.

Agent problem:
Vector index failed to load the latest safety manual.

Bad agent behavior:
Answers from an old version of the manual.
```

With normal software, a missing dependency often causes an obvious error.

With AI systems, missing dependencies can cause **confident wrong behavior**.

That is the core risk.

---

## 4. The startup contract

A **startup contract** is a written promise about what must be true before the system accepts traffic.

Example:

```text
This agent service is ready only when:

- required environment variables are present
- config schema validates
- model provider is reachable
- prompt bundle version is known
- tool registry contains exactly the expected tools
- tool permissions are loaded
- vector index version matches the document snapshot
- memory store schema is migrated
- policy engine has loaded the active policy bundle
- tracing and audit logging are writable
- health and readiness checks pass
```

This contract should be enforced by code.

Do not rely on a README checklist alone.

---

## 5. Startup phases

Use **phases** instead of random initialization.

### Phase 0 - process starts

The process exists, but nothing should serve traffic yet.

### Phase 1 - load static configuration

Load:

- config files
- environment variables
- deployment profile
- feature flags
- model names
- prompt bundle version
- tool allowlist
- policy bundle version

Rule:

> No network calls yet. Just load and validate local inputs.

### Phase 2 - validate configuration

Check:

- required fields exist
- paths are valid
- model names are allowed
- tool names are known
- numeric limits are sane
- dangerous feature flags are not enabled by accident

Fail fast if config is invalid.

### Phase 3 - connect dependencies

Connect to:

- model provider
- vector database
- relational database
- cache
- queue
- memory store
- object storage
- auth provider
- observability backend

Rule:

> Connect, verify, and record versions. Do not silently continue with missing dependencies unless the app explicitly supports degraded mode.

### Phase 4 - register tools

Build the tool registry.

For each tool, register:

- name
- description
- input schema
- risk level
- timeout
- permission rule
- owner
- audit policy
- idempotency behavior

Do not let tools appear dynamically without control.

### Phase 5 - load prompts and policies

Load:

- system prompts
- agent role prompts
- RAG prompt templates
- tool-use instructions
- runtime security policies
- refusal policies
- human approval rules

Each should have a version.

### Phase 6 - hydrate memory and state

Load:

- session state
- long-term memory
- user preferences
- workflow checkpoints
- agent graph checkpoints
- task queues

Run migrations before serving.

### Phase 7 - verify retrieval indexes

Check:

- index exists
- document snapshot version matches expected version
- embedding model version matches the stored vectors
- top-k retrieval smoke test works
- access control filters are installed

### Phase 8 - warm critical paths

Warm:

- model client
- tokenizer or local model runtime
- embedding model
- vector search path
- common prompt template rendering
- common tool schema validation

This reduces first-request surprises.

### Phase 9 - startup self-test

Run a short self-test:

- safe prompt call
- safe retrieval query
- read-only tool call
- policy block test
- audit log write
- readiness report

### Phase 10 - mark ready

Only now should `/readyz` return success.

Before this point, `/livez` may be true, but `/readyz` should be false.

---

## 6. Liveness vs readiness

This distinction matters.

| Check | Meaning | Should traffic be sent? |
|---|---|---|
| Liveness | The process is alive | Not necessarily |
| Readiness | The service is ready to handle requests | Yes |

An agent service can be alive but not ready.

Example:

```text
/livez  -> 200 OK
/readyz -> 503 Not Ready
```

This is correct while startup is still running.

Bad design:

```text
/health -> 200 OK
```

even though tools, memory, or policies are missing.

Professional rule:

> Liveness asks "should this process be restarted?" Readiness asks "should this process receive user traffic?"

---

## 7. Startup manifest

A startup manifest is a simple document that describes exactly what the agent expects at boot.

Example:

```yaml
service: hardware_support_agent
version: 0.4.2

startup:
  required_env:
    - MODEL_PROVIDER
    - MODEL_API_KEY
    - VECTOR_DB_URL
    - AUDIT_LOG_URL

  models:
    chat:
      name: gpt-example-prod
      required: true
    embedding:
      name: text-embedding-example
      required: true

  prompts:
    bundle: hardware_support_prompts
    version: 2026-04-23

  retrieval:
    index: hardware_docs_index
    document_snapshot: docs_2026_04_20
    embedding_model: text-embedding-example

  tools:
    - name: search_docs
      risk: low
      required: true
    - name: create_ticket
      risk: medium
      required: true
    - name: send_email
      risk: high
      required: false

  policies:
    bundle: agent_runtime_policy
    version: 8

  readiness:
    require_audit_log: true
    require_policy_engine: true
    require_retrieval_smoke_test: true
```

The manifest makes startup reviewable.

If something changes, the diff is visible.

---

## 8. Config validation example

Use **structured config** instead of loose environment access scattered through the codebase.

```python
from pydantic import BaseModel, Field, HttpUrl


class ModelConfig(BaseModel):
    chat_model: str
    embedding_model: str
    temperature: float = Field(ge=0.0, le=2.0)
    max_tokens: int = Field(gt=0, le=8192)


class RetrievalConfig(BaseModel):
    vector_db_url: HttpUrl
    index_name: str
    document_snapshot: str
    top_k: int = Field(gt=0, le=50)


class RuntimePolicyConfig(BaseModel):
    policy_bundle: str
    policy_version: int = Field(gt=0)
    require_human_approval_for_high_risk_tools: bool = True


class AgentConfig(BaseModel):
    service_name: str
    environment: str
    models: ModelConfig
    retrieval: RetrievalConfig
    policy: RuntimePolicyConfig


def load_config(raw: dict) -> AgentConfig:
    config = AgentConfig.model_validate(raw)

    if config.environment == "prod" and config.models.temperature > 0.7:
        raise ValueError("production temperature is too high for this agent")

    return config
```

Important point:

> Configuration errors should fail during startup, not during the first customer request.

---

## 9. Tool registry determinism

The **tool registry** must be predictable.

Bad pattern:

```python
tools = discover_all_tools_from_folder("tools/")
```

Why this is risky:

- file ordering may vary
- accidental tools may load
- experimental tools may appear in production
- permissions may not match the tool set
- review is difficult

Better pattern:

```python
EXPECTED_TOOLS = [
    "search_docs",
    "create_ticket",
    "lookup_part_number",
    "summarize_datasheet",
]


def build_tool_registry(tool_factories: dict) -> dict:
    registry = {}

    for name in EXPECTED_TOOLS:
        if name not in tool_factories:
            raise RuntimeError(f"missing required tool: {name}")

        tool = tool_factories[name]()
        validate_tool_schema(tool)
        validate_tool_policy(tool)
        registry[name] = tool

    extra_tools = set(tool_factories) - set(EXPECTED_TOOLS)
    if extra_tools:
        raise RuntimeError(f"unexpected tools available: {sorted(extra_tools)}")

    return registry
```

Professional rule:

> Production tools should be explicitly registered, versioned, and policy-checked.

---

## 10. Prompt determinism

Prompts are code-like assets.

They should have:

- names
- versions
- owners
- tests
- changelogs
- rollback path

Bad pattern:

```python
SYSTEM_PROMPT = "You are helpful."
```

Better pattern:

```yaml
prompt_bundle: hardware_support_agent
version: 2026-04-23

prompts:
  system:
    file: prompts/system.md
    sha256: "..."
  tool_router:
    file: prompts/tool_router.md
    sha256: "..."
  rag_answer:
    file: prompts/rag_answer.md
    sha256: "..."
```

At startup, verify:

- files exist
- hashes match
- required variables are present
- rendering works with test data
- prompt versions are logged

This makes incidents easier to investigate.

If a model produces bad output, you need to know which prompt version was active.

---

## 11. Retrieval determinism

**RAG startup** must verify that retrieval is not silently broken.

Check:

- vector database reachable
- index exists
- expected document count range
- embedding dimension matches
- embedding model version matches
- access-control filters exist
- sample query returns expected documents

Example startup smoke test:

```python
def verify_retrieval(index, expected_snapshot: str):
    metadata = index.get_metadata()

    if metadata["snapshot"] != expected_snapshot:
        raise RuntimeError(
            f"index snapshot mismatch: expected {expected_snapshot}, "
            f"got {metadata['snapshot']}"
        )

    results = index.search("ESP32-C6 UART pin configuration", top_k=3)
    ids = {item.document_id for item in results}

    if "esp32c6_uart_guide" not in ids:
        raise RuntimeError("retrieval smoke test failed")
```

Do not rely on "the vector DB connection works."

Connection success only proves that the database answered.

It does not prove that the right index is loaded.

---

## 12. Memory determinism

Agent memory is powerful but dangerous.

At startup, decide:

- which memory stores are loaded
- which sessions are resumed
- which memories are expired
- which memories are quarantined
- which schema migrations must run
- which checkpoint version is supported

Bad pattern:

```text
load all previous memory into context automatically
```

Better pattern:

```text
load only memory that:
  - belongs to this user
  - belongs to this tenant
  - matches current schema
  - is not expired
  - is not quarantined
  - is relevant to the current task
```

Memory startup checks:

- schema version matches
- migrations completed
- memory count is within expected range
- quarantine table is readable
- checkpoint replay works for one test session

Professional rule:

> Memory is not just data. It is future prompt context. Treat it like executable influence.

---

## 13. Policy determinism

**Runtime security policies** must load before tools are usable.

Bad startup:

```text
agent starts
  -> tools register
  -> service accepts traffic
  -> policy engine loads later
```

This creates a window where tools may run without enforcement.

Correct startup:

```text
load policy bundle
  -> validate policy syntax
  -> register tools
  -> bind tool to policy
  -> run allow/block self-test
  -> mark tool layer ready
```

Self-test example:

```python
def verify_policy_engine(policy_engine):
    allowed = policy_engine.evaluate(
        user_role="engineer",
        tool="search_docs",
        arguments={"query": "Jetson audio setup"},
    )
    assert allowed.decision == "allow"

    blocked = policy_engine.evaluate(
        user_role="guest",
        tool="export_customer_records",
        arguments={"format": "csv"},
    )
    assert blocked.decision == "block"
```

If the block test fails, the service should not start.

---

## 14. Model client determinism

**Model clients** should not be created lazily without checks.

At startup, verify:

- provider credentials exist
- selected model is allowed
- timeout is configured
- retry policy is configured
- circuit breaker is configured
- fallback model is known
- model response path works for a tiny test request

Do not run an expensive prompt at startup.

Use a cheap sanity check:

```python
def verify_model_client(client):
    response = client.generate(
        messages=[
            {"role": "system", "content": "Return exactly OK."},
            {"role": "user", "content": "health check"},
        ],
        max_tokens=4,
        temperature=0,
        timeout=5,
    )

    if "OK" not in response.text:
        raise RuntimeError("model client health check failed")
```

This does not prove model quality.

It proves the model path is reachable and correctly configured.

---

## 15. Deterministic graph startup

Workflow agents often use a graph:

```text
planner -> retriever -> tool_router -> executor -> reviewer -> responder
```

At startup, verify:

- all nodes exist
- all edges are valid
- no unreachable required node exists
- cycles are intentional
- checkpointing is enabled where needed
- human approval nodes exist for high-risk paths
- graph version is logged

Example graph manifest:

```yaml
graph: support_agent_graph
version: 12

nodes:
  - planner
  - retriever
  - tool_router
  - executor
  - reviewer
  - responder

edges:
  planner:
    - retriever
    - tool_router
  retriever:
    - responder
  tool_router:
    - executor
    - reviewer
  executor:
    - reviewer
  reviewer:
    - responder
```

Startup should reject a graph that references a missing node.

---

## 16. Idempotent startup

Startup should be **safe to run more than once**.

This matters because containers, systemd services, edge devices, and cloud platforms may restart processes.

Idempotent startup means repeated startup does not duplicate state or corrupt data.

Bad examples:

- create duplicate background jobs every restart
- re-send "startup complete" notifications every restart
- recreate indexes without checking version
- run destructive migrations automatically
- append duplicate system memories

Better examples:

- create queue only if missing
- run migrations with version tracking
- register worker lease with expiration
- load prompt bundle by immutable version
- write startup event with unique boot ID

Use a boot ID:

```python
import uuid

BOOT_ID = str(uuid.uuid4())
```

Attach it to logs:

```json
{
  "boot_id": "2d8b...",
  "event": "startup_phase_complete",
  "phase": "tool_registry",
  "status": "ok"
}
```

Now you can group all startup logs from one process boot.

---

## 17. Degraded mode

Sometimes a service can start with **limited capability**.

Example:

- chat works, but RAG is unavailable
- read-only tools work, but write tools are disabled
- local model works, but cloud fallback is unavailable

This is acceptable only if the degraded mode is explicit.

Bad degraded mode:

```text
retrieval broken, but agent answers from memory without telling anyone
```

Good degraded mode:

```text
retrieval unavailable
  -> readiness reports degraded
  -> RAG features disabled
  -> agent tells user it cannot access documents
  -> alert is emitted
```

Represent this in readiness:

```json
{
  "ready": true,
  "mode": "degraded",
  "disabled_features": ["rag_search"],
  "reason": "vector index unavailable"
}
```

For high-risk systems, degraded mode may not be allowed.

---

## 18. Startup timeline example

A clean startup log should tell a story.

Example:

```text
00.000 boot_id=42 service=assistant start
00.018 phase=config_load ok config_version=prod-17
00.026 phase=config_validate ok
00.143 phase=dependency_connect ok vector_db=ready cache=ready audit=ready
00.181 phase=prompt_load ok prompt_bundle=assistant_prompts@2026-04-23
00.214 phase=policy_load ok policy_bundle=runtime_policy@8
00.266 phase=tool_registry ok tools=7 high_risk=2
00.402 phase=memory_migration ok schema=5
00.611 phase=retrieval_verify ok snapshot=docs_2026_04_20
00.902 phase=model_warmup ok model=prod-small latency_ms=288
01.104 phase=self_test ok
01.105 readiness=true
```

Bad startup log:

```text
server started
```

That tells you almost nothing.

---

## 19. Deterministic startup on edge devices

This roadmap cares about Jetson-class and embedded AI systems.

**Edge startup** is harder because:

- power may be unstable
- network may be unavailable
- local models may take time to load
- sensors may appear late
- audio devices may enumerate differently
- accelerators may need warmup
- storage may be slow
- device clocks may be wrong at boot

For an AI smart speaker or local assistant, deterministic startup might include:

```text
system boot
  -> audio device detected
  -> wake-word engine loaded
  -> local ASR model loaded
  -> TTS voice loaded
  -> tool registry loaded
  -> home devices paired
  -> memory store mounted
  -> network state detected
  -> cloud fallback optional
  -> readiness announced
```

If the microphone array is not ready, the assistant should not pretend it is listening.

If the smart-home controller is unavailable, device-control commands should be disabled.

Edge agent rule:

> Local AI products must know which capabilities are actually available after boot.

---

## 20. Startup test plan

Test startup like you test features.

### Test 1 - clean boot

Expected:

- all startup phases pass
- readiness becomes true
- startup time is within budget

### Test 2 - missing config

Remove a required environment variable.

Expected:

- startup fails early
- clear error message
- no traffic accepted

### Test 3 - missing tool

Remove a required tool implementation.

Expected:

- tool registry phase fails
- service stays not ready

### Test 4 - broken vector index

Point to the wrong document snapshot.

Expected:

- retrieval verification fails
- RAG disabled or startup fails depending on policy

### Test 5 - policy engine failure

Load an invalid policy file.

Expected:

- policy validation fails
- high-risk tools never become available

### Test 6 - restart idempotency

Start, stop, start again.

Expected:

- no duplicate jobs
- no duplicate memories
- no duplicate indexes
- same manifest version

### Test 7 - cold edge boot

Reboot the device from power-off.

Expected:

- sensors and audio devices are detected
- local models load
- service reports real capability status

---

## 21. Practical startup checklist

Before you call an agent system production-ready, answer these questions.

### Configuration

- Are all required config fields validated?
- Are unsafe defaults rejected in production?
- Are model, prompt, policy, and graph versions logged?

### Dependencies

- Does startup verify every required dependency?
- Is degraded mode explicit?
- Are timeouts and retries configured?

### Tools

- Is the tool registry explicit?
- Are unexpected tools rejected?
- Are high-risk tools bound to policies?
- Are tool schemas validated?

### Retrieval and memory

- Is the vector index version checked?
- Is the embedding model version checked?
- Are memory migrations complete before serving?
- Are quarantined memories excluded?

### Runtime safety

- Does the policy engine load before tools are usable?
- Does a block-policy self-test run?
- Are audit logs writable before readiness?

### Operations

- Are `/livez` and `/readyz` separate?
- Is startup time measured?
- Is every startup phase logged?
- Is there a boot ID?
- Can the system restart safely?

---

## 22. Common mistakes

### Mistake 1 - serving traffic before readiness

The process starts, the port opens, and traffic begins before tools, memory, or policy are ready.

Fix:

> Keep `/readyz` false until the full startup contract passes.

### Mistake 2 - lazy-loading critical tools

The first user request discovers that a tool is broken.

Fix:

> Load and validate required tools at startup.

### Mistake 3 - relying on whatever files are present

The system discovers prompts, tools, or configs dynamically and accidentally loads experimental assets.

Fix:

> Use explicit manifests and allowlists.

### Mistake 4 - no version record

The system cannot tell which prompt, policy, graph, model, or index version caused an incident.

Fix:

> Log versions at startup and attach them to request traces.

### Mistake 5 - treating edge boot as normal server boot

Audio, sensors, local models, and hardware devices may not be ready when the process starts.

Fix:

> Add hardware capability checks and feature-level readiness.

---

## 23. Design exercise

Design deterministic startup for this system:

> A local AI engineering assistant runs on a Jetson. It supports voice input, local RAG over hardware docs, code editing inside a project folder, and smart-lab device control through Zigbee.

Create a startup manifest with:

- required environment variables
- required local devices
- model paths
- prompt bundle version
- vector index snapshot
- tool registry
- policy bundle
- readiness checks
- degraded mode rules

Then answer:

1. Which features can run offline?
2. Which features require network?
3. Which features require human approval?
4. Which startup failures should block the whole service?
5. Which startup failures should only disable one feature?

---

## Key takeaways

- Deterministic startup means the agent system boots into a known-good state before accepting traffic.
- It does not make LLM outputs deterministic. It makes the surrounding system predictable.
- Startup should be phase-based, logged, validated, and testable.
- Required tools, prompts, policies, memory, retrieval indexes, and model clients should be verified before readiness.
- `/livez` and `/readyz` must be separate.
- Tool registries should be explicit and policy-bound.
- RAG startup must verify the right index, snapshot, and embedding model.
- Memory startup must handle schema, expiration, quarantine, and checkpoint compatibility.
- Edge AI systems need capability-aware startup because hardware and network state may vary at boot.
- A production agent should fail early rather than serve traffic in a half-ready state.

---

## References

- [Kubernetes probes: liveness, readiness, and startup probes](https://kubernetes.io/docs/concepts/configuration/liveness-readiness-startup-probes/)
- [Twelve-Factor App: Config](https://12factor.net/config)
- [OpenTelemetry](https://opentelemetry.io/)
- [OWASP Top 10 for Large Language Model Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/)

---

*Next: [Lecture 28](Lecture-28.md)*
