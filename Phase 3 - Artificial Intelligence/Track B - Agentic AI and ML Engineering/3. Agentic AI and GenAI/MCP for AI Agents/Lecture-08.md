# Lecture 08 - Production MCP: Gateways, Registry, Eval & Capstone

**Collection:** [MCP for AI Agents](README.md) | **Previous:** [← Lecture 07](Lecture-07.md) | **Next:** [Course index](README.md)

---

Everything to this point has been about *one* server: write it (Lecture 05), serve it remotely with OAuth 2.1 (Lecture 06), and threat-model it (Lecture 07). Production is a different animal, because in production you do not run one server — you run twenty. A real agent platform reaches a filesystem server, a GitHub server, a Postgres server, three internal SaaS wrappers, a Playwright server, and whatever a product team stood up last week. The moment you have more than a handful, two problems dominate everything else: **how do you put them all behind one trustworthy front door**, and **how do you keep their combined tool surface from drowning the model**.

This lecture is the operations manual for that world. We cover **composition and proxying** — chaining servers and aggregating many behind one endpoint — and the **MCP gateway** that centralizes auth, policy, rate-limiting, and observability so each server does not reinvent them. We cover the **tool-context budget**: the unglamorous, decisive fact that dumping 200 tools into a model's context degrades its tool selection and inflates every request's token cost, and the namespacing/filtering/curation that fixes it. We cover the **official registry** as the place servers are published and discovered, the **trust signals** that decide whether you actually install one, and the **observability** you must wire around every MCP call. We close with **MCPMark** — the evaluation that tells you whether an agent-plus-server combination is production-ready rather than demo-ready — and a **capstone** that ties the whole course into one shippable artifact.

Like every lecture in this course, the facts pin to the **2025-11-25** stable spec, and the final section details what the **2026-07-28 release candidate** changes — because the stateless core, MCP Apps, and the Tasks extension reshape exactly the deployment story this lecture is about.

---

## Learning objectives

By the end of this lecture you should be able to:

- Explain composition, proxying, and the role of an **MCP gateway**, and draw N servers behind one gateway that centralizes auth, policy, rate-limiting, and observability.
- Diagnose the **tool-context budget** problem and apply the right mitigation — namespacing, per-agent tool subsets, gateway-side filtering/curation, or dynamic discovery.
- Publish to and discover from the **official MCP registry**, and read the **trust signals** that justify installing a third-party server (tying back to Lecture 07).
- Specify what to **log per MCP call** and connect it to the evaluation/observability discipline of [`../Lectures/Lecture-23.md`](../Lectures/Lecture-23.md).
- Interpret an **MCPMark** result and use it to decide whether an agent+server combination is production-ready.
- Describe what the **2026-07-28 RC** unlocks — stateless core, MCP Apps, Tasks, tighter OAuth/OIDC — and execute the course **capstone** against concrete acceptance criteria.

---

## 1. Scaling beyond one server: composition, proxying & the gateway

There are two distinct ways servers combine, and conflating them causes real design mistakes.

**Composition (chaining)** is when one server is itself the *client* of another. Your "deploy" server calls a "GitHub" server to open a PR, then a "Slack" server to announce it. The host sees one server; behind it sits a small graph of MCP connections it never learns about. Composition is how you build higher-level capabilities out of lower-level ones without teaching the host about every leaf.

**Proxying / aggregation** is when one endpoint *fronts* many servers and re-exposes their tools, resources, and prompts as if they were its own. The host opens a single connection and sees the union of everything behind the proxy. This is the pattern that scales an agent platform: instead of the host managing twenty Clients with twenty auth configs, it manages one.

An **MCP gateway** is an aggregating proxy with the production concerns bolted on. It is the single front door for a fleet of servers, and it centralizes the four things you do not want re-implemented twenty times:

- **Auth** — one place to terminate OAuth 2.1 (Lecture 06), validate the audience-bound token, and map the caller to the servers and tools they are allowed to reach. Backend servers can trust a short-lived internal credential the gateway mints rather than handling end-user OAuth each.
- **Policy** — allow/deny which tools a given agent or tenant may call, enforce argument constraints, and apply data-egress rules at one choke point.
- **Rate-limiting** — per-tenant, per-tool, per-token quotas, so one runaway agent cannot exhaust a downstream API or your bill.
- **Observability** — every call flows through the gateway, so it is the natural place to log tool name, latency, errors, and token cost (Section 4).

```text
                          ┌───────────────────────────────────────────┐
        OAuth 2.1 token   │                MCP GATEWAY                  │
   AGENT ───────────────► │  ┌─────────┬─────────┬───────────┬──────┐  │
   (one client,           │  │  AUTH   │ POLICY  │ RATE-LIMIT │ OBS  │  │
    one connection)       │  │ (L06)   │ allow/  │ per-tenant │ log  │  │
                          │  │ verify  │ deny    │ /per-tool  │ every│  │
                          │  │ audience│ tools   │ quotas     │ call │  │
                          │  └─────────┴─────────┴───────────┴──────┘  │
                          │     curated, namespaced tool surface        │
                          └───┬──────────┬──────────┬──────────┬───────┘
                              │ internal │ creds     │          │
                              ▼          ▼           ▼          ▼
                        ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
                        │ github  │ │ postgres│ │ files   │ │ slack   │
                        │ server  │ │ server  │ │ server  │ │ server  │
                        └─────────┘ └─────────┘ └─────────┘ └─────────┘
```

The gateway does *not* dissolve the 1:1 Client↔Server rule from Lecture 02 — it relocates it. The host still runs one Client to one endpoint (the gateway); the gateway runs one Client per backend server. The invariant holds at every hop; the gateway is simply a host-and-server in one process.

---

## 2. The tool-context budget

Here is the failure mode nobody warns you about until it bites: aggregate enough servers and the agent gets *worse*, not better. Every tool a server exposes contributes its name, description, and full input schema to the model's context on every turn that lists tools. Twenty servers averaging ten tools each is 200 tool definitions — easily tens of thousands of tokens — sitting in front of the model before the user has said anything.

This costs you twice. **Token cost** is the obvious tax: those definitions are billed on every request that carries them, and they crowd out the working context the agent actually needs. The **selection-accuracy** tax is worse and less visible: a model choosing among 200 near-duplicate tools (three different `search` tools, two `create_issue` variants) picks wrong, picks the shadowed one, or thrashes. More tools is not more capable past a point — it is a worse classifier with a bigger prompt.

The discipline is to treat the exposed tool surface as a **budget you spend deliberately**, not a pile you accumulate. The gateway is where you spend it, because the gateway is the curator that decides which slice of the fleet each agent actually sees.

| Strategy | What it does | When to reach for it |
| --- | --- | --- |
| **Namespacing** | Prefix tools by server (`github.create_issue`, `postgres.query`) so collisions are impossible and provenance is legible | Always — the cheapest fix; do it by default at the gateway |
| **Per-agent tool subsets** | Expose only the tools a given agent/role needs; a triage agent does not get write tools | When one fleet serves many agent roles |
| **Tool filtering / curation** | Gateway allowlists/denylists tools, drops duplicates, trims verbose descriptions | When backend servers expose more than the agent should ever call |
| **Dynamic discovery** | Start with a small core set; fetch more tools on demand (search-a-tool, or a "load server X" meta-tool) only when the task needs them | When the *catalog* is large but any one task uses few tools |

The rule of thumb: an agent should see the **smallest tool set that lets it finish its tasks**, and the gateway is responsible for enforcing that, not the model. If you find yourself reasoning about "how does the model pick among all these," you have already over-exposed — narrow the surface upstream. (This is the production-scale echo of "structured tools beat computer use": fewer, sharper, well-scoped tools beat a maximal surface, at the protocol level just as at the single-agent level.)

---

## 3. The registry: publishing, discovery & trust

By 2026 the **official MCP registry** lists **more than 2,000 community servers**, and it is the canonical place servers are *published* and *discovered* — the npm/PyPI of the MCP world. Publishing means submitting a server with its metadata: name, namespace, description, the transport it speaks, the tools/resources/prompts it exposes, and its source. Discovery means searching that catalog rather than trading GitHub links in Slack.

Discoverability is not trust. A registry entry tells you a server *exists*, not that it is safe to point an agent at — and Lecture 07 is the whole reason that distinction matters. A server's tool descriptions are injected straight into your model's context, so a malicious description is a prompt-injection vector (tool poisoning), and a compromised dependency is a supply-chain breach with your agent's credentials behind it. The registry is a discovery surface that an attacker can publish to as easily as anyone else.

So before you install a third-party server, read its **trust signals**:

| Signal | What it tells you | Red flag |
| --- | --- | --- |
| **Publisher / namespace ownership** | Whether the org claiming `github.*` actually owns it (verified namespace) | Unverified namespace impersonating a known vendor |
| **Source & build provenance** | Open source you can audit; reproducible, signed builds | Closed binary, no source, no signature |
| **Version pinning** | You can pin an exact version rather than floating `latest` | Only a moving tag; behavior can change under you |
| **Signature / integrity** | The artifact is signed and verifiable against the publisher | No signature; you cannot prove what you ran is what they shipped |
| **Maintenance & adoption** | Recent commits, responsive issues, real install base | Abandoned, or freshly published with a too-good description |

Operationally: **pin a specific version, verify its signature, and audit the tool descriptions and permissions before first run** — exactly the supply-chain hygiene Lecture 07 prescribes, applied at the registry boundary. Treat an unpinned, unsigned, unverified-namespace server the way you would treat `curl | sudo bash` from a stranger, because in agent terms that is what installing it is. For anything an internal agent depends on, prefer running your own copy behind your gateway over reaching a third-party host you do not control.

---

## 4. Observability: log every MCP call

You cannot operate, evaluate, or debug what you do not measure, and an agent driving MCP tools is a distributed system whose most interesting failures happen *between* the calls. The non-negotiable baseline is to **log every MCP call** — and the gateway (Section 1) is the natural place to do it, because every call already flows through it.

For each tool invocation, capture at minimum:

| Field | Why it matters |
| --- | --- |
| **Tool name** (namespaced) | Which capability ran; lets you see selection patterns and dead tools nothing ever calls |
| **Arguments size** (and a redacted/hashed shape — never raw secrets) | Spot oversized or malformed inputs; correlate cost without leaking data |
| **Latency** | The slow tool is usually the one wrecking the agent's wall-clock; find your tail |
| **Outcome / error** | Success vs failure, error class, and whether the agent recovered or gave up |
| **Token cost** | Tokens the call's definition + result spent; ties tool surface (Section 2) to the bill |
| **Session / trace ID** | Stitch a single agent task across its 15–20 calls into one trace |

```python
import time, logging

log = logging.getLogger("mcp.audit")

async def traced_call(gateway, session_id, tool, args):
    t0 = time.perf_counter()
    err = None
    try:
        return await gateway.call_tool(tool, args)
    except Exception as e:
        err = type(e).__name__
        raise
    finally:
        log.info("mcp_call", extra={
            "session_id": session_id,
            "tool": tool,                 # namespaced, e.g. "github.create_issue"
            "args_bytes": len(repr(args)), # size, not contents
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "error": err,                  # None on success
            # token_cost filled from the model-call accounting around this turn
        })
```

That per-call log is the raw material for the evaluation and observability discipline developed in [`../Lectures/Lecture-23.md`](../Lectures/Lecture-23.md): traces become the dataset you evaluate over, latency and token-cost distributions become your SLOs, and the error/recovery field is exactly what a benchmark like MCPMark scores. Observability is not an add-on to evaluation — it is its substrate.

---

## 5. Evaluation with MCPMark

A server that passes MCP Inspector and answers a happy-path question is a *demo*. Production asks a harder question: when an agent drives this server across a real, multi-step task — and something goes wrong mid-task, as it always does — does it recover and finish? That is what **MCPMark** measures.

MCPMark is an expert-curated benchmark of **127 tasks** spanning **Notion, GitHub, Postgres, Filesystem, and Playwright** — real systems, not toy stubs. Its defining characteristic is depth: the tasks average **16.2 turns** and **17.4 tool calls** each, and they are deliberately constructed so that an agent must **chain operations, read intermediate state, and recover from failure** to succeed. A task is not "call one tool"; it is "navigate a real workspace to a goal," and the scoring rewards getting there, including after a wrong turn.

That design is exactly why it predicts production behavior. The 16-turn average means MCPMark stresses the very things Sections 2 and 4 are about — whether the agent can still select correctly deep into a long context, whether tool latency compounds into an unusable wall-clock, and whether a failed call is a dead end or a recoverable step. A single-shot tool-calling benchmark cannot see any of that.

Use it as a **go/no-go gate**, not a vanity number. Concretely:

- Run the relevant MCPMark slice against your specific **agent + server (or gateway) combination** — the score is a property of the pair, not the server alone.
- Read the **failure-recovery** behavior, not just pass rate: tasks that fail on turn 3 and never recover indict your tools or descriptions; tasks that wander for 30 turns indict your tool-context budget.
- Treat the run as **regression coverage**: re-run it when you change the tool surface, swap a model, or upgrade a backend server, and gate the deploy on it.

MCPMark sits inside the broader benchmark ladder catalogued in [`../Lectures/Lecture-23.md` §8](../Lectures/Lecture-23.md) — the progression from single-tool correctness up to long-horizon, multi-system agentic tasks. The ladder is how you choose the *right* evaluation for the claim you are making; MCPMark is the rung that answers "is this agent-plus-server combination ready to carry real, multi-step work?"

---

## 6. The 2026 roadmap: the 2026-07-28 release candidate

The **2026-07-28 release candidate** is the largest revision since MCP launched, and it changes the production story this whole lecture describes. Four pieces matter, each unlocking something concrete.

| RC feature | What it is | What it unlocks for production |
| --- | --- | --- |
| **Stateless core** | A core protocol mode that carries no required per-session server state | Scale on ordinary HTTP / serverless — any instance handles any request, no sticky routing, no shared session store; the gateway fleet becomes trivially horizontal |
| **MCP Apps** | Server-rendered **interactive UIs**, not just text/data results | A tool can return a real UI surface (a form, a chart, a confirmation widget) the host renders — richer than a JSON blob, with the human-in-the-loop interaction inline |
| **Tasks extension** | First-class **long-running / async** work | A tool can kick off work that outlives one request — a long migration, a crawl, a build — and the agent polls/awaits completion instead of holding a connection open |
| **Tighter OAuth/OIDC** | Closer alignment with standard OAuth 2.1 / OIDC | Cleaner integration with enterprise identity providers; the Lecture 06 auth story gets more standard and less bespoke at the gateway |

Two of these resolve tensions raised earlier in the course. The **stateless core** is the answer to the scaling friction Lecture 02 flagged — a per-session-stateful server needs affinity or a shared store, and the stateless mode removes the requirement entirely, which is what makes a gateway-fronted fleet cheap to scale. The **Tasks** extension closes the gap between "a tool call returns in seconds" and "real work takes minutes" — without it, long jobs force you to fake async with polling tools and external state.

Alongside the spec RC, the official **`mcp` SDK is heading to v2** (in beta through 2026), so the FastMCP surfaces from Lecture 05 will shift. Treat all of this as **forthcoming until the RC is ratified**: build against **2025-11-25** today, and design your gateway and servers so that adopting the stateless core and Tasks later is a configuration change, not a rewrite.

---

## 7. Capstone: ship a production-grade MCP server

This is the course in one artifact. You will take a server from a blank file to something another team can trust, register, and depend on — every prior lecture shows up exactly once. Pick a real domain (a wrapper over an internal API, a curated database, a document store); the grading is on engineering, not novelty.

**Step 1 — Build the server (Lecture 05).** Implement at least **three tools, two resources, and one prompt** with the official SDK / FastMCP. Tools carry complete, accurate **JSON-Schema** inputs and structured output; resources are URI-addressed; the prompt is a reusable template. Demonstrate it green in **MCP Inspector**.

**Step 2 — Serve it remotely with OAuth 2.1 (Lecture 06).** Expose it over **Streamable HTTP** (not the deprecated SSE transport) behind spec-compliant **OAuth 2.1**: resource-server metadata (RFC 9728), **scoped, audience-bound tokens**, and the auth-server/resource-server split. A request without a valid, correctly-audienced token is rejected.

**Step 3 — Threat-model and harden it (Lecture 07).** Write a one-page threat model naming the relevant attacks — **tool poisoning, prompt injection via tool results, confused deputy, token passthrough, supply-chain** — and the specific control that stops each. Confirm you do **not** pass tokens through to downstream services and that tool descriptions are clean.

**Step 4 — Publish to the registry (Section 3).** Submit the server with correct metadata, a **verified namespace**, a **pinned version**, and a **signed** artifact. Document the trust signals a consumer should check before installing it.

**Step 5 — Put it behind a gateway (Sections 1–2).** Front it with at least one other server behind a **gateway** that centralizes auth, applies a **policy** (allow/deny tools), enforces a **rate limit**, and **logs every call** (Section 4). **Namespace** the tools and expose only a **curated subset** to the agent — demonstrate you stayed inside the tool-context budget.

**Step 6 — Evaluate it (Section 5).** Run an **MCPMark-style evaluation** of an agent driving your server (or the closest MCPMark slice plus a few domain tasks of your own). Report **pass rate, average turns/tool calls, failure-recovery behavior, latency, and token cost**, and state a **go/no-go** verdict with the reasoning.

**Acceptance criteria** — you are done when all of these hold:

| # | Criterion | Evidence |
| --- | --- | --- |
| 1 | Server exposes ≥3 tools, ≥2 resources, ≥1 prompt, all schema-correct | MCP Inspector shows them green; schemas validate |
| 2 | Reachable only over Streamable HTTP behind OAuth 2.1 | Unauthenticated / wrong-audience request → rejected; scoped token → succeeds |
| 3 | Threat model maps each named attack to a control | One-page doc; no token passthrough; tool descriptions audited |
| 4 | Published to the registry, verified namespace, pinned + signed | Registry entry; a consumer can pin the exact signed version |
| 5 | Runs behind a gateway with ≥1 other server | Centralized auth + policy + rate limit; namespaced, curated tool surface |
| 6 | Every MCP call is logged | Trace shows tool, args size, latency, error, token cost per call |
| 7 | MCPMark-style evaluation reported with a verdict | Pass rate + turns/tool-calls + recovery + cost; explicit go/no-go |

If you can do all seven, you have built the thing this course exists to teach: not a tool an agent can call once, but an MCP server **other people can safely depend on at scale**.

---

## Current as of

This lecture is current as of **June 2026**, pinned to the latest stable MCP specification, **2025-11-25** — the revision to build production servers and gateways against today. The scale figures (the official registry passing **2,000+ servers**, the `mcp` package at ~97M monthly downloads) and the **MCPMark** specification (**127 tasks** across Notion/GitHub/Postgres/Filesystem/Playwright, averaging **16.2 turns / 17.4 tool calls**, rewarding failure recovery) reflect early-2026 reporting; re-pull MCPMark's task list and scoring before you cite a number in a review. The **2026-07-28 release candidate** is the largest revision since launch — a **stateless core** (scale on ordinary HTTP/serverless without session affinity), **MCP Apps** (server-rendered interactive UIs), a **Tasks** extension (long-running/async work), and tighter **OAuth/OIDC** alignment — and the official **`mcp` SDK is heading to v2** (beta through 2026); treat all of these as forthcoming until the RC is ratified, and design so adopting them is configuration, not rewrite. Transport status: **stdio** and **Streamable HTTP** are current; **HTTP+SSE** has been deprecated since **2025-03-26**. Gateway products, registry tooling, and SDK surfaces move release-to-release — treat any code here as a snapshot and verify against your installed versions; the protocol and benchmark facts above are the stable ground.
