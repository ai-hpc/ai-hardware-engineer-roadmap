# Module 3B — Agentic AI & GenAI

**Parent:** [Phase 3 — Artificial Intelligence](../../Guide.md) · Track B

> *Build applications on top of large language models — agents, RAG, tool use, GenAI products.*

**Prerequisites:** Module 1 (Neural Networks), Module 2 (Frameworks — understand transformers and PyTorch).

**Role targets:** Agentic AI Engineer · GenAI Engineer · AI Engineer

---

## Lecture Series

40 numbered lectures, one supplemental Lecture 24b, and 5 hands-on labs covering everything from LLM fundamentals to production multi-agent systems. **[→ Start here](Lectures/README.md)**

**Course currency note:** exact model names, context windows, SDK features, and pricing change quickly. This module teaches stable interfaces and system patterns: model APIs, tool protocols, runtime loops, workflow graphs, gateways, telemetry, and policy boundaries. Always check provider documentation before copying model IDs or prices into a real deployment.

| # | Lecture | # | Lecture |
|---|---------|---|---------|
| [01](Lectures/Lecture-01.md) | LLM Fundamentals | [07](Lectures/Lecture-07.md) | Agent SDKs |
| [02](Lectures/Lecture-02.md) | Prompt Engineering | [08](Lectures/Lecture-08.md) | Multi-Agent Systems |
| [03](Lectures/Lecture-03.md) | Tool Use | [09](Lectures/Lecture-09.md) | RAG — Ingestion |
| [04](Lectures/Lecture-04.md) | Agent Architecture | [10](Lectures/Lecture-10.md) | RAG — Retrieval |
| [05](Lectures/Lecture-05.md) | Memory Systems | [11](Lectures/Lecture-11.md) | Evaluation |
| [06](Lectures/Lecture-06.md) | LangGraph | [12](Lectures/Lecture-12.md) | Production |
|  |  | [13](Lectures/Lecture-13.md) | Runtime Discipline |
|  |  | [14](Lectures/Lecture-14.md) | Deterministic Startup |
|  |  | [15](Lectures/Lecture-15.md) | OpenClaw Gateway |
|  |  | [16](Lectures/Lecture-16.md) | Routing and Sessions |
|  |  | [17](Lectures/Lecture-17.md) | Isolation and Memory |
|  |  | [18](Lectures/Lecture-18.md) | Operations and Security |
|  |  | [19](Lectures/Lecture-19.md) | Agent Loop |
|  |  | [20](Lectures/Lecture-20.md) | Cron and Scheduled Agent Runs |
|  |  | [21](Lectures/Lecture-21.md) | System Prompt Architecture |
|  |  | [22](Lectures/Lecture-22.md) | App SDK and Typed RPCs |
|  |  | [23](Lectures/Lecture-23.md) | Gateway RPC Protocol |
|  |  | [24](Lectures/Lecture-24.md) | Agent Harness |
|  |  | [24b](Lectures/Lecture-24b.md) | Event-Sourced Agent State |
|  |  | [25](Lectures/Lecture-25.md) | OpenCoven Workspace |
|  |  | [26](Lectures/Lecture-26.md) | OpenKnots Interfaces |
|  |  | [27](Lectures/Lecture-27.md) | Agent Security Engineer |
|  |  | [28](Lectures/Lecture-28.md) | Pi Minimal Agent |
|  |  | [29](Lectures/Lecture-29.md) | Agent Skills |
|  |  | [30](Lectures/Lecture-30.md) | Agentic SDLC |
|  |  | [31](Lectures/Lecture-31.md) | Runtime Strategy |
|  |  | [32](Lectures/Lecture-32.md) | LLM From Scratch |
|  |  | [33](Lectures/Lecture-33.md) | Structured Tools |
|  |  | [34](Lectures/Lecture-34.md) | Multimodal Sub-Agents |
|  |  | [35](Lectures/Lecture-35.md) | GPU Kernel Translation |
|  |  | [36](Lectures/Lecture-36.md) | FP8 KV-Cache |
|  |  | [37](Lectures/Lecture-37.md) | TraceLens |
|  |  | [38](Lectures/Lecture-38.md) | AutoSP |
|  |  | [39](Lectures/Lecture-39.md) | Agent Skills Eval |
|  |  | [40](Lectures/Lecture-40.md) | ZAYA1-8B |

**Labs:** [Lab 01 — Research Agent](Lectures/Lab-01-Research-Agent.md) · [Lab 02 — Multi-Agent Pipeline](Lectures/Lab-02-Multi-Agent-Pipeline.md) · [Lab 03 — Production RAG](Lectures/Lab-03-Production-RAG.md) · [Lab 04 — TokenJuice Output Compaction](Lectures/Lab-04-TokenJuice-Output-Compaction.md) · [Lab 05 — OpenMeow App SDK Dogfood](Lectures/Lab-05-OpenMeow-App-SDK-Dogfood.md)

---

## Why This Matters for AI Hardware

Agentic AI creates the **inference demand** that drives chip design:
- Long-context attention (128K+ tokens) → L5: HBM bandwidth, memory hierarchy
- Multi-turn tool calling → L3: low-latency kernel launch, stream scheduling
- Batch inference serving → L2: TensorRT-LLM, in-flight batching optimization
- RAG vector search → L1: cuVS/FAISS acceleration on GPU

Understanding these workloads helps L2 (compiler) and L5 (architecture) engineers design for real usage patterns.

---

## Current AI Application Trends (2026)

The strongest application trend is that AI is moving from **chat interfaces** to **persistent, tool-using systems** that act inside real workflows.

Two useful reference patterns are:
- **agentic coding systems** such as [Claude Code](https://code.claude.com/docs/en/overview)
- **local-first personal assistant systems** such as [OpenClaw](https://github.com/openclaw/openclaw)

These are worth studying because they show what modern AI applications actually look like in production-like usage, not just in demos.

### 1. Coding Agents Are Replacing Single-Step “Code Completion”

This is the clearest shift.

Tools like Claude Code are no longer limited to autocomplete in an editor. They act more like software workers:
- read and navigate a repository
- plan changes across multiple files
- run tests and verification loops
- create commits and pull requests
- connect to external tools through MCP
- load reusable skills, hooks, and plugins

Anthropic’s official docs describe this directly: Claude Code can automate routine engineering work, work with git, connect tools through MCP, spawn multiple agents, and run in CI/CD workflows. Its public repository and plugin system make it a good reference for how coding agents are becoming a real application category rather than a novelty.

**Why this matters for hardware:**
- coding agents create long-running, tool-rich inference sessions instead of short chat turns
- they increase demand for low-latency iteration loops, larger context windows, and higher background inference volume
- they push AI products into developer infrastructure, where reliability, permissioning, and automation matter as much as model quality

### 2. Personal AI Is Moving Toward Local-First Control Planes

OpenClaw is a useful reference for a different trend: AI assistants that are not “one web page with one chat box,” but a **control plane** that stays running and connects many surfaces.

From the official OpenClaw repo and docs:
- one long-lived local Gateway owns channels, sessions, tools, and events
- the assistant can operate across WhatsApp, Telegram, Slack, Discord, WebChat, and other channels
- it supports voice, mobile nodes, live canvas UI, and multi-agent routing
- it treats inbound messages as untrusted input and documents a concrete security model

This shows where application design is going:
- persistent assistants instead of one-off prompts
- multi-channel delivery instead of one frontend
- local or operator-controlled infrastructure instead of only cloud-hosted chat
- agents as routed services with isolated workspaces and memory

**Why this matters for hardware:**
- always-on assistants create steady inference demand, not just bursty usage
- multimodal assistants increase pressure on device memory, streaming, and local inference paths
- local-first designs make edge hardware, Jetson-class devices, mobile nodes, and hybrid cloud/edge deployment more relevant

### 3. MCP, Plugins, and Hooks Are Becoming the Real Application Surface

Another major trend is that the model alone is no longer the full product.

Modern AI systems are increasingly defined by:
- **MCP connectors** to external systems
- **plugins** for reusable workflows
- **hooks** and automations around model actions
- **skills** and custom agents for domain-specific behavior

Claude Code’s docs explicitly position MCP, plugins, skills, hooks, monitors, and custom agents as first-class extension surfaces. OpenClaw similarly treats tools, plugins, channels, nodes, and gateway protocols as part of the product architecture.

The implication is important: the application layer is becoming a **tool-and-protocol ecosystem**, not just a prompt template.

### 4. Agent SDKs Are Becoming Runtime Layers, Not Just API Wrappers

Modern agent SDKs now sit above raw model calls. They increasingly manage:
- agent loops
- tool dispatch
- handoffs between specialists
- sessions and state
- guardrails and human review
- tracing and evaluation
- MCP server integration

This matters because production agent systems need more than a `messages.create()` call. They need a repeatable runtime contract: what tools are available, which identity executes them, which actions require approval, where state is stored, and what gets logged.

The practical design rule for this course is:

```text
Provider API details belong in adapters.
Product behavior belongs in your runtime contract.
Security decisions belong outside the LLM.
```

This is why Lecture 07 teaches SDKs and runtime APIs as a general layer instead of treating one vendor SDK as the architecture.

### 5. Multi-Agent Structure Is Becoming Practical, Not Theoretical

The industry has moved beyond “one model, one prompt, one answer.”

Current systems increasingly use:
- a lead agent plus worker agents
- isolated workspaces per task or user
- explicit routing rules
- background monitors and event-driven triggers

Claude Code exposes multiple-agent workflows and custom agents. OpenClaw exposes multi-agent routing with isolated workspaces, session stores, and bindings from inbound channels to specific agents.

This is a practical trend because it maps well to real products:
- support workflows
- developer workflows
- personal assistant workflows
- project automation and governance workflows

### 6. Security and Permission Boundaries Are Now Core Product Features

This is one of the biggest changes from early LLM apps.

Current AI applications increasingly ship with:
- pairing and allowlists
- tool permission boundaries
- gateway auth and signed connections
- sandboxing
- safe output policies
- moderation and prompt-injection defenses

OpenClaw’s docs emphasize pairing approval, DM safety, sandbox modes, and gateway auth. Claude Code’s plugin system and tool model emphasize explicit structure, scoped extensions, and operational safety. GitHub’s agentic workflow material also frames security, permissions, and isolated execution as central rather than optional.

This means “application architecture” now includes trust boundaries, not just prompts and UI.

### 7. What To Learn From These Examples

Do not study OpenClaw and Claude Code just because they are popular. Study them because they represent two high-signal application patterns:

- **Claude Code pattern:** AI embedded into developer workflows, repositories, CI, tools, and review loops
- **OpenClaw pattern:** AI embedded into messaging, voice, mobile nodes, control planes, and personal automation

Together they show that the current application frontier is:
- agentic
- tool-connected
- persistent
- permissioned
- multi-surface
- operationally observable

For this roadmap, the important takeaway is that **Track B should teach the workloads and architectures that real AI products are converging toward**, because those products are what downstream systems and hardware will ultimately serve.

---

## 1. LLM Fundamentals for Engineers

* **Transformer architecture:** attention mechanism, KV-cache, positional encoding
* **Tokenization:** BPE, SentencePiece, vocabulary size impact on embedding layer
* **Inference mechanics:** prefill (compute-bound) vs decode (memory-bound), autoregressive generation
* **Scaling laws:** parameter count vs dataset size vs compute budget

---

## 2. Agentic AI

* **What agents are:** LLM + tools + memory + planning loop
* **Agent runtimes and frameworks:** raw provider APIs, OpenAI Agents SDK, LangGraph, MCP servers, OpenClaw-style gateways, CrewAI/AutoGen for experiments
* **Tool use:** function calling, API integration, code execution
* **Security boundaries:** prompt injection, tool abuse, least-privilege credentials, human approval for risky actions
* **Memory:** conversation history, vector store retrieval, working memory
* **Planning:** chain-of-thought, ReAct, tree-of-thought, self-reflection
* **Multi-agent systems:** task decomposition, agent collaboration, orchestration

**Projects:**
1. Build an agent that uses tools (web search, calculator, code execution) to answer complex questions.
2. Build a multi-step research agent: given a topic, search → synthesize → write a report.

---

## 3. RAG (Retrieval-Augmented Generation)

* **Architecture:** document ingestion → chunking → embedding → vector store → retrieval → generation
* **Embedding models:** sentence-transformers, OpenAI embeddings, Cohere
* **Vector stores:** FAISS, Chroma, Pinecone, Weaviate, Milvus
* **Chunking strategies:** fixed-size, recursive, semantic, document-structure-aware
* **Retrieval:** similarity search, hybrid (dense + sparse), re-ranking
* **RAG security:** treat documents as untrusted input, screen uploads and retrieved chunks, defend against indirect prompt injection
* **Evaluation:** faithfulness, relevance, hallucination detection

**Projects:**
1. Build a RAG pipeline over a technical documentation corpus. Evaluate retrieval quality.
2. Compare FAISS (CPU) vs FAISS (GPU) vs cuVS for vector search latency at 1M documents.

---

## 4. GenAI Product Development

* **Prompt engineering:** system prompts, few-shot, chain-of-thought, output formatting
* **Fine-tuning:** LoRA, QLoRA, full fine-tuning on domain data
* **Evaluation:** automated metrics (ROUGE, BLEU), LLM-as-judge, human evaluation
* **Guardrails:** input moderation, output filtering, content safety, prompt-injection defense, hallucination mitigation
* **Production deployment:** API design, streaming, rate limiting, cost management
* **Runtime discipline:** live telemetry, tool-call policy gates, least-privilege agent identities, audit trails, and runtime incident response
* **Deterministic startup:** startup contracts, readiness checks, prompt/tool/policy versioning, memory hydration, and reproducible agent boot
* **Agent control planes:** gateways, sessions, routing, and multi-surface delivery
* **Scheduled agent execution:** cron jobs, isolated run sessions, delivery fallback, failure routing, retries, run logs, and retention
* **OpenClaw-style agent development:** multi-agent isolation, workspace design, pairing, and operations for persistent local-first assistants

**Projects:**
1. Fine-tune a 7B model with QLoRA on a domain-specific dataset. Measure improvement vs base model.
2. Deploy a GenAI application with streaming, safety guardrails, and cost tracking.
3. Build a secure agent or RAG pipeline with prompt-attack detection, tool constraints, and output validation.

---

## Resources

| Resource | What it covers |
|----------|---------------|
| [LangChain Documentation](https://python.langchain.com/) | Agent and RAG framework |
| [LangGraph Documentation](https://docs.langchain.com/oss/python/langgraph/overview) | Durable, stateful agent workflows, human-in-the-loop, memory, and tracing |
| [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) | Agent loops, tools, handoffs, guardrails, sessions, tracing, and MCP integration |
| [OpenAI API Agents Guide](https://platform.openai.com/docs/guides/agents) | Current OpenAI guidance for code-first agent apps, tools, orchestration, and observability |
| [Model Context Protocol Specification](https://modelcontextprotocol.io/specification/2025-11-25) | Standard protocol for tools, resources, prompts, hosts, clients, servers, and safety considerations |
| [Claude Code Overview](https://code.claude.com/docs/en/overview) | Agentic coding workflows, MCP, multi-agent use, and CI patterns |
| [Claude Code Plugins](https://code.claude.com/docs/en/plugins) | Skills, agents, hooks, MCP servers, plugin structure, and distribution |
| [Claude Code Repository](https://github.com/anthropics/claude-code) | Public implementation surface, examples, plugins, and project layout |
| [Anthropic Cookbook](https://github.com/anthropics/anthropic-cookbook) | Practical Claude API examples |
| [OpenClaw Repository](https://github.com/openclaw/openclaw) | Local-first assistant architecture, channels, gateway model, and security defaults |
| [OpenClaw Gateway Architecture](https://docs.openclaw.ai/concepts/architecture) | Long-lived gateway, WS protocol, nodes, pairing, and remote access model |
| [OpenClaw Features](https://docs.openclaw.ai/concepts/features) | Multi-agent routing, media, channels, tools, apps, and provider support |
| [GitHub Agentic Workflows](https://github.github.com/gh-aw/slides/github-agentic-workflows.pdf) | Official GitHub framing for agentic CI/CD, permissions, and safe outputs |
| [OWASP Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/) | Prompt injection, insecure output handling, plugin/tool risk, excessive agency, and LLM app security |
| [NIST AI RMF Generative AI Profile](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-generative-artificial-intelligence) | Governance and risk-management framing for generative AI systems |
| [RAG best practices](https://docs.llamaindex.ai/) | LlamaIndex documentation |
| *Build a Large Language Model (From Scratch)* (Raschka) | LLM internals |

---

## Next

→ [**Module 4B — ML Engineering & MLOps**](../4.%20ML%20Engineering%20and%20MLOps/Guide.md)
