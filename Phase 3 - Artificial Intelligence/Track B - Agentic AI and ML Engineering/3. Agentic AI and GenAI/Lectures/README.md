# AI Agent Development 2026 — Lecture Index

A hands-on lecture series building from the modern agent harness, through fundamentals and the core parts of an agent, to a real harness case study (OpenClaw) and a capstone build (**genie-claw**).

> This is the **flat, numbered index** of all lectures. For the recommended **theory-first reading order** grouped into modules, see the **[course Guide](../Guide.md)**.

## How to Read This Course in 2026

Model names, context windows, SDK features, and token prices change quickly. Treat vendor-specific examples as implementation snapshots, not permanent recommendations.

The durable concepts in this course are:

- **Model API:** the direct text, structured output, tool-call, and streaming interface.
- **Agent runtime:** the loop that manages turns, tools, sessions, handoffs, guardrails, and traces.
- **Tool protocol:** MCP-style tools, resources, and prompts exposed by external systems.
- **Workflow control:** graphs, checkpoints, retries, human review, and deterministic startup.
- **Product control plane:** gateways, channels, sessions, routing, identities, and audit logs.
- **Runtime security:** least privilege, policy gates, telemetry, incident response, and evidence.

The lectures are numbered 01–41 in the recommended reading order. They group into six modules — *start here* (the modern agent, the harness, and how to build one), *fundamentals*, *core building blocks*, *production & runtime*, the *OpenClaw* example, and the *genie-claw* practice capstone. See the **[course Guide](../Guide.md)** for the full module breakdown.

## Lecture Index

<div class="lecture-map" markdown>

| # | Title | Topics |
|---|-------|--------|
| [Lecture 01](Lecture-01.md) | The Modern AI Agent in 2026: What Changed | Agent definition, 2023→2026 shifts, harness AI, reference systems, course map |
| [Lecture 02](Lecture-02.md) | What Is an AI Agent Harness? The Runtime Around the Model | Harness vs model, six core responsibilities, Claude Code / Cursor / Codex compared, hardware impact |
| [Lecture 03](Lecture-03.md) | Building Agents I: Foundations (Model, Tools, Instructions) | What an agent is, when to build one, and the three components — model, tools, instructions |
| [Lecture 04](Lecture-04.md) | Building Agents II: Orchestration & Guardrails | Single vs multi-agent, manager & decentralized patterns, guardrail types, human-in-the-loop |
| [Lecture 05](Lecture-05.md) | LLM Fundamentals for Agents | Transformers, tokenization, inference mechanics, context windows |
| [Lecture 06](Lecture-06.md) | LLM From Scratch - Model Mechanics for Agent and GPU Engineers | Tokenizers, transformer blocks, training loop, inference, prefill/decode, GPU kernel intuition |
| [Lecture 07](Lecture-07.md) | Prompt Engineering & Structured Output | System prompts, few-shot, JSON mode, function calling |
| [Lecture 08](Lecture-08.md) | Tool Use & Function Calling | Tool schemas, parallel calls, error handling, safety |
| [Lecture 09](Lecture-09.md) | Structured Tools Beat Computer Use - Interface Hierarchy for Agents | Reflex benchmark, structured API vs vision, tool schemas, verification, security, OpenClaw tool design |
| [Lecture 10](Lecture-10.md) | Memory Systems | Short-term, long-term, episodic, semantic memory |
| [Lecture 11](Lecture-11.md) | RAG — Ingestion & Embeddings | Chunking, embedding models, vector stores, indexing |
| [Lecture 12](Lecture-12.md) | RAG — Retrieval & Reranking | Hybrid search, MMR, cross-encoder reranking, evaluation |
| [Lecture 13](Lecture-13.md) | Qdrant, pgvector, and Embedding Model Selection | vector stores, HNSW, IVFFlat, dense/sparse/hybrid retrieval, Granite alternatives, embedding evals, migration |
| [Lecture 14](Lecture-14.md) | Efficient Local RAG Stack - Qwen3.5-4B INT4 and Granite Embeddings | Jetson RAG, Granite 97M, Qdrant, chunking, reranking, INT4, llama.cpp, vLLM, TensorRT-LLM, KV cache |
| [Lecture 15](Lecture-15.md) | Agent Architecture Patterns | ReAct, CoT, Reflexion, plan-and-execute |
| [Lecture 16](Lecture-16.md) | LangGraph — Stateful Workflows | Nodes, edges, state, checkpointing, human-in-the-loop |
| [Lecture 17](Lecture-17.md) | Agent SDKs and Runtime APIs | SDKs, provider adapters, MCP, handoffs, streaming, runtime policy |
| [Lecture 18](Lecture-18.md) | OpenAI Agents SDK - Native Sandbox and Durable Agent Harness | sandbox agents, manifests, shell/apply_patch, MCP, skills, AGENTS.md, state recovery, harness/compute separation |
| [Lecture 19](Lecture-19.md) | Multi-Agent Systems | CrewAI, AutoGen, supervisor patterns, coordination |
| [Lecture 20](Lecture-20.md) | Nemotron 3 Nano Omni - Multimodal Perception Sub-Agents | Unified video/audio/image/text reasoning, hybrid MoE, EVS, throughput, OpenClaw sub-agent architecture |
| [Lecture 21](Lecture-21.md) | Agent Skills - Workflow Discipline for Reliable Coding Agents | Skill workflows, anti-rationalization, verification evidence, progressive disclosure, scope discipline |
| [Lecture 22](Lecture-22.md) | Agent Skills Eval - Benchmarking SKILL.md Files | with-skill vs baseline evals, LLM judge assertions, artifacts, CI gates, OpenClaw skill regression testing |
| [Lecture 23](Lecture-23.md) | Evaluation & Observability | LLM-as-judge, RAGAS, tracing, cost tracking |
| [Lecture 24](Lecture-24.md) | Runtime Discipline & AI Runtime Security | Runtime controls, tool policy, telemetry, auditability, agent risk |
| [Lecture 25](Lecture-25.md) | AI Agent Security Engineer - A Practitioner's Roadmap | 8-phase curriculum, prompt-injection trust boundaries, sandboxing tiers, red-team practice, audit log discipline, hardware-rooted trust |
| [Lecture 26](Lecture-26.md) | Session as Source of Truth: Event-Sourced Agent State | Session vs context window, event schema, `wake(sessionId)`, streaming-crash recovery, tool idempotency |
| [Lecture 27](Lecture-27.md) | Deterministic Startup for AI Agent Systems | Startup contracts, readiness gates, tool registries, prompt versions, memory hydration |
| [Lecture 28](Lecture-28.md) | Runtime Strategy for Agent Systems - Node, Bun, Rust, and Edge Packaging | Bun Zig-to-Rust signal, Node baseline, Rust offload, runtime measurements, edge packaging |
| [Lecture 29](Lecture-29.md) | Agentic SDLC - Explore Fast, Ship Safely | Cheap code, implementation as exploration, tests as contracts, evolving specs, dual-mode agents |
| [Lecture 30](Lecture-30.md) | Production Deployment | Streaming, caching, model routing, safety, scaling |
| [Lecture 31](Lecture-31.md) | OpenClaw Case Study - Gateway Architecture | Control plane, channels, clients, nodes, agent loop |
| [Lecture 32](Lecture-32.md) | OpenClaw Case Study - Routing and Sessions | Channel routing, session keys, DM isolation, reply determinism |
| [Lecture 33](Lecture-33.md) | OpenClaw Case Study - Multi-Agent Isolation | Workspaces, state, sessions, memory boundaries |
| [Lecture 34](Lecture-34.md) | OpenClaw Case Study - Operations and Security | Pairing, supervision, sandbox, tool policy, remote access |
| [Lecture 35](Lecture-35.md) | OpenClaw Case Study - The Agent Loop | Intake, queues, locks, streaming, tools, hooks, persistence |
| [Lecture 36](Lecture-36.md) | OpenClaw Case Study - Cron and Scheduled Agent Runs | Cron expressions, isolated jobs, delivery, retries, logs, validation |
| [Lecture 37](Lecture-37.md) | OpenClaw Case Study - System Prompt Architecture | Prompt ownership, bootstrap context, skills, prompt modes, provider overlays |
| [Lecture 38](Lecture-38.md) | OpenClaw Case Study - App SDK Dogfooding and Typed Gateway RPCs | App SDK, happy path, event normalization, future RPC surfaces |
| [Lecture 39](Lecture-39.md) | OpenClaw Case Study - Gateway RPC Protocol | WebSocket frames, handshake, roles, scopes, pairing, features, node transport |
| [Lecture 40](Lecture-40.md) | OpenClaw Threat Model - MITRE ATLAS for Agent Security | threat matrix, attack chains, trust boundaries, prompt injection, skill supply chain, tool execution controls |
| [Lecture 41](Lecture-41.md) | Pi - A Minimal Coding Agent and the Substrate Beneath OpenClaw | Tiny core (4 tools), no-MCP rationale, custom messages in session log, hot reload, tree-structured sessions, TUI vs LLM-tool surfaces |

</div>

## Lab Index

<div class="lecture-map" markdown>

| # | Title | Build |
|---|-------|-------|
| [Lab 01](Lab-01-Research-Agent.md) | Research Agent with Tool Use | Web search + code execution + citations |
| [Lab 02](Lab-02-Multi-Agent-Pipeline.md) | Multi-Agent Code Review | Planner → Coder → Reviewer → Summarizer |
| [Lab 03](Lab-03-Production-RAG.md) | Production RAG System | Ingestion pipeline + hybrid search + RAGAS eval |
| [Lab 04](Lab-04-TokenJuice-Output-Compaction.md) | TokenJuice Output Compaction | Deterministic terminal-output reduction, raw bypasses, artifact recovery, project reducers |
| [Lab 05](Lab-05-OpenMeow-App-SDK-Dogfood.md) | OpenMeow App SDK Dogfood on macOS | Test the OpenClaw App SDK with OpenCoven's OpenMeow adapter, fixtures, UI reducers, live Gateway smoke tests, and optional Coven sessions |
| [Lab 06](Lab-06-Genie-Claw-Capstone.md) | **Capstone: Build genie-claw** | Your own minimal agent harness — run loop, tools, durable sessions, guardrails, human-in-the-loop, wired to a local LLM runtime |

</div>

## Prerequisites

- Python 3.10+
- PyTorch basics (Phase 3 Core — Neural Networks)
- API keys for whichever provider examples you run

```bash
pip install anthropic openai pydantic fastapi uvicorn \
            langchain langgraph langchain-anthropic langchain-openai \
            chromadb sentence-transformers ragas opentelemetry-api
```

Install only the packages needed for the lecture you are running. For production work, pin versions in `requirements.txt` or `pyproject.toml` and review provider migration notes before upgrading SDKs.

Code snippets use placeholder model IDs such as `your-agent-model-id`, `your-fast-model-id`, and `your-embedding-model-id`. Replace them with current model IDs from your provider before running the examples.

---

## External References

<div class="lecture-map" markdown>

| Resource | What it covers |
|----------|----------------|
| [The OpenClaw Book](https://openclawconsultant.com/openclaw-book/) | Practitioner OpenClaw guide: architecture, setup, skills, prompting, planning, optimization, sub-agents, security |
| [LangChain Documentation](https://python.langchain.com/docs/) | Agent and RAG framework |
| [LangGraph Documentation](https://langchain-ai.github.io/langgraph/) | Durable, stateful agent workflows, human-in-the-loop, memory, and tracing |
| [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) | Agent loops, tools, handoffs, guardrails, sessions, tracing, and MCP integration |
| [OpenAI API Agents Guide](https://platform.openai.com/docs/guides/agents) | Code-first agent apps, tools, orchestration, and observability |
| [Model Context Protocol Specification](https://modelcontextprotocol.io/) | Standard protocol for tools, resources, prompts, hosts, clients, servers, and safety |
| [Claude Code Overview](https://docs.claude.com/en/docs/claude-code/overview) | Agentic coding workflows, MCP, multi-agent use, and CI patterns |
| [Claude Code Plugins](https://docs.claude.com/en/docs/claude-code/plugins) | Skills, agents, hooks, MCP servers, plugin structure, and distribution |
| [Claude Code Repository](https://github.com/anthropics/claude-code) | Public implementation surface, examples, plugins, and project layout |
| [Anthropic Cookbook](https://github.com/anthropics/anthropic-cookbook) | Practical Claude API examples |
| OpenClaw Repository | Local-first assistant architecture, channels, gateway model, and security defaults |
| OpenClaw Gateway Architecture | Long-lived gateway, WS protocol, nodes, pairing, and remote access model |
| OpenClaw Features | Multi-agent routing, media, channels, tools, apps, and provider support |
| GitHub Agentic Workflows | Official GitHub framing for agentic CI/CD, permissions, and safe outputs |
| [OWASP Top 10 for LLM Applications](https://genai.owasp.org/llm-top-10/) | Prompt injection, insecure output handling, tool risk, excessive agency, LLM app security |
| [NIST AI RMF Generative AI Profile](https://www.nist.gov/itl/ai-risk-management-framework) | Governance and risk-management framing for generative AI systems |
| [LlamaIndex Documentation](https://docs.llamaindex.ai/) | RAG best practices |
| [Build a Large Language Model (From Scratch) — Raschka](https://github.com/rasbt/LLMs-from-scratch) | LLM internals |

</div>
