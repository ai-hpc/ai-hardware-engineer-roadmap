# Agentic AI Development - Lecture Series

A hands-on lecture series building from LLM fundamentals to production multi-agent systems.

## How to Read This Course in 2026

Model names, context windows, SDK features, and token prices change quickly. Treat vendor-specific examples as implementation snapshots, not permanent recommendations.

The durable concepts in this course are:

- **Model API:** the direct text, structured output, tool-call, and streaming interface.
- **Agent runtime:** the loop that manages turns, tools, sessions, handoffs, guardrails, and traces.
- **Tool protocol:** MCP-style tools, resources, and prompts exposed by external systems.
- **Workflow control:** graphs, checkpoints, retries, human review, and deterministic startup.
- **Product control plane:** gateways, channels, sessions, routing, identities, and audit logs.
- **Runtime security:** least privilege, policy gates, telemetry, incident response, and evidence.

Lectures 01-08 build the agent mechanics. Lectures 09-12 add data, evaluation, and deployment. Lectures 13-21 focus on production discipline and OpenClaw-style gateway systems.

## Lecture Index

| # | Title | Topics |
|---|-------|--------|
| [Lecture 01](Lecture-01.md) | LLM Fundamentals for Agents | Transformers, tokenization, inference mechanics, context windows |
| [Lecture 02](Lecture-02.md) | Prompt Engineering & Structured Output | System prompts, few-shot, JSON mode, function calling |
| [Lecture 03](Lecture-03.md) | Tool Use & Function Calling | Tool schemas, parallel calls, error handling, safety |
| [Lecture 04](Lecture-04.md) | Agent Architecture Patterns | ReAct, CoT, Reflexion, plan-and-execute |
| [Lecture 05](Lecture-05.md) | Memory Systems | Short-term, long-term, episodic, semantic memory |
| [Lecture 06](Lecture-06.md) | LangGraph — Stateful Workflows | Nodes, edges, state, checkpointing, human-in-the-loop |
| [Lecture 07](Lecture-07.md) | Agent SDKs and Runtime APIs | SDKs, provider adapters, MCP, handoffs, streaming, runtime policy |
| [Lecture 08](Lecture-08.md) | Multi-Agent Systems | CrewAI, AutoGen, supervisor patterns, coordination |
| [Lecture 09](Lecture-09.md) | RAG — Ingestion & Embeddings | Chunking, embedding models, vector stores, indexing |
| [Lecture 10](Lecture-10.md) | RAG — Retrieval & Reranking | Hybrid search, MMR, cross-encoder reranking, evaluation |
| [Lecture 11](Lecture-11.md) | Evaluation & Observability | LLM-as-judge, RAGAS, tracing, cost tracking |
| [Lecture 12](Lecture-12.md) | Production Deployment | Streaming, caching, model routing, safety, scaling |
| [Lecture 13](Lecture-13.md) | Runtime Discipline & AI Runtime Security | Runtime controls, tool policy, telemetry, auditability, agent risk |
| [Lecture 14](Lecture-14.md) | Deterministic Startup for AI Agent Systems | Startup contracts, readiness gates, tool registries, prompt versions, memory hydration |
| [Lecture 15](Lecture-15.md) | OpenClaw Case Study - Gateway Architecture | Control plane, channels, clients, nodes, agent loop |
| [Lecture 16](Lecture-16.md) | OpenClaw Case Study - Routing and Sessions | Channel routing, session keys, DM isolation, reply determinism |
| [Lecture 17](Lecture-17.md) | OpenClaw Case Study - Multi-Agent Isolation | Workspaces, state, sessions, memory boundaries |
| [Lecture 18](Lecture-18.md) | OpenClaw Case Study - Operations and Security | Pairing, supervision, sandbox, tool policy, remote access |
| [Lecture 19](Lecture-19.md) | OpenClaw Case Study - The Agent Loop | Intake, queues, locks, streaming, tools, hooks, persistence |
| [Lecture 20](Lecture-20.md) | OpenClaw Case Study - Cron and Scheduled Agent Runs | Cron expressions, isolated jobs, delivery, retries, logs, validation |
| [Lecture 21](Lecture-21.md) | OpenClaw Case Study - System Prompt Architecture | Prompt ownership, bootstrap context, skills, prompt modes, provider overlays |

## Lab Index

| # | Title | Build |
|---|-------|-------|
| [Lab 01](Lab-01-Research-Agent.md) | Research Agent with Tool Use | Web search + code execution + citations |
| [Lab 02](Lab-02-Multi-Agent-Pipeline.md) | Multi-Agent Code Review | Planner → Coder → Reviewer → Summarizer |
| [Lab 03](Lab-03-Production-RAG.md) | Production RAG System | Ingestion pipeline + hybrid search + RAGAS eval |

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
