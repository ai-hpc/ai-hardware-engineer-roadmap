# Agentic AI Development — Lecture Series

A hands-on lecture series building from LLM fundamentals to production multi-agent systems.

## Lecture Index

| # | Title | Topics |
|---|-------|--------|
| [Lecture 01](Lecture-01.md) | LLM Fundamentals for Agents | Transformers, tokenization, inference mechanics, context windows |
| [Lecture 02](Lecture-02.md) | Prompt Engineering & Structured Output | System prompts, few-shot, JSON mode, function calling |
| [Lecture 03](Lecture-03.md) | Tool Use & Function Calling | Tool schemas, parallel calls, error handling, safety |
| [Lecture 04](Lecture-04.md) | Agent Architecture Patterns | ReAct, CoT, Reflexion, plan-and-execute |
| [Lecture 05](Lecture-05.md) | Memory Systems | Short-term, long-term, episodic, semantic memory |
| [Lecture 06](Lecture-06.md) | LangGraph — Stateful Workflows | Nodes, edges, state, checkpointing, human-in-the-loop |
| [Lecture 07](Lecture-07.md) | Claude Agent SDK | Subagents, tool loops, streaming, computer use |
| [Lecture 08](Lecture-08.md) | Multi-Agent Systems | CrewAI, AutoGen, supervisor patterns, coordination |
| [Lecture 09](Lecture-09.md) | RAG — Ingestion & Embeddings | Chunking, embedding models, vector stores, indexing |
| [Lecture 10](Lecture-10.md) | RAG — Retrieval & Reranking | Hybrid search, MMR, cross-encoder reranking, evaluation |
| [Lecture 11](Lecture-11.md) | Evaluation & Observability | LLM-as-judge, RAGAS, tracing, cost tracking |
| [Lecture 12](Lecture-12.md) | Production Deployment | Streaming, caching, model routing, safety, scaling |
| [Lecture 13](Lecture-13.md) | Runtime Discipline & AI Runtime Security | Runtime controls, tool policy, telemetry, auditability, agent risk |

## Lab Index

| # | Title | Build |
|---|-------|-------|
| [Lab 01](Lab-01-Research-Agent.md) | Research Agent with Tool Use | Web search + code execution + citations |
| [Lab 02](Lab-02-Multi-Agent-Pipeline.md) | Multi-Agent Code Review | Planner → Coder → Reviewer → Summarizer |
| [Lab 03](Lab-03-Production-RAG.md) | Production RAG System | Ingestion pipeline + hybrid search + RAGAS eval |

## Prerequisites

- Python 3.10+
- PyTorch basics (Phase 3 Core — Neural Networks)
- API keys: Anthropic, OpenAI (optional)

```bash
pip install anthropic langchain langgraph langchain-anthropic \
            chromadb sentence-transformers ragas openai
```
