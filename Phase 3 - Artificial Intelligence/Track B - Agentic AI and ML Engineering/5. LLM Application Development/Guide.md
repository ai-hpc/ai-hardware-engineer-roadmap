# Module 5B — LLM Application Development

**Parent:** [Phase 3 — Artificial Intelligence](../../Guide.md) · Track B

> *Ship GenAI products — from prompt engineering to production deployment.*

**Prerequisites:** Module 3B (Agentic AI), Module 4B (ML Engineering).

**Role targets:** AI Engineer · GenAI Engineer · LLM Application Developer · Full-Stack AI Engineer

---

## Why This Matters for AI Hardware

LLM applications are the **largest consumer of GPU inference capacity** in 2025–2026:
- ChatGPT serves 200M+ weekly users → massive GPU fleet
- Enterprise RAG deployments → GPU-accelerated vector search + LLM inference
- Code assistants → long-context attention, streaming generation
- Understanding these patterns helps hardware engineers design chips that serve real demand

---

## 1. Prompt Engineering (Advanced)

* **System prompts:** persona, constraints, output format specification
* **Few-shot learning:** example selection, dynamic few-shot, chain-of-thought
* **Structured output:** JSON mode, function calling, tool use schemas
* **Prompt optimization:** iterative refinement, A/B testing prompts, automated evaluation
* **Long-context strategies:** context window management, chunking, summarization chains

---

## 2. Fine-Tuning LLMs

* **When to fine-tune vs prompt engineering vs RAG**
* **LoRA / QLoRA:** parameter-efficient fine-tuning, adapter merging
* **Full fine-tuning:** when you need maximum quality and have enough data/compute
* **Data preparation:** instruction formatting, chat templates, quality filtering
* **Evaluation:** perplexity, task-specific metrics, human eval, LLM-as-judge

**Projects:**
1. Fine-tune Llama-3-8B with QLoRA on a domain-specific Q&A dataset. Evaluate vs base model.
2. Merge LoRA adapters and export to ONNX for deployment.

---

## 3. Production RAG Architecture

* **Advanced retrieval:** hybrid search (dense + BM25), re-ranking (cross-encoder), query expansion
* **Chunking optimization:** recursive splitting, semantic chunking, parent-child retrieval
* **Multi-modal RAG:** images + text, document layout understanding
* **Evaluation framework:** RAGAS, context precision/recall, faithfulness scoring
* **Scaling:** distributed vector stores, caching, embedding batch processing

**Projects:**
1. Build a production RAG system with hybrid retrieval + re-ranking. Evaluate with RAGAS.
2. Add citation tracking — every generated claim linked to source chunks.

---

## 4. Production Deployment

* **API design:** streaming responses, structured output, error handling
* **Scaling patterns:** load balancing, auto-scaling, GPU right-sizing
* **Cost optimization:** caching (semantic cache, exact cache), model routing (small → large), prompt compression
* **Observability:** token usage tracking, latency monitoring, quality scoring
* **Safety:** content filtering, PII detection, output validation, rate limiting

**Projects:**
1. Deploy a RAG application with streaming, caching, and cost tracking. Measure tokens/$ efficiency.
2. Implement semantic caching — cache similar queries to reduce GPU inference calls by 30%+.

---

## 5. Safety, Moderation, and Prompt Security

If you ship an LLM product, this is part of the product, not an optional add-on.

You need controls for:
- harmful or illegal requests
- jailbreaks and prompt-injection attempts
- unsafe retrieval context and hostile documents
- PII leakage in prompts or responses
- application-specific denied topics such as regulated advice, political persuasion, or internal-only subjects when policy requires it

### 5.1 Defense-in-Depth Architecture

```text
User input
  -> normalize and sanitize text
  -> blocklist / denied-topic checks
  -> moderation + jailbreak / prompt-injection detection
  -> retrieval or tool access with scoped permissions
  -> model with hardened system prompt
  -> output moderation + schema validation + PII redaction
  -> logs, review queue, metrics, threshold tuning
```

System prompts help, but they do not replace input filtering, tool constraints, and output validation.

### 5.2 Pre-Model Controls

* **Input normalization:** strip or normalize hidden Unicode, homoglyph tricks, and malformed encodings before policy checks.
* **Keyword and pattern filters:** use fast first-pass checks for obvious harmful terms, denied topics, secrets, or policy-sensitive phrases.
* **Classifier-based moderation:** run a text classifier or managed moderation API to score toxicity, hate, self-harm, sexual, violence, or illicit content.
* **Prompt-injection detection:** classify jailbreak attempts such as role-play overrides, "ignore previous instructions", or hostile document payloads.
* **Document screening:** for RAG, treat uploaded files, webpages, and retrieved chunks as untrusted input. Scan them before they enter the prompt.

### 5.3 In-Model Controls

* **System prompt hardening:** define role, scope, refusal behavior, and tool-use limits clearly.
* **Constrained tool use:** whitelist tools, validate arguments, and scope credentials so the model cannot escalate through tools.
* **Structured outputs:** force JSON or schema-constrained outputs for actions that trigger downstream systems.
* **Generation limits:** cap output length, tool calls, and recursion depth to reduce abuse and runaway cost.

### 5.4 Post-Model Controls

* **Output moderation:** rescan completions before returning them to the user.
* **PII detection and redaction:** scrub emails, phone numbers, account IDs, secrets, or regulated identifiers from outputs and logs.
* **Grounding and validation:** for RAG, require citations or evidence checks before presenting factual claims as trusted.
* **Safe fallback behavior:** replace blocked outputs with a refusal or escalation path instead of exposing raw unsafe text.

### 5.5 Operations and Evaluation

* **Log blocked and borderline prompts:** you need examples for policy tuning and incident review.
* **Measure false positives and false negatives:** strict policies can break legitimate workflows.
* **Red-team continuously:** adversaries adapt. Test jailbreaks, encoded prompts, multilingual attacks, and indirect prompt injection.
* **Version policies:** thresholds, blocklists, regex rules, and system prompts should be versioned like code.

### 5.6 Minimal Python Sketch

```python
import re

BLOCKED_TOPICS = ["bomb", "kill", "credit card dump"]
JAILBREAK_PATTERNS = [
    r"ignore previous instructions",
    r"you are dan",
    r"pretend to be",
]

def sanitize_unicode(text: str) -> str:
    return "".join(c for c in text if not (0xE0000 <= ord(c) <= 0xE007F))

def keyword_block(text: str) -> bool:
    lowered = text.lower()
    return not any(term in lowered for term in BLOCKED_TOPICS)

def detect_jailbreak(text: str) -> bool:
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in JAILBREAK_PATTERNS)

def secure_prompt_pipeline(user_input: str) -> str:
    clean = sanitize_unicode(user_input)

    if not keyword_block(clean):
        return "Blocked: denied topic detected."

    if detect_jailbreak(clean):
        return "Blocked: prompt attack detected."

    # Then call your moderation service, model, output moderation,
    # and PII redaction steps.
    return "Safe to continue."
```

The point of the example is the pipeline shape, not the exact classifier choice. In production, replace the placeholders with real moderation services, prompt-attack detectors, and PII tooling.

### 5.7 Tools You Should Know

| Layer | Tool / Service | What it is useful for |
|------|-----------------|-----------------------|
| Input / output moderation | [OpenAI Moderation](https://platform.openai.com/docs/guides/moderation) | Managed text and image moderation for harmful content classification |
| Prompt-attack detection | [Azure AI Content Safety Prompt Shields](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/quickstart-jailbreak) | Detects user prompt attacks and document attacks before generation |
| AI firewall / policy gateway | [Google Cloud Model Armor](https://docs.cloud.google.com/model-armor/overview) | Screens prompts and responses for prompt injection, harmful content, sensitive data, and malicious URLs |
| Managed guardrails | [Amazon Bedrock Guardrails](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html) | Content filters, denied topics, word filters, sensitive information filters, and prompt-attack detection |
| PII redaction | [Microsoft Presidio](https://microsoft.github.io/presidio/) | Detect and anonymize sensitive data in text and images |
| Programmable guardrails | [NVIDIA NeMo Guardrails](https://docs.nvidia.com/nemo-guardrails/index.html) | Open-source guardrail flows for input, output, retrieval, and security checks |
| Open-source prompt / safety classifiers | [Meta Llama Guard and Prompt Guard](https://huggingface.co/meta-llama) | Self-hosted safety and prompt-attack classifiers when you need local or customizable controls |

### 5.8 Projects

1. Build a secure chat or RAG gateway: Unicode normalization -> moderation -> jailbreak detection -> model call -> output moderation -> PII redaction.
2. Evaluate a jailbreak defense stack on a prompt corpus. Measure block rate, false positives, and added latency.
3. Compare managed vs self-hosted guardrails on one workload: OpenAI or Azure or Bedrock vs NeMo Guardrails or Llama Guard.
4. Add document screening for RAG uploads and retrieved chunks, then test indirect prompt injection cases.

---

## Connection to Hardware

| Application pattern | Hardware implication |
|--------------------|---------------------|
| Long-context attention (128K tokens) | HBM bandwidth, KV-cache memory |
| Streaming token generation | Low-latency kernel scheduling |
| Batch inference serving | In-flight batching, GPU utilization |
| Vector search (RAG retrieval) | cuVS / FAISS on GPU |
| Multi-model routing | Multi-GPU scheduling, MIG partitioning |
| Moderation sidecars and guard models | Extra latency, memory footprint, and deployment topology choices |
| On-device prompt security on Jetson or edge NPUs | Small classifier selection, quantization, and CPU/GPU partitioning |

---

## Resources

| Resource | What it covers |
|----------|---------------|
| [Maxime Labonne's LLM Course](https://github.com/mlabonne/llm-course) | Free LLM roadmap and Colab notebooks covering LLM fundamentals, the LLM Scientist path, and the LLM Engineer path: fine-tuning, quantization, RAG, evaluation, and deployment |
| [Anthropic API Documentation](https://docs.anthropic.com/) | Claude API, tool use, streaming |
| [OpenAI Cookbook](https://cookbook.openai.com/) | GPT API patterns and best practices |
| [LlamaIndex](https://docs.llamaindex.ai/) | RAG framework |
| [RAGAS](https://docs.ragas.io/) | RAG evaluation framework |
| [Azure AI Content Safety](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/overview) | Managed content moderation and configurable safety controls |
| [Google Cloud Model Armor](https://docs.cloud.google.com/model-armor/overview) | Prompt / response screening, sensitive data protection, malicious URL detection |
| [Amazon Bedrock Guardrails](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html) | Managed content, topic, and PII safeguards |
| [Microsoft Presidio](https://microsoft.github.io/presidio/) | PII detection and anonymization |
| [NVIDIA NeMo Guardrails](https://docs.nvidia.com/nemo-guardrails/index.html) | Programmable LLM guardrails |
| *Building LLM Applications* (various) | End-to-end LLM app development |
