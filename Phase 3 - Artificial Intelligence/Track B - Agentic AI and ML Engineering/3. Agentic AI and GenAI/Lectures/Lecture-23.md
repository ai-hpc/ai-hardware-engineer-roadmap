# Lecture 23 — Evaluation & Observability

**Course:** [AI Agent Development 2026](../Guide.md) | **Previous:** [Lecture 22](Lecture-22.md) | **Next:** [Lecture 24](Lecture-24.md)

---

## Learning Objectives

By the end of this lecture you will be able to:

1. Score LLM outputs for correctness and quality using an LLM-as-judge pattern.
2. Compute RAGAS metrics (faithfulness, answer relevancy, context precision, context recall) on a RAG system.
3. Trace LLM calls with LangSmith and understand what to log.
4. Build a cost-tracking decorator that accumulates token spend per session.
5. Log every LLM call with full input/output/token/latency details.
6. Construct an evaluation dataset from production traffic and run A/B prompt tests.

---

## 1. LLM-as-Judge

The **"LLM-as-judge"** pattern uses a capable LLM to evaluate the output of another LLM call. It is **cheaper and faster than human annotation** while correlating well with human judgment for well-designed rubrics.

```python
# pip install openai

import os
import json
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "sk-fake"))

JUDGE_PROMPT = """You are an expert evaluator. Score the answer on the following rubric.
Return a JSON object with keys: score (0-5), reasoning (one sentence).

Rubric:
  5 = Completely correct, well-cited, nothing to add.
  4 = Correct but missing minor details.
  3 = Partially correct with one factual error.
  2 = Mostly wrong or misleading.
  1 = Completely wrong.
  0 = Refused to answer or empty.

Question: {question}
Reference Answer: {reference}
Model Answer: {answer}
"""

def judge_answer(question: str, reference: str, answer: str) -> dict:
    prompt = JUDGE_PROMPT.format(
        question=question, reference=reference, answer=answer
    )
    response = client.chat.completions.create(
        model=os.environ.get("OPENAI_JUDGE_MODEL", "your-judge-model-id"),
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(response.choices[0].message.content)


# Example evaluation
test_cases = [
    {
        "question": "What is the memory bandwidth of the H100 SXM5?",
        "reference": "3.35 TB/s using HBM3 memory.",
        "answer": "The H100 SXM5 provides approximately 3.35 terabytes per second of memory bandwidth via HBM3.",
    },
    {
        "question": "What is the memory bandwidth of the H100 SXM5?",
        "reference": "3.35 TB/s using HBM3 memory.",
        "answer": "The H100 SXM5 has 2 TB/s memory bandwidth.",  # wrong
    },
]

for tc in test_cases:
    result = judge_answer(tc["question"], tc["reference"], tc["answer"])
    print(f"Score: {result['score']}/5 — {result['reasoning']}")
    print(f"Answer was: '{tc['answer'][:60]}...'")
    print()
```

### 1.1 Batch Evaluation

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def batch_evaluate(test_cases: list[dict], max_workers: int = 4) -> list[dict]:
    """Evaluate multiple test cases in parallel."""
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(judge_answer, tc["question"], tc["reference"], tc["answer"]): tc
            for tc in test_cases
        }
        for future in as_completed(futures):
            tc = futures[future]
            result = future.result()
            results.append({**tc, **result})

    results.sort(key=lambda x: x["score"], reverse=True)
    avg = sum(r["score"] for r in results) / len(results)
    print(f"\nMean score: {avg:.2f}/5.0 over {len(results)} cases")
    return results
```

---

## 2. RAGAS Metrics

**RAGAS** (Retrieval Augmented Generation Assessment) defines **four complementary metrics** for RAG systems. All are computed without needing ground-truth answers for the first two.

```python
# pip install ragas langchain-openai datasets

import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Prepare evaluation data
# Each row: question, answer (from RAG), contexts (list of retrieved chunks), ground_truth
eval_data = {
    "question": [
        "What is the H100 memory bandwidth?",
        "What does HBM3 stand for?",
        "How does NVLink 4.0 compare to NVLink 3.0?",
    ],
    "answer": [
        "The H100 SXM5 delivers 3.35 TB/s of memory bandwidth using HBM3.",
        "HBM3 stands for High Bandwidth Memory generation 3.",
        "NVLink 4.0 provides 900 GB/s bidirectional bandwidth, double NVLink 3.0's 600 GB/s.",
    ],
    "contexts": [
        ["H100 SXM5 uses HBM3 providing 3.35 TB/s bandwidth.", "The H100 PCIe offers 2 TB/s."],
        ["HBM3 is the third generation of High Bandwidth Memory DRAM standard."],
        ["NVLink 4.0 delivers 900 GB/s total bidirectional bandwidth.", "NVLink 3.0 provided 600 GB/s."],
    ],
    "ground_truth": [
        "3.35 TB/s",
        "High Bandwidth Memory generation 3",
        "NVLink 4.0 doubles NVLink 3.0 bandwidth to 900 GB/s",
    ],
}

dataset = Dataset.from_dict(eval_data)

# Configure LLM and embeddings for RAGAS internal evaluation
llm = ChatOpenAI(model=os.environ.get("OPENAI_EVAL_MODEL", "your-eval-model-id"), temperature=0)
embeddings = OpenAIEmbeddings(model=os.environ.get("OPENAI_EMBEDDING_MODEL", "your-embedding-model-id"))

results = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=llm,
    embeddings=embeddings,
)

print("\nRAGAS Evaluation Results:")
print(results.to_pandas().to_string(index=False))
```

### 2.1 Understanding Each Metric

| Metric | Measures | Range | Formula |
|---|---|---|---|
| **Faithfulness** | Is the answer factually grounded in the retrieved context? | 0–1 | claims in context / total claims |
| **Answer Relevancy** | Does the answer address the question? | 0–1 | cosine sim of regenerated questions |
| **Context Precision** | Are retrieved chunks relevant (precision)? | 0–1 | relevant chunks / total retrieved |
| **Context Recall** | Are all ground-truth facts covered by context? | 0–1 | facts in context / total facts |

```python
# Manual faithfulness calculation (illustrative)
import re

def compute_faithfulness_manual(answer: str, context: str, llm) -> float:
    """Decompose answer into claims and check each against context."""

    # Step 1: extract claims from the answer
    claims_prompt = f"List every factual claim in this answer as a JSON array of strings:\n{answer}"
    claims_response = llm.invoke(claims_prompt).content
    try:
        claims = json.loads(claims_response)
    except Exception:
        claims = [answer]  # fallback

    if not claims:
        return 1.0

    # Step 2: verify each claim against context
    supported = 0
    for claim in claims:
        verify_prompt = (
            f"Context: {context}\n\n"
            f"Claim: {claim}\n\n"
            "Is this claim supported by the context? Reply only YES or NO."
        )
        verdict = llm.invoke(verify_prompt).content.strip().upper()
        if verdict.startswith("YES"):
            supported += 1

    return supported / len(claims)
```

---

## 3. Tracing with LangSmith

**LangSmith** records every LangChain invocation with full **input/output, timing, and token counts**. When no API key is available, we can mock the tracing interface.

```python
# With a real LangSmith account:
# export LANGCHAIN_TRACING_V2=true
# export LANGCHAIN_API_KEY=ls__...
# export LANGCHAIN_PROJECT=my-rag-project
# All subsequent LangChain calls are automatically traced.

import os

def setup_tracing(project_name: str = "rag-evaluation"):
    """Configure LangSmith tracing if credentials are available."""
    api_key = os.environ.get("LANGCHAIN_API_KEY")
    if api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = project_name
        print(f"LangSmith tracing enabled → project: {project_name}")
    else:
        print("LANGCHAIN_API_KEY not set — tracing disabled (running locally)")

setup_tracing()
```

### 3.1 Manual Run Logging (Mock)

When LangSmith is not available, log runs to a **local JSONL file**:

```python
import json
import time
import uuid
from pathlib import Path
from functools import wraps
from typing import Any, Callable

TRACE_FILE = Path("./traces.jsonl")

def trace_llm_call(run_name: str):
    """Decorator that logs LLM calls to a local JSONL trace file."""
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs) -> Any:
            run_id = str(uuid.uuid4())[:8]
            start = time.perf_counter()
            error = None
            result = None
            try:
                result = fn(*args, **kwargs)
                return result
            except Exception as e:
                error = str(e)
                raise
            finally:
                elapsed = time.perf_counter() - start
                record = {
                    "run_id": run_id,
                    "name": run_name,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "latency_s": round(elapsed, 4),
                    "inputs": {"args": str(args)[:200], "kwargs": str(kwargs)[:200]},
                    "output": str(result)[:500] if result is not None else None,
                    "error": error,
                }
                with TRACE_FILE.open("a") as f:
                    f.write(json.dumps(record) + "\n")
        return wrapper
    return decorator


@trace_llm_call("summarize")
def summarize(text: str) -> str:
    # Mocked — replace with real LLM call
    return f"Summary of: {text[:50]}..."

summarize("The H100 GPU achieves 3.35 TB/s memory bandwidth using HBM3...")
print(f"Trace written to {TRACE_FILE}")
```

---

## 4. Cost Tracking

```python
import os
import time
import uuid
from dataclasses import dataclass, field
from openai import OpenAI

# Example pricing per million tokens.
# Do not use this table for billing decisions. Keep real prices in deployment
# config and update them from the current provider pricing page.
PRICING = {
    "fast-model": {"input": 0.15, "output": 0.60},
    "balanced-agent-model": {"input": 3.00, "output": 15.00},
    "reasoning-model": {"input": 15.00, "output": 75.00},
    "embedding-model": {"input": 0.02, "output": 0.0},
}

@dataclass
class CostTracker:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    calls: list[dict] = field(default_factory=list)

    def record(self, model: str, input_tokens: int, output_tokens: int, latency_s: float):
        pricing = PRICING.get(model, {"input": 0.0, "output": 0.0})
        cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost_usd += cost
        self.calls.append({
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": round(cost, 6),
            "latency_s": round(latency_s, 4),
        })

    def summary(self) -> dict:
        return {
            "session_id": self.session_id,
            "total_calls": len(self.calls),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
        }


# Global tracker for the session
tracker = CostTracker()

def tracked_completion(messages: list[dict], model: str | None = None) -> str:
    """OpenAI chat completion with automatic cost tracking."""
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "sk-fake"))
    model = model or os.environ.get("OPENAI_MODEL", "your-model-id")
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model, messages=messages, temperature=0
    )
    elapsed = time.perf_counter() - start
    usage = response.usage
    tracker.record(model, usage.prompt_tokens, usage.completion_tokens, elapsed)
    return response.choices[0].message.content


# Demo
# answer = tracked_completion([{"role": "user", "content": "What is HBM3?"}])
# print(tracker.summary())
```

---

## 5. Structured LLM Call Logger

```python
import json
import logging
import time
import uuid
from pathlib import Path
from openai import OpenAI

# Configure structured logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("llm_logger")
log_file = Path("llm_calls.jsonl")

class LLMLogger:
    """Logs every LLM call to both console and a JSONL file."""

    def __init__(self, model: str | None = None, log_path: Path = log_file):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "sk-fake"))
        self.model = model or os.environ.get("OPENAI_MODEL", "your-model-id")
        self.log_path = log_path

    def _write(self, record: dict):
        self.log_path.open("a").write(json.dumps(record) + "\n")
        # Brief console log
        logger.info(
            f"[LLM] call_id={record['call_id']} "
            f"model={record['model']} "
            f"tokens={record['usage']['total_tokens']} "
            f"latency={record['latency_s']:.3f}s "
            f"cost=${record['cost_usd']:.5f}"
        )

    def complete(self, messages: list[dict], **kwargs) -> str:
        call_id = uuid.uuid4().hex[:8]
        start = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=0, **kwargs
        )
        latency = time.perf_counter() - start
        usage = response.usage
        pricing = PRICING.get(self.model, {"input": 0.0, "output": 0.0})
        cost = (
            usage.prompt_tokens * pricing["input"]
            + usage.completion_tokens * pricing["output"]
        ) / 1_000_000

        record = {
            "call_id": call_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": self.model,
            "messages": [{"role": m["role"], "content": m["content"][:300]} for m in messages],
            "response": response.choices[0].message.content[:500],
            "usage": {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            },
            "latency_s": round(latency, 4),
            "cost_usd": round(cost, 6),
        }
        self._write(record)
        return response.choices[0].message.content


# Usage
# llm = LLMLogger()
# answer = llm.complete([{"role": "user", "content": "Explain HBM3 in one sentence."}])
```

---

## 6. Building an Evaluation Dataset from Production Traffic

```python
import json
import random
from pathlib import Path
from datetime import datetime

TRACE_FILE = Path("./traces.jsonl")
EVAL_DATASET_FILE = Path("./eval_dataset.json")


def sample_production_traces(
    trace_file: Path,
    sample_size: int = 50,
    min_response_length: int = 50,
) -> list[dict]:
    """Sample high-quality traces to build a labeled eval dataset."""
    traces = []
    if not trace_file.exists():
        print(f"Trace file not found: {trace_file}")
        return []

    with trace_file.open() as f:
        for line in f:
            try:
                record = json.loads(line)
                if (
                    record.get("output")
                    and len(record["output"]) >= min_response_length
                    and not record.get("error")
                ):
                    traces.append(record)
            except json.JSONDecodeError:
                continue

    sample = random.sample(traces, min(sample_size, len(traces)))
    print(f"Sampled {len(sample)} traces from {len(traces)} total")
    return sample


def build_eval_dataset(traces: list[dict]) -> list[dict]:
    """
    Convert production traces into evaluation examples.
    In production: send these to human annotators for labeling.
    Here: auto-generate placeholder labels.
    """
    dataset = []
    for trace in traces:
        dataset.append({
            "id": trace.get("run_id", "unknown"),
            "question": _extract_question(trace),
            "answer": trace.get("output", ""),
            "label": None,       # to be filled by human annotator
            "score": None,       # to be filled by judge
            "sampled_at": datetime.utcnow().isoformat(),
        })
    EVAL_DATASET_FILE.write_text(json.dumps(dataset, indent=2))
    print(f"Saved {len(dataset)} eval examples → {EVAL_DATASET_FILE}")
    return dataset


def _extract_question(trace: dict) -> str:
    """Pull the user question from a trace record."""
    inputs = trace.get("inputs", {})
    if isinstance(inputs, dict):
        for key in ("question", "query", "input", "user_message"):
            if key in inputs:
                return inputs[key]
    return str(inputs)[:200]
```

---

## 7. A/B Testing Prompts

```python
import hashlib
import random
from typing import Callable

PROMPT_A = """Answer the following question concisely using only facts from the context.
Context: {context}
Question: {question}"""

PROMPT_B = """You are a precise technical assistant. Using ONLY the provided context,
answer the question. If unsure, say "I don't know."
Context: {context}
Question: {question}"""


def ab_router(user_id: str, variants: list[str], weights: list[float] | None = None) -> str:
    """
    Deterministically assign a user to a prompt variant using their ID hash.
    Same user always gets the same variant (sticky assignment).
    """
    if weights:
        # Weighted random assignment (not deterministic, for gradual rollout)
        return random.choices(variants, weights=weights)[0]

    # Deterministic: hash user_id to choose variant
    hash_int = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
    return variants[hash_int % len(variants)]


class PromptABTest:
    def __init__(self, variant_a: str, variant_b: str):
        self.variants = {"A": variant_a, "B": variant_b}
        self.results: dict[str, list[float]] = {"A": [], "B": []}

    def run(self, user_id: str, context: str, question: str, judge_fn: Callable) -> dict:
        variant_key = ab_router(user_id, ["A", "B"])
        prompt = self.variants[variant_key].format(context=context, question=question)

        # Generate answer (mock here)
        answer = f"[Variant {variant_key}] Answer about: {question[:40]}..."

        # Score the answer
        score = judge_fn(question=question, reference="ground truth placeholder", answer=answer)
        self.results[variant_key].append(score["score"])

        return {"variant": variant_key, "answer": answer, "score": score["score"]}

    def report(self) -> dict:
        report = {}
        for key, scores in self.results.items():
            if scores:
                report[key] = {
                    "n": len(scores),
                    "mean_score": round(sum(scores) / len(scores), 2),
                    "min": min(scores),
                    "max": max(scores),
                }
        winner = max(report, key=lambda k: report[k]["mean_score"]) if report else None
        return {"variants": report, "winner": winner}


# Demo
# ab = PromptABTest(PROMPT_A, PROMPT_B)
# for i in range(20):
#     ab.run(f"user_{i}", context="H100 has 3.35 TB/s bandwidth.", question="H100 bandwidth?", judge_fn=judge_answer)
# print(ab.report())
```

---

## Key Takeaways

- **LLM-as-judge** scales evaluation to thousands of examples quickly. Use a rubric with clear integer scores (0-5) and require JSON output for reliable parsing.
- **RAGAS** gives you four orthogonal RAG metrics. Faithfulness catches hallucinations; context recall diagnoses retrieval gaps.
- **LangSmith** tracing is zero-code with environment variables set. Always trace in staging even if not in production.
- **Cost tracking** should accumulate per-session and log per-call. Early visibility on costs prevents budget surprises.
- **Structured call logging** (JSONL) is your flight recorder — essential for debugging failures and building eval datasets.
- **A/B testing** with sticky user assignment ensures reproducible experiments. Always run for a statistically meaningful number of samples before declaring a winner.

---

## Exercises

### Exercise 1 — Multi-Criteria Judge

Extend the LLM-as-judge to evaluate on three separate criteria: (1) factual accuracy, (2) completeness, and (3) conciseness. Each criterion should return a score of 0–5 with a one-sentence justification. Build a `MultiCriteriaJudge` class that averages the three scores and returns a combined verdict. Test it on at least 5 question-answer pairs from a domain you know well.

### Exercise 2 — RAGAS on Your Own Data

Create a small corpus of 5–10 documents on any technical topic you choose. Write 5 test questions with known ground-truth answers. Build a simple RAG system (from Lecture 11/10), run it on all 5 questions, and evaluate using RAGAS. Report all four metric scores in a table. For any metric below 0.7, identify one concrete change to the RAG system that would improve it.

### Exercise 3 — Cost Dashboard

Build a `CostDashboard` class that reads from the JSONL log file produced by `LLMLogger` and renders a summary report. The report should include: total spend, spend by model, average cost per call, top 5 most expensive calls (with their inputs), and a call count time series grouped by hour. The report should be printable as a formatted text table.

---

*Next: [Lecture 24](Lecture-24.md)*
