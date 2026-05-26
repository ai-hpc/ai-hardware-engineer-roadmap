# Module 07 — Long-Context Evaluation

**Parent:** [Long-Context MoE Foundation Training](README.md)

**One-line purpose:** Build an honest evaluation harness that measures what matters — per-position retrieval, multi-hop reasoning, long-document synthesis, codebase tasks — and explain why perplexity alone is misleading at long context.

**Prerequisites:** Modules 01–06. Familiarity with standard LLM eval tooling (`lm-evaluation-harness`, `vllm`, etc.).

**Artifact:** A reproducible harness producing a per-length, per-task breakdown across RULER, LongBench, and at least one needle-in-haystack variant; a written summary of where your model is and is not actually using long context.

---

## Why it matters

A model that scores well on a long-context benchmark **on average** can still fail catastrophically at specific positions or specific task types. Without a per-position, per-task breakdown you cannot tell whether your model "supports 256K context" or "supports 4K and gets lucky."

Every architectural and data change you make in this course should be paired with an eval delta. This module is the foundation for those deltas.

---

## Mental model

### Why perplexity at long context lies

Perplexity is the average negative log-likelihood per token. At long context, the per-position likelihood is dominated by **easy** local-syntax tokens. A few catastrophically wrong tokens at the answer-bearing position contribute almost nothing to the average. The perplexity curve can look fine while retrieval is broken.

**Rule**: never publish a "long-context model" claim based on perplexity alone. Always pair with a downstream task.

### The benchmark families you must run

#### Needle-in-a-Haystack (NIAH)

The simplest probe. Insert a small fact ("the secret code is `BLUE123`") at a controlled position inside a long document of unrelated text, then ask the model to retrieve it.

Variants:

- **Single needle**: one fact, one position.
- **Multi-needle**: several facts, must retrieve all.
- **Distractor-heavy**: similar-looking but wrong facts scattered through the context.

Metric: per-(needle position, total length) accuracy. Plot as a heatmap; "lost in the middle" appears as a darker diagonal band.

#### RULER

A modern long-context benchmark covering 13 task types: NIAH variants, variable tracking, common-words extraction, frequent-words extraction, multi-key NIAH, multi-value NIAH, multi-query NIAH, QA, and more. Evaluates at multiple context lengths (4K, 8K, 16K, 32K, 64K, 128K).

RULER is the current default reference for long-context model claims. Source: <https://github.com/hsiehjackson/RULER>.

#### LongBench

Realistic-task benchmark with 21 tasks across multi-doc QA, summarization, few-shot learning, synthetic, code completion. Less synthetic than RULER, more "what would users actually ask." Source: <https://github.com/THUDM/LongBench>.

#### LongBench-v2 / InfiniteBench / 1M-context probes

Push beyond 128K. As of 2026 these are useful for marketing-grade long-context claims (Gemini 2.0, Claude-4 long-context); they are noisier than RULER but cover the regime that RULER does not yet exercise.

#### Code-specific long-context

- **RepoQA / CrossCodeEval**: cross-file code reasoning.
- **SWE-Bench**: agentic patch generation across a real repo.

If your target use case involves codebases, these matter more than RULER.

#### Multi-hop QA

- **HotpotQA / MuSiQue**: chains of reasoning across distant evidence in long inputs.

The hardest end of long context — requires both retrieval and composition.

### What broad evaluation should also cover

A long-context model that regresses on short-context tasks is broken. Always include:

- **MMLU / MMLU-Pro**: knowledge.
- **GPQA Diamond**: hard reasoning.
- **HumanEval / MBPP / LiveCodeBench**: code.
- **IFEval / Arena-Hard**: instruction following.
- **HellaSwag / WinoGrande / Arc-Challenge**: common-sense (less informative for modern models but cheap to run).

The shape of a healthy long-context fine-tune: long-context metrics improve, short-context metrics hold within `±1%`.

### Honest reporting

A long-context evaluation report should always include:

1. **Per-position heatmap** for NIAH-style probes (not just aggregate accuracy).
2. **Per-context-length curve** for RULER tasks (`L = 4K, 8K, ..., 128K`).
3. **Per-task breakdown**, not a single LongBench average.
4. **Adjacent broad benchmarks** to show short-context did not regress.
5. **Confidence intervals**: long-context evals have small N (often 100–200 examples per cell). Bootstrap CIs are mandatory.

A single "RULER avg = 73%" number tells you almost nothing.

---

## Build it

### 1. NIAH probe

The minimal version:

```python
# niah.py
import random
from pathlib import Path

def make_haystack(filler_text, target_len_tokens, tokenizer):
    # Repeat filler until length target met
    out = []
    while len(tokenizer(" ".join(out)).input_ids) < target_len_tokens:
        out.append(filler_text)
    return " ".join(out)

def insert_needle(haystack, needle, pos_fraction, tokenizer):
    ids = tokenizer(haystack).input_ids
    needle_ids = tokenizer(needle).input_ids
    insert_at = int(len(ids) * pos_fraction)
    new = ids[:insert_at] + needle_ids + ids[insert_at:]
    return tokenizer.decode(new)

def probe_one(model, tokenizer, context_len, pos_fraction, needle, question, filler):
    haystack = make_haystack(filler, context_len, tokenizer)
    text = insert_needle(haystack, needle, pos_fraction, tokenizer)
    prompt = f"{text}\n\nQ: {question}\nA:"
    out = model.generate(prompt, max_tokens=64, temperature=0.0)
    return needle.lower() in out.lower()
```

Sweep `context_len ∈ {4K, 8K, 16K, 32K, 64K, 128K}` and `pos_fraction ∈ {0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0}`. With 10 needles per cell and bootstrap CIs, you get ~70 cells × 10 = 700 generations — feasible on one GPU in an hour.

Plot as a heatmap with `pos_fraction` on the x-axis and `context_len` on the y-axis.

### 2. RULER

Clone the official repo. It ships generation and scoring scripts:

```
git clone https://github.com/hsiehjackson/RULER
cd RULER
# Configure model endpoint (vLLM, HuggingFace, or your own)
bash run.sh <model> <context_length>
```

Produces per-task scores at each context length. Save the CSV.

### 3. LongBench

```
git clone https://github.com/THUDM/LongBench
cd LongBench
python pred.py --model <your_model>
python eval.py
```

Per-task scores in `result.json`. Save it.

### 4. Broad short-context basket

Use `lm-evaluation-harness`:

```
lm_eval --model vllm --model_args pretrained=<your_model> \
    --tasks mmlu,mmlu_pro,gpqa_diamond,humaneval,mbpp,ifeval,arc_challenge \
    --batch_size auto --output_path eval_short.json
```

### 5. Bring it together

A single dashboard:

```
| Task        | Context | Score | CI    | Notes              |
|-------------|---------|-------|-------|--------------------|
| NIAH mean   | 32K     | 91.2  | ±1.4  |                    |
| NIAH worst-pos | 32K  | 76.0  | ±3.1  | pos_fraction=0.5   |
| RULER avg   | 32K     | 78.5  | ±0.8  | 13-task mean       |
| LongBench   | 32K     | 49.2  | ±1.1  | 21-task mean       |
| MMLU        | 4K      | 67.8  | ±0.5  | (unchanged)        |
| HumanEval   | 4K      | 71.2  | ±2.4  | (unchanged)        |
```

This is what you report. Not just the mean.

---

## Use it in the real stack

vLLM and SGLang both expose an OpenAI-style endpoint that all these harnesses point at. Spin up your model behind one of those, then run RULER / LongBench / `lm-evaluation-harness` against the endpoint.

For long-context inference, configure vLLM with the matching position-encoding scaling (RoPE base / YaRN). If your model config encodes it, vLLM picks it up; otherwise pass via CLI: `--rope-scaling '{"type":"yarn","factor":4,"original_max_position_embeddings":32768}'`.

For the cacheon-sglang-miner project you worked on, the same external benchmark harnesses give you the "outside-its-training-distribution" signal that internal validators cannot provide.

---

## Measure it

The evaluation itself has costs. Track:

- Total tokens generated per eval run.
- Wall-clock per benchmark (RULER at 128K can take many hours).
- API/compute cost per eval iteration.

A target cadence: a full eval basket per iteration, with smaller daily smoke evals (one RULER task at one length, a small NIAH grid) between full runs.

---

## Ship it

Drop into `lcm-course/`:

1. `niah.py` and its heatmap PNG.
2. `ruler_results.csv` (per-task, per-length).
3. `longbench_results.json`.
4. `eval_short.json` from `lm-evaluation-harness`.
5. `eval_dashboard.md` — the table above filled in for one of your runs, with bootstrap CIs and a written paragraph identifying where this model genuinely uses long context vs where it does not.

---

## Related pages

- [Module 03 — Position encoding](03-Position-Encoding.md)
- [Module 06 — Adaptive data pipelines](06-Adaptive-Data-Pipelines.md)
- [Module 10 — From experiment to general-purpose model](10-General-Purpose-Model.md)
- RULER: <https://github.com/hsiehjackson/RULER>
- LongBench: <https://github.com/THUDM/LongBench>
- lm-evaluation-harness: <https://github.com/EleutherAI/lm-evaluation-harness>
- "Lost in the Middle" paper: <https://arxiv.org/abs/2307.03172>
