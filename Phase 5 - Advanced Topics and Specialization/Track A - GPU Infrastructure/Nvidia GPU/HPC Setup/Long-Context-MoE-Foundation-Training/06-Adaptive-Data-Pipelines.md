# Module 06 — Adaptive Data Pipelines

**Parent:** [Long-Context MoE Foundation Training](README.md)

**One-line purpose:** Build a closed-loop data pipeline that turns evaluation failure modes into next-iteration training examples, with the data-quality, length-curriculum, and deduplication discipline that long-context MoE training actually needs.

**Prerequisites:** Modules 01–05. Familiarity with HuggingFace `datasets`, tokenizer pipelines, deduplication tools (`text-dedup`, MinHash).

**Artifact:** One full closed loop — train baseline, evaluate, identify a failure mode, generate or filter targeted data, retrain, re-evaluate, and document the delta.

---

## Why it matters

A static, "scraped + filtered" dataset will get you to the average performance of every model trained on a similar mix. To go further, the data must respond to the model's actual weaknesses. This is the difference between a model that scores well on benchmarks by coincidence and a model that is reliably good at the things you care about.

For long-context + MoE specifically, the data discipline is harder than for dense short-context training because: long documents are scarce, length distribution matters (curriculum), and MoE experts only specialize if the data has enough internal heterogeneity to differentiate.

---

## Mental model

### The closed loop

```
[base model]
     │
     ▼
[training run on dataset_v_i]
     │
     ▼
[evaluation harness — broad + targeted]
     │
     ▼
[failure-mode triage]    ←─ explicit categorization
     │
     ▼
[targeted data generation / curation / filtering]
     │
     ▼
[dataset_v_{i+1} = mix(dataset_v_i, targeted_additions)]
     │
     └──────────────────► back to training
```

Two design decisions distinguish a serious loop from a notebook demo:

1. **Failure-mode triage is explicit.** "The eval went down" is not a category. "Per-position retrieval accuracy in positions 40K–80K dropped" is.
2. **Targeted additions are bounded.** You do not replace the dataset; you mix in `5–15%` of new data targeting the failure mode, controlled by a config you can roll back.

### Data quality fundamentals (the table-stakes layer)

Before anything adaptive, you need standard data hygiene. Skipping these means your "improvements" later are confounded by noise.

| Step | Tool family | Why |
|------|-------------|-----|
| Document-level dedup | MinHash + LSH, `text-dedup` | Prevents memorization of duplicated boilerplate |
| Substring dedup | Suffix arrays (Google's `deduplicate-text-datasets`) | Removes near-duplicate spans that fool MinHash |
| Quality filter | KenLM perplexity + simple heuristic rules | Removes machine-generated SEO sludge |
| Language ID + filter | fastText / GlotLID | Drops mismatched-language documents |
| PII scrub | Pattern-based + classifier | Compliance and downstream safety |
| Toxicity filter | Per-domain threshold | Aligns with target use cases |

The MoE-and-long-context layer comes on top of this. A dirty corpus does not get better by being mixed with adaptive additions.

### Length curriculum

A direct jump from 4K to 256K training sequences is wasteful: the model spends most of its early gradient steps figuring out positional patterns it could have learned faster on shorter inputs first.

A typical schedule:

| Stage | Tokens | Sequence length | Position-encoding setting |
|-------|--------|-----------------|---------------------------|
| Base pretrain | 1T+ | 4K | base RoPE θ = 10000 |
| Extension I | 50–100B | 32K | RoPE θ → 500_000 (or YaRN factor=4) |
| Extension II | 20–50B | 128K | YaRN factor=16 |
| Extension III | 5–20B | 256K–1M | YaRN factor=32+, very careful data |

The token counts are illustrative; the principle is: each stage uses ~5–10× fewer tokens than the previous, because they are progressively more expensive per token.

**Crucially**, each extension stage's data must contain documents that genuinely need the longer context. Padding short documents to 256K with random text teaches the model that long positions are noise.

### Sources of long-context-rich data

- **Books**: long-form, high coherence. Books3 was the canonical source pre-controversy; reproducible alternatives include public-domain corpora, ArXiv full-text, Project Gutenberg.
- **Codebases**: file-level + repo-level. Allows multi-file reasoning if structured properly.
- **Web crawls of long pages**: documentation, wikis, legal filings, scientific papers.
- **Conversational logs / multi-turn agent traces**: where actually-using-prior-context matters.
- **Synthetic long-context tasks**: chained reasoning, retrieval-augmented documents, multi-hop QA. Necessary for teaching active context use; risky if quality is low.

For a real long-context MoE foundation training, the mix is typically `~30%` code, `~30%` books/papers, `~20%` curated web, `~10%` conversations, `~10%` synthetic — with the exact split tuned per evaluation failure.

### What "adaptive" actually changes

Adaptive does not mean "the model picks its own data" (that is dangerous: reward hacking, distribution narrowing). It means **the human-controlled pipeline routes evaluation failures into targeted data additions**.

Failure → targeted data mapping (examples):

| Failure mode | Targeted addition |
|--------------|-------------------|
| "Lost in the middle": retrieval accuracy dips at positions 30–70% | Documents structured to require mid-context retrieval; distractor-heavy synthetic data |
| Code multi-file reasoning weak | Repo-scoped tasks: "patch the bug in file X given files A–G" |
| Long chain-of-thought collapses past step 12 | Math/proof traces with verified long step chains |
| Multilingual long-context degrades vs English | Long-form non-English documents with parallel structure |
| MoE expert collapse on a domain | Domain rebalancing in the mix |

Each addition is **bounded**, **versioned**, and **measured**. If the targeted addition does not move the targeted metric in the next eval, you remove it.

### Synthetic data — when and how

Synthetic data is necessary at the long-context end (real 1M-token documents with informative dependencies are rare). It is also dangerous: generated by an LLM, it inherits and amplifies that LLM's biases and errors.

Rules:

- **Verify mechanically when possible.** Math: re-run the proof. Code: compile and test. Multi-hop QA: re-check answer against the provided context.
- **Bound the synthetic fraction.** Typically `<= 15%` of any extension stage's mix.
- **Diversify generators.** If all synthetic data comes from one generator model, you are aligning your model to its quirks.

### Anti-patterns that wreck this whole pipeline

- **Training on the evaluation set.** Catastrophic and embarrassingly common. Set up your eval and training shards with explicit "no overlap" enforcement at hash-of-document level.
- **Targeted addition for benchmark X moves benchmark X but tanks benchmark Y.** Watch a basket of broad metrics every iteration, not just the targeted one.
- **Aggressive deduplication after extension data is added.** If extension data is similar to base data, MinHash may delete the extension. Run dedup per-stage, not across stages.
- **Length curriculum without RoPE/YaRN adjustment.** You teach the model 32K, then evaluate at 256K with a position encoding that has never seen those positions.

---

## Build it

A real closed loop, scoped to one iteration:

### Step 0 — Baseline training and eval

Take an existing 8B-class long-context base model. Run the eval harness (Module 07 details). Save the per-task, per-position breakdown.

### Step 1 — Failure-mode triage

Open the eval report. Identify the largest single failure. Common ones for first iterations:

- "Position 32K–48K accuracy on needle is 65%, vs 92% at position 8K."
- "Multi-file Python edit task pass-rate is 18%."
- "Long-form summary coherence drops past 16K input."

Pick one. Write a one-sentence hypothesis: "*The model under-uses positions in the latter half of the context because the training mix has too few documents where the answer-relevant span is in the late part.*"

### Step 2 — Targeted data generation / curation

For the example above:

- Pull long-form documents (books, papers) ≥ 40K tokens.
- For each, generate a synthetic QA pair where the answer-bearing span is sampled uniformly across positions, with distractors interleaved.
- Verify: the answer must be reconstructable from the document.
- Filter: drop low-quality generations using a heuristic + a small judge model.

Aim for `~100M tokens` of targeted data — enough to matter, not enough to dominate.

### Step 3 — Mix and retrain

Construct `dataset_v_{i+1}` = `0.9 · dataset_v_i + 0.1 · targeted_data`. Continue training for a fixed token budget (typically `~5–20B` tokens — enough to see effect, short enough to iterate).

Use Megatron-LM's `--data-blend` or HuggingFace `datasets.interleave_datasets` with the appropriate ratio.

### Step 4 — Re-evaluate

Run the same eval harness. Compare:

- **Targeted metric**: should improve (otherwise the addition was wrong).
- **Adjacent metrics**: should not regress meaningfully.
- **Broad metrics**: should hold or improve.

### Step 5 — Document the delta

Write a short report:

```
What failure: positions 32K-48K retrieval accuracy 65% → ?
What hypothesis: under-coverage of mid-context relevant spans
What data: 100M synthetic QA tokens, position-uniform answer placement
What result: 65% → 84% on targeted; -1.2% on summary; +0.3% on broad average
Decision: keep this addition; iterate on summary next
```

This document is your loop's memory. Without it the loop is just random training.

---

## Use it in the real stack

- **Megatron-LM data-blend**: `--data-path 0.9 /path/to/v_i 0.1 /path/to/targeted`. Each weight is a fraction; weights need not sum to 1.0 (they are normalized).
- **HuggingFace `datasets.interleave_datasets`** with `stopping_strategy="all_exhausted"` produces a consistent mix.
- **Deduplication**: `text-dedup` (<https://github.com/ChenghaoMou/text-dedup>) for MinHash + LSH; Google's `deduplicate-text-datasets` for substring dedup.
- **Quality filtering**: CCNet (<https://github.com/facebookresearch/cc_net>) is the canonical pipeline, still useful as a reference for KenLM + heuristic combinations.

For synthetic data verification:

- **Code**: spin up a sandboxed `pytest` runner.
- **Math**: SymPy + Lean / Isabelle for proof verification.
- **QA**: judge-model with a temperature-0 strict prompt asking "is this answer derivable from this passage."

---

## Measure it

Per loop iteration:

- Targeted metric pre/post.
- Adjacent metrics pre/post.
- Broad benchmark basket pre/post.
- Wall-clock cost of the iteration (training tokens, GPU-hours).
- Cost per unit of metric improvement.

The cost-per-unit metric is what tells you whether to keep iterating on the same failure or move to the next one.

---

## Ship it

Drop into `lcm-course/`:

1. `data_pipeline_loop.md` — the full one-iteration report following the Step 5 template.
2. `dataset_v_i.yaml` / `dataset_v_{i+1}.yaml` — the actual mix configs.
3. `eval_before.csv` and `eval_after.csv` — paired evaluation outputs.
4. One Markdown table summarizing the decision: keep / discard / iterate.

---

## Related pages

- [Module 03 — Position encoding](03-Position-Encoding.md)
- [Module 07 — Long-context evaluation](07-Long-Context-Evaluation.md)
- [Module 10 — From experiment to general-purpose model](10-General-Purpose-Model.md)
- CCNet pipeline: <https://github.com/facebookresearch/cc_net>
- text-dedup: <https://github.com/ChenghaoMou/text-dedup>
- ICML 2025 Long Context FM workshop (data section): <https://longcontextfm.github.io/>
