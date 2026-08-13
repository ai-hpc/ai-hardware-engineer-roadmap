# Module 10 — From Experiment to General-Purpose Model

**Parent:** [Long-Context MoE Foundation Training](README.md)

**One-line purpose:** Validate that the model you trained is genuinely useful **outside** the narrow distribution of its training and internal evaluation — through honest external benchmarking, real-task harnesses, public comparison, and an explicit decision on what is shippable.

**Prerequisites:** Modules 01–09. A trained model checkpoint and the eval harness from Module 07.

**Artifact:** A capstone report comparing your model against a public open-source baseline on tasks **outside** your training distribution, with honest "where we win, where we lose, where we draw" and a shippable decision.

---

## Why it matters

It is possible — and common — to train a model that scores well on internal validation, your custom evaluation harness, and even some public benchmarks, while being **bad** at the tasks real users would actually run on it. The failure mode has a few names: distribution overfitting, benchmark gaming, reward hacking. The cure is to evaluate on tasks that were never part of any feedback loop during training, and to compare against models trained by other teams under their own incentives.

This module is the discipline that separates "we shipped a training pipeline" from "we shipped a useful model."

---

## Mental model

### The three failure modes you are guarding against

#### 1. Eval contamination

The training data accidentally contains the evaluation data, or paraphrases of it. The model "wins" the benchmark by recall rather than reasoning.

Defense: hash-of-document overlap checks between every training shard and every eval set. Standard tools: `BigBench-Hard-Contamination`, `lm-evaluation-harness` contamination flags.

#### 2. Distribution overfitting

The training data and the internal eval data live in a narrow distribution (e.g. "QA over Wikipedia passages") that the model becomes specialized for. The model fails on adjacent but realistic tasks (a different document style, an unusual question format).

Defense: evaluate on tasks the model has **provably** never been targeted at. Treat public benchmarks the team did not pick for the project as adversarial.

#### 3. Reward hacking

If the training feedback loop is a scalar reward (RLHF, custom validators, internal metrics), the model can find ways to maximize the reward signal without solving the underlying task. Even pretraining loops have shadow versions of this — adaptive-data loops over-targeting one metric is reward hacking by another name.

Defense: rotate evaluation targets, never train on the evaluation set, watch broad-basket metrics for collateral damage.

### What "outside the training distribution" really means

Strict version: a task whose data was generated **after** your training data cutoff, by people who did not know about your project.

Practical version: a public benchmark that the team did not optimize for during the loop. The model has never seen its examples, no targeted data has been added because of failures on it.

You need both versions. The strict version protects against subtle leakage. The practical version is what a hiring manager or external user will recognize.

### A benchmark basket for general usefulness

This is on top of the long-context basket from Module 07.

#### General reasoning and knowledge

- **MMLU-Pro** — harder than MMLU, more discriminative for modern frontier models.
- **GPQA Diamond** — graduate-level science questions designed to resist search.
- **BBH (BigBench Hard)** — multi-step reasoning tasks.

#### Coding

- **HumanEval / MBPP** — entry-level.
- **LiveCodeBench** — refreshed periodically, contamination-resistant.
- **SWE-Bench / SWE-Bench-Verified** — full-repo agentic patches.

#### Instruction following

- **IFEval** — instruction-following at the level of "respond in exactly 3 sentences, ending with a period."
- **MT-Bench** / **Arena-Hard** — multi-turn conversation with judge model.

#### Tool use and agents

- **WebArena / VisualWebArena** — agentic browsing.
- **OSWorld** — desktop / OS agent.
- **τ-bench** — function-calling realism.

#### Long-form generation quality

- Pairwise human (or LLM-judge) comparisons against a strong baseline on real prompts.

#### Multilingual

- **XCOPA / XTREME / Belebele** — if your model claims multilingual support, you must show it.

You will not run all of these. Pick the basket that matches your target deployment.

### The comparison harness

Pair every benchmark with a strong public baseline:

- **Dense baselines**: Llama 3.1 70B Instruct, Qwen 2.5 72B Instruct.
- **MoE baselines**: Mixtral 8×22B Instruct, Qwen 2.5 MoE A14B, DeepSeek-V3.
- **Long-context baselines**: Gemini 2.0 Pro long-context, Claude 4 long-context, Llama 3.1 405B Instruct.

For each benchmark, report:

- Your model's score (with bootstrap CI).
- Baseline's score (with bootstrap CI).
- Delta and significance.
- A one-line interpretation: clear win / clear loss / within noise.

### Honest "shippable" decision

After the comparison, write the shippable-decision section:

- **Where the model is competitive** — name the specific benchmarks and the margin.
- **Where the model is not competitive** — name them.
- **The deployment shape this implies** — e.g. "this model is good for long-document QA and code-repository reasoning; it is not a general chat replacement; do not market it as such."

Models without this section get over-claimed and embarrass their authors.

### Beyond benchmarks: real-world dogfooding

Benchmarks are necessary but insufficient. Before declaring a model usable:

- **Internal team dogfooding** for at least a few days on real work tasks.
- **Latency / throughput at target deployment scale** (KV cache fit at target context, TPS at target batch size).
- **Failure-mode survey on real prompts**: collect 100 prompts from your target use case, classify the outputs (correct / partial / wrong / refusal), report the distribution.

This is the kind of evidence that justifies an external launch.

---

## Build it

### 1. Contamination check

```bash
# Using lm-eval-harness's contamination check
python -m lm_eval --tasks mmlu_pro \
    --decontamination ngram_size=13 \
    --decontamination_ngrams_path /path/to/your/train/ngrams \
    --model dummy --output_path contamination.json
```

Or roll your own MinHash overlap between training shards and eval prompts. Flag and remove any eval items with ≥ 50% MinHash similarity to a training document.

### 2. Run the basket

Pick 8–12 benchmarks across the categories above. Run them through `lm-evaluation-harness` for the standard ones, project-specific harnesses for the rest (SWE-Bench has its own; WebArena has its own).

Run **the same baseline model through the same harnesses** at the same time. Score parity matters: a "your model scores 67 on MMLU" with no baseline number on your harness is meaningless.

### 3. Long-context external sweep

Reuse your RULER + LongBench runs from Module 07. Run the same benchmarks on the baseline. Plot pair-wise per-task scores.

### 4. Human / LLM-judge head-to-head

For long-form generation:

- Sample 50 realistic prompts from your target use case.
- Generate from your model and from the baseline.
- Anonymize the outputs; have at least three human raters (or a careful judge prompt to a strong judge model) score pairwise preference.
- Report win/loss/tie counts with bootstrap CIs.

### 5. The capstone report

Use this template:

```
# <Project name> — capstone report

## Model
- Architecture: dense / MoE (E experts, top-k)
- Context: trained at N, evaluated up to N
- Training tokens: T
- Data mix: <summary>

## Headline numbers
| Benchmark         | Our model | Baseline    | Δ      | Note |
|-------------------|-----------|-------------|--------|------|
| MMLU-Pro          | ...       | Llama 3.1 ..|  +0.x  | win  |
| HumanEval         | ...       |             |        |      |
| RULER 32K avg     | ...       |             |        |      |
| LongBench         | ...       |             |        |      |
| SWE-Bench Verified| ...       |             |        |      |

## Where we win
<one paragraph>

## Where we lose
<one paragraph, including a hypothesis>

## Within-noise / draws
<one paragraph>

## Contamination check
<one paragraph with overlap stats>

## Real-task survey
<bullet summary of the 100-prompt classification>

## Shippable decision
"This model is fit for <task list>. It is not fit for <task list>.
 Marketing language: <one sentence>.
 Known limitations: <bullet list>.
 Next investment: <one paragraph>."
```

---

## Use it in the real stack

When a team publishes a model card, this report is what fills the "evaluation" section. Look at recent Llama / Qwen / Mistral / DeepSeek model cards; the good ones include almost exactly this structure (basket of benchmarks, baseline comparison, qualitative dogfooding, explicit limitations). The weak ones publish a single average number.

For internal tracking, a similar report should be produced at every major training milestone, version-controlled, and linked from the run's wandb / mlflow record.

---

## Measure it

- **Coverage**: how many benchmark families are represented in the basket?
- **Baseline parity**: every metric paired with at least one credible baseline on the same harness?
- **Confidence intervals**: every reported number with a CI?
- **Contamination**: explicit overlap check completed, with documented residual risk?
- **Dogfooding**: at least N realistic prompts surveyed with output classification?

A capstone report that answers "yes" to all five is shippable. One that answers "yes" to two is not.

---

## Ship it

In `lcm-course/capstone/`:

1. `capstone_report.md` — the filled-in template above.
2. `benchmark_basket.csv` — per-benchmark scores for your model + baseline + delta + CI.
3. `contamination_report.md` — methodology and overlap stats.
4. `dogfood_survey.csv` — the 100-prompt classification with rater notes.
5. `headline_plot.png` — a single bar / radar chart making the headline visual.

This package is the deliverable a hiring manager, a CTO, or a future user can read in 10 minutes and understand what your model is for.

---

## Where to go after this course

- **Frontier training research**: scaling laws, MoE routing innovations, long-context architectures (state-space hybrids, ring-attention variants), training-time RL.
- **Post-training**: SFT, DPO, RLHF, GRPO, agentic fine-tuning, tool-use post-training.
- **Inference systems**: vLLM / SGLang / TensorRT-LLM internals, the cacheon-style production serving stack, FP8 / FP4 inference, speculative decoding.
- **Evaluation depth**: building per-customer eval harnesses, judge-model alignment, red-team / safety eval, agentic-task eval design.
- **Hardware co-design**: training for Blackwell-class hardware, FP4 numerics, distributed checkpoint formats for very large models, fault-tolerant training research.

The frontier moves fast. This course gives you the systems-and-method foundation needed to follow along.

---

## Related pages

- [Module 06 — Adaptive data pipelines](06-Adaptive-Data-Pipelines.md)
- [Module 07 — Long-context evaluation](07-Long-Context-Evaluation.md)
- [Module 09 — Distributed training infrastructure](09-Distributed-Training-Infrastructure.md)
- [README — course overview](README.md)
- lm-evaluation-harness: <https://github.com/EleutherAI/lm-evaluation-harness>
- SWE-Bench: <https://github.com/princeton-nlp/SWE-bench>
- LiveCodeBench: <https://github.com/LiveCodeBench/LiveCodeBench>
- Arena-Hard: <https://github.com/lmarena/arena-hard-auto>
