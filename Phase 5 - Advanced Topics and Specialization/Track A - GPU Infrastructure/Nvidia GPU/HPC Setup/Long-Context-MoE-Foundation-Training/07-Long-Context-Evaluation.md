# Module 07 — Measure whether the model uses its context

A model can accept 128K tokens yet answer using only the last paragraph. An evaluation must separate **accepted length**, **retrieval**, **reasoning over evidence**, and **runtime cost**. A single aggregate score cannot tell you which failed.

**Before this lesson:** [the data split and token-count contract](06-Adaptive-Data-Pipelines.md). **Outcome:** calculate perplexity correctly, design a retrieval test without prompt leakage, and specify a paired benchmark comparison. The arithmetic below is executable on CPU. The external model benchmarks are source-checked integration plans, not runs performed by this lesson.

## 1. Average loss is not perplexity

For N valid next-token targets with natural-log negative log likelihoods ℓᵢ:

`mean NLL = sum(ℓᵢ) / N`, and `perplexity = exp(mean NLL)`.

Exclude padding and all ignored targets. Count each intended target once when scoring overlapping windows. Record context length, stride, document boundaries, tokenizer, and the exact target mask. Changing any of these can change the measurement even when model weights stay fixed.

Two batches have NLL sums 20 and 10, with 10 and 2 valid targets. The pooled mean NLL is 30/12=2.5 and perplexity is about **12.1825**. The unweighted average of batch mean losses is 3.5; the average of batch perplexities is about 77.90. Neither equals the token-weighted result.

Run from any directory:

```bash
python3 - <<'PYCODE'
import math
batches = [(20.0, 10), (10.0, 2)]  # summed natural-log NLL, valid targets
mean_nll = sum(s for s, n in batches) / sum(n for s, n in batches)
print('mean_nll', mean_nll)
print('perplexity', round(math.exp(mean_nll), 4))
PYCODE
```

Perplexity is comparable only under compatible tokenization and evaluation protocols. For answer quality, also score the actual task. A model can predict common filler well while missing the one safety-relevant sentence. The [real two-process CPU collectives lab](../NCCL-Deep-Dive/CPU-Collectives-Lab/Guide.md) demonstrates why unequal valid-token counts also require care when averaging gradients.

## 2. Build a retrieval fixture that cannot pass by accident

Use a generated key/value record, such as `inspection_code = ZXQ-4187`, among unrelated records. Ask for the value of that key, and compare the **newly generated answer** with the expected value under a predefined normalization rule. Searching the returned prompt plus completion would find the answer already in the input and falsely pass every case.

Use different keys and values for independent trials. Confirm that the answer is absent from filler and the question, and that no conflicting record has the same key. Include absent-key cases with an explicit unknown answer, distractor records, and multiple-evidence tasks. A single unique phrase tests retrieval; it does not establish reasoning or reliable tool use.

Account for the final serialized prompt, including system messages, chat-template tokens, question, and special tokens. For an illustrative total budget of 4,096 tokens, reserve 64 for output. If the final prompt wrapper/question uses 160 and the needle uses 32, filler has at most **3,840** tokens. Re-tokenize the complete serialized prompt to check the actual count: component token counts need not add exactly after concatenation. Reject or rebuild overlength cases instead of letting a server silently truncate them.

A token-based filler generator should tokenize once, reject empty filler, allocate the budget, insert at a recorded token position, and verify the final input. Repeatedly tokenizing an ever-growing string wastes work and can loop forever on empty input. Report actual token counts and positions, not only requested character percentages.

For six lengths, seven insertion positions, and ten trials per cell, there are **420 requests**. This is a design count, not a one-hour runtime promise. Share fixture IDs between candidates so comparison is paired; repeated positions from the same source are related observations.

## 3. Choose a benchmark for a specific question

| Evaluation | What it helps answer | What a high score leaves unresolved |
|---|---|---|
| Generated retrieval and absent-key fixtures | Can this configuration recover a defined fact at each tested length/position? | Real-document reasoning and robustness to ambiguity |
| RULER's configured synthetic tasks | How does performance vary across retrieval and other controlled tasks? | Broad real-world competence |
| LongBench v1 or v2 with a pinned protocol | How does the model handle the selected real-document tasks? | Unseen-data provenance and your application's success rate |
| Held-out application tasks | Does the candidate satisfy the selected use case? | Generalization beyond that task distribution |

RULER's repository describes four task categories and 13 original task configurations. Its checked `scripts/run.sh` takes **model name and benchmark name**; the second argument is not context length. Models, task lists, lengths, directories, and backend requirements live in separate configuration files. A new backend must be checked against its own current API; the repository's older launch snippets are not a promise that every current server version accepts them. [Pinned README](https://github.com/NVIDIA/RULER/blob/c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a/README.md), [checked runner](https://github.com/NVIDIA/RULER/blob/c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a/scripts/run.sh).

LongBench's repository now contains v2 at its root and the original v1 under `LongBench/`. They have different tasks and protocols; a v1 score and v2 score are not successive measurements of the same test. In v2, published context lengths are described in **words**, which are not your tokenizer's tokens. Its named Hugging Face split `train` is the distributed evaluation dataset; that label is not permission to train on it and then claim held-out performance. [Pinned benchmark description](https://github.com/THUDM/LongBench/blob/2e00731f8d0bff23dc4325161044d0ed8af94c1e/README.md).

A serving endpoint that generates text may lack the log-probability interface needed by a likelihood-scored task. Verify chat formatting, raw completion behavior, tokenizer identity, stop handling, and score support independently. The checked LM Evaluation Harness interface uses commands such as `lm-eval run`; task names and backend parameters remain version-specific. Use its installed help and validate task configurations rather than assuming an old command or generic endpoint is compatible. [Pinned interface](https://github.com/EleutherAI/lm-evaluation-harness/blob/d6de81643928d653435c431bae19945d41d32520/docs/interface.md).

## 4. Make the comparison fair and inspectable

Freeze the model/tokenizer revisions, prompts, fixture IDs, output budget, sampling configuration, and scorer before comparing candidates. Record truncation, timeouts, invalid outputs, and missing requests as outcomes under a stated rule; silently dropping failures inflates accuracy. For code tasks, define the test environment and resource limits used to execute generated code.

For each length and task family report valid example count, success count, uncertainty, input/output tokens, latency distribution, and peak memory under the same hardware/load. Separate prompt processing from generation and report their units. A shorter generated answer can reduce total latency without making the underlying implementation faster.

Use a confidence method that matches the sampling unit. With several questions from one document, resample whole documents for a cluster bootstrap, preserving paired candidate results. Repeated decoding seeds for one question are not new independent documents. Very small groups give weak uncertainty estimates; show raw counts instead of hiding them behind a narrow-looking interval. Decide the acceptable regression and sample size from the application before inspecting results.

**Worked paired example:** among 100 shared cases, the baseline passes 80 and the candidate passes 82. If the candidate fixes 12 baseline failures but breaks 10 baseline successes, the net gain is two cases, yet ten previously working cases regressed. Report both transitions. Two aggregate percentages conceal that information.

## Exercise and answer checks

1. A prompt exceeds the configured input limit and the server drops its beginning. Can a retrieval failure measure positional encoding quality? **Answer:** not cleanly; first establish that the evidence reached the model intact.
2. Every generated answer is wrong, but a scorer searches `prompt + answer` for the key's value. What result can appear? **Answer:** 100% retrieval because every prompt already contains it.
3. A new context configuration raises long-document scores and lowers short-document scores. What must the report show? **Answer:** both slices, their counts and uncertainty, and the predefined acceptance rule; averaging them without the intended workload weights can conceal a regression.

Submit the token-budget calculation, a leakage-resistant fixture specification, a pinned benchmark/backend plan, and a paired results schema. Mark unexecuted model comparisons as **planned**; the CPU arithmetic verifies accounting, not long-context model quality.

**Continue:** [Module 08 — Combine context and expert parallelism](08-Combining-LongContext-and-MoE.md).
