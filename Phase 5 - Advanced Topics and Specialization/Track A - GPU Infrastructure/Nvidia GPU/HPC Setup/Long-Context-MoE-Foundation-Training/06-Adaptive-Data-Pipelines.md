# Module 06 — Build a data curriculum you can audit

A data curriculum changes which examples a model sees as training progresses. The useful question is whether a specific change improves an agreed task at an affordable cost. There is no universal percentage of books, code, synthetic text, or long documents that makes a mixture optimal.

**Before this lesson:** understand tokens, training/validation/test splits, and the routing behavior in [Module 05](05-MoE-Systems-Infrastructure.md). **Outcome:** produce a versioned data manifest, calculate the effective token mixture, and propose one controlled curriculum comparison. This lesson contains worked arithmetic and a standard-library exercise. It does not download a corpus or launch foundation-model training.

## 1. Split the source before making training windows

Suppose a robotics manual has 20 chapters. You create overlapping 4K-token windows, then randomly allocate those windows to training and evaluation. Adjacent windows share text; different chapters may repeat the same diagrams, instructions, and answers. The evaluation now partly measures recognition of training content.

Use a **source family** as the grouping unit: a manual and its revisions, a code repository and forks, or an original document and its translations/paraphrases. Assign groups to training, development, or final test before chunking. A longer window derived from a training manual remains training data at the next curriculum stage.

Maintain a global duplicate/overlap index across **all stages and all splits**. An exact content hash detects exact copies under your chosen normalization; approximate matching can propose near-duplicate groups for review. Neither establishes that a public benchmark was absent from an unknown pretrained model's history. Keep both the original content hash and the normalized matching hash so aggressive normalization does not erase provenance.

| Record | What it lets you check |
|---|---|
| Source ID, source-family ID, revision and content hash | Where a window came from and whether related versions cross splits |
| Split, chunk offsets, tokenizer revision, valid token count | Whether the same source was repackaged or counted differently |
| License/access record, extraction/filter revision | Whether this particular use and transformation are documented |
| Stage, selection probability, actual presentations | How often the model consumed the example |
| Parent IDs and generator/checker revisions for synthetic text | Which original problems and generated variants share lineage |

A duplicate policy may deliberately retain a full manual instead of several overlapping fragments. Record that decision globally; do not keep the same content in evaluation merely because it appears in a different training stage. Audit filters by language and source type: removing short lines can remove code, while removing uncommon words can remove technical terminology.

## 2. Sampling examples is different from sampling tokens

Consider two sources. Manual examples average 8,000 valid training tokens; code examples average 2,000. Selecting each source with probability 0.5 produces an expected manual token share of:

`(0.5 × 8,000) / (0.5 × 8,000 + 0.5 × 2,000) = 0.8`.

For equal expected token shares, choose example probabilities inversely proportional to mean valid length: manuals 0.2 and code 0.8. This calculation assumes stable source lengths and the stated sampling process. Padding, truncation, completion-only loss masks, exhausted sources, and packing can change the realized shares. Log **consumed valid loss tokens**, source IDs, and repetitions instead of inferring exposure from a configuration file.

Run this from any working directory; it uses only Python's standard library:

```bash
python3 - <<'PYCODE'
lengths = {'manual': 8000, 'code': 2000}
for probabilities in ({'manual': .5, 'code': .5},
                      {'manual': .2, 'code': .8}):
    exposure = {k: probabilities[k] * lengths[k] for k in lengths}
    total = sum(exposure.values())
    print({k: round(v / total, 3) for k, v in exposure.items()})
PYCODE
```

Expected token shares are `{'manual': 0.8, 'code': 0.2}` and then `{'manual': 0.5, 'code': 0.5}`. These are calculated expectations, not measurements from a loaded corpus.

If you use Hugging Face Datasets, its current `interleave_datasets` chooses sources using example probabilities. `first_exhausted` stops early; `all_exhausted` can repeat shorter sources; `all_exhausted_without_replacement` avoids repeated examples under that strategy. Streaming workers apply exhaustion behavior independently. Pin the installed package and record the strategy; a seed alone is not a complete replay contract. [Official interleaving API](https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.interleave_datasets).

## 3. Packing changes the learning problem

Packing combines shorter examples into one sequence to reduce padding. Decide whether examples can attend across their boundaries and whether position IDs reset. These choices affect the attention mask, loss mask, and what context predicts each target. Separating documents with an end token does not by itself block attention across them.

For a packed sequence containing a 300-token document, a 500-token document, and 224 padding tokens, the count of predicted loss tokens is determined by the target/label mask. It is **not automatically 1,024**, and may be less than 800 because first-token or prompt targets are excluded. Use the implementation's actual nonignored label count. This same denominator must reach distributed gradient reduction and perplexity in [Module 07](07-Long-Context-Evaluation.md).

Retain enough source offsets to reconstruct a batch. A resume checkpoint needs the data cursor, shuffle state, packing buffer, and any adaptive mixture state, or an equivalent deterministic replay mechanism. Without them, a restarted run can repeat easy examples and omit later ones while its loss still looks plausible.

## 4. Make one curriculum change measurable

Start from a fixed mixture and a small pilot budget. Compare it with one intervention—for example, more complete manuals with cross-chapter questions—while matching total valid training tokens and recording compute time. Changing sequence length changes attention cost, so equal tokens do not imply equal GPU hours. Use the [cost ledger](labs/ledger.py) to explain that difference before choosing a budget.

Choose a few development tasks connected to the intended use: retrieving a torque limit from the correct manual revision, joining information from two chapters, or generating code that passes held-out tests. Keep a final test set sealed while selecting mixture, model, prompt, and checkpoint. Once you use a test failure to create training data, that case has become development evidence; replace the final test protocol with a new untouched holdout when reporting a new claim.

Change context length and positional encoding only with the model's supported configuration and a separate comparison. A larger RoPE base or scaling factor is not a universal curriculum schedule. Verify short-context behavior as well as the new length, and retain the exact checkpoint/configuration used for both.

## 5. Synthetic examples need a defined checker

For a generated robotics calculation, a unit-aware reference calculation can check the numeric answer. For generated code, tests check only the cases and environment they exercise. A language-model judge can triage or score a stated rubric, but deterministic decoding does not turn the judge into an oracle. Record the prompt, model revision, checker, rejected examples, and manual audit sample.

Synthetic variants inherit their source family's split. Generating ten paraphrases of a held-out problem and putting them in training contaminates that evaluation even when their exact text differs. Measure whether adding accepted synthetic examples improves development performance over a budget-matched baseline; do not assume a fixed synthetic percentage is safe or helpful.

## Exercise and worked check

You have 100 manual examples of 8,000 tokens and 400 code examples of 2,000 tokens. A single pass over both consumes 800,000 tokens from each source, so its token mixture is 50/50. Sampling sources 50/50 with replacement has a different expected mixture, 80/20. Explain both results without changing the dataset.

Then design a 4K-to-16K continuation experiment. Submit the source-family split rule, duplicate audit, tokenizer/mask contract, fixed development tasks, realized token counts, and one comparison with a fixed-mixture baseline. Do not invent an expected quality improvement. Completion means you can explain exactly what changed and which evidence would reject your hypothesis.

**Continue:** [Module 07 — Long-context evaluation](07-Long-Context-Evaluation.md).
