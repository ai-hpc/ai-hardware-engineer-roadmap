# Module 10 — Turn the experiment into a defensible capability claim

A trained model and a passing test suite are useful artifacts. The claim you make about them must match the tasks, data, runtime, and uncertainty that were actually evaluated. “General purpose” is not a property established by completing a fixed number of benchmark checkboxes.

**Before this lesson:** [data provenance](06-Adaptive-Data-Pipelines.md), [evaluation](07-Long-Context-Evaluation.md), and [recovery](09-Distributed-Training-Infrastructure.md). **Outcome:** produce a reproducible comparison and state what the result supports. Two completion levels below let you finish useful work on CPU and keep the larger model study clearly separate.

## 1. Complete the CPU evidence route first

Run the [CPU training and dispatch lab](CPU-Training-Lab.md). It includes real optimization of a small top-k MoE, held-out synthetic regression, router/gradient checks, model-weight reload, and a separate two-process dispatch/combine fixture. Use its commands, dependency lock, retained results, and explicit scope.

These artifacts support different statements:

| Artifact | Supported statement | Claim still requiring separate evidence |
|---|---|---|
| Ledger and its tests | Listed FLOP/tensor/mesh arithmetic satisfies the stated formulas | Actual GPU memory peak or throughput |
| Sparse MoE versus dense oracle | Specified forward/backward calculations agree on the tested inputs | Correctness of every router policy or distributed model |
| Training/validation regression | This small network learns the included synthetic mapping | Language, long-context, or robot-control competence |
| Weight save/reload equality | Stored model weights reproduce the specified probe output | Optimizer/data resume after interruption |
| Two-process Gloo dispatch | The separate affine-expert fixture returns the expected results | Distributed training of the learned network or NCCL performance |

In your report, compare the auxiliary-loss settings using both held-out error and routing statistics. A more even assignment histogram is not automatically better task performance. A capacity-limited run can lose accuracy or change behavior with the group of tokens admitted together. State the capacity policy, batch size, dropped-assignment fraction, and fraction of tokens receiving no expert output.

Choose one independent variation with a prediction: a new held-out random seed, a changed capacity factor, or a changed auxiliary coefficient. Preserve the original result and explain whether the variation supports your prediction. Repeatedly selecting a configuration on the same validation set makes that set development evidence; reserve a new evaluation set for a final selected-model claim.

## 2. Define the larger task before choosing a baseline

For a robotics-manual assistant, a testable claim might be: “Under this prompt and document budget, the candidate answers revision-specific maintenance questions with a cited supporting passage and abstains when evidence is absent.” That claim names an input, a behavior, and a failure mode. It does not claim that the model can operate a robot.

Choose a baseline with a retrievable model revision, compatible license, supported tokenizer/runtime, and a feasible memory budget. Compare the same task inputs and output budget. Record any differences in retrieval, prompting, tool access, sampling, quantization, and compute. A stronger retrieval pipeline or a longer reasoning budget can explain a result even when the underlying model has not improved.

Use tasks that probe the intended behavior: direct lookup from the correct revision, reconciliation of two sections, conflicting or absent evidence, and an independently checked computation. Keep the answer key and scorer separate from the candidate's prompt. If a task executes generated code, retain the tests, execution environment, resource limits, and handling of errors/timeouts.

## 3. Be precise about “unseen”

A private source-family split helps prevent leakage within your own dataset. It cannot prove that an external base model never saw related material. A document published after a stated training cutoff gives useful provenance evidence, but the model may still have learned the same task pattern or underlying fact elsewhere.

Exact matching, n-gram matching, and approximate duplicate detection identify different kinds of overlap. Keep the matching procedure, thresholds, inspected examples, and unresolved cases. There is no universal n-gram size or similarity cutoff that proves an evaluation clean. A public benchmark may also have influenced model selection without appearing in training text.

Report the original prescribed benchmark score and any separately labeled overlap-audited subset. Explain exclusions before interpreting the subset; removing difficult or low-scoring cases after seeing results changes the question and can inflate the score. A newly written paraphrase of a known benchmark answer is not automatically independent evaluation data.

For broader context, use the [pinned benchmark and backend distinctions in Module 07](07-Long-Context-Evaluation.md). Do not invent a decontamination flag or assume a benchmark's name guarantees that its data is uncontaminated.

## 4. Separate automatic checks from subjective judgments

Prefer task-specific checks where available: exact values with units, executable tests, or citation spans that actually support the answer. These checks still have a defined coverage boundary. A number can be correct while referring to the wrong robot revision; a code submission can pass a narrow test suite while failing other inputs.

For open-ended answers, define a rubric, blind the candidate identity, randomize answer order, and inspect order swaps. A language-model judge is one measurement instrument with possible style, length, and model-family preferences. Keep judge revision, prompts, scoring rules, and disagreement examples. Use qualified human review on a defined sample, especially for errors the automatic rubric cannot resolve; agreement between judges alone does not establish correctness.

If a model generates robot actions, add action units/frame, normalization, timing, and paired closed-loop outcomes. Text-answer quality or offline action similarity cannot replace that evidence. The [action-parity bridge](../../../../Track%20D%20-%20Robotics/VLA%20Optimization%20and%20Action-Parity%20Harness/CPU-Action-Parity-Lab.md) shows an actual learned policy whose identical weights can produce different outcomes when action delivery changes. Its small cart experiment is explicitly limited to that environment.

## 5. Predefine the decision and retain regressions

Choose success criteria from the application before comparing candidates. Include the key error types, acceptable regression, latency/memory budget, and how invalid or absent outputs count. A fixed 1% tolerance or a fixed number of prompts is not suitable for every task.

Use paired cases and report which baseline successes were lost and which failures were fixed, alongside uncertainty at the appropriate source-group level. For example, 200 test cases may contain only ten independent manuals. Treating every question as an independent manual exaggerates how much evidence you have about new source families.

If you compare many variants and keep the best, record that search. Use development data for selection and a fresh final comparison; do not present the selected maximum as though it came from one prespecified trial. Keep a baseline when evidence is inconclusive rather than turning a small apparent gain into a broad capability claim.

## 6. Assemble the capstone

Create one report with the following fields. This is a reporting schema, not a command that launches a training framework:

```text
claim and intended input distribution:
model/tokenizer/code/data revisions:
training objective, valid-token budget, and selected checkpoint:
source-family split and overlap audit:
baseline, prompts, runtime, hardware, and resource budget:
paired evaluation cases, scorer, failures, and uncertainty method:
results by task family and context length:
latency/memory and training/recovery evidence:
regressions, excluded cases, and unexecuted work:
conclusion limited to the measured conditions:
```

**CPU completion:** include the lab commands/results, one independent variation, a hand-checked mesh/payload calculation, and an evaluation design. State plainly that no foundation language model was trained or evaluated by that route.

**Model-study completion:** additionally include the real pinned model artifacts, data lineage, training/resume evidence, actual benchmark outputs, and the prespecified paired comparison under the target runtime. Mark missing pieces as remaining work; a plausible launch template is not execution evidence.

## Final exercise

A candidate's retrieval accuracy improves, but it uses twice the output budget and fails more absent-evidence cases. Can you call it a better manual assistant? **Answer:** the result is a tradeoff until evaluated against the predefined application criteria. Repeat with matched budgets, report the abstention regression, and keep the conclusion specific. That reasoning is the capstone skill: connecting a system change to evidence that actually measures its intended benefit.

**Return:** [course overview](README.md) or [CPU training lab](CPU-Training-Lab.md).
