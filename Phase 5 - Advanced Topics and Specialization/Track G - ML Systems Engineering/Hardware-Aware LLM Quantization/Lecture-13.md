# Module 13 — Capstone: Build a Quantization Policy Evaluator

**Collection:** [Hardware-Aware LLM Quantization](README.md) | **Previous:** [Previous lesson](Lecture-12.md) | **Next:** [Next lesson](README.md)

Build a small reusable evaluator that can explain why it selected a policy and
which evidence could invalidate it. The capstone's former “TurboQuant” label has
been removed to avoid confusing this course project with separately published
quantization research. It is not an implementation or reproduction of such work.

## 1. Choose the scope you can execute

The CPU route uses the trained nonlinear robot-stop classifier and reference
block-FP4 packer in [CPU Quantization Lab](CPU-Quantization-Lab.md). It demonstrates
calibration, payload accounting, evaluation and distribution shift with real
PyTorch. It does not execute NVIDIA FP4 instructions or measure LLM accuracy.

The optional GPU route uses a licensed checkpoint and a pinned engine with an
explicitly supported quantization path. Reproduce a baseline before optimization;
community model-card performance is an attributed reference, not your baseline.

## 2. The five-stage artifact

| Stage | Input | Output | Failure to test |
|---|---|---|---|
| Inventory | model/config/shard metadata | exact payloads, assumed traffic, state budget | missing scale tensor, wrong architecture |
| Calibration | training/calibration IDs and supported policies | candidate scales and reconstruction/output error | zero block, tail padding, held-out leakage |
| Selection | development results and constraints | feasible allocation with reason | no feasible policy, incompatible kernel |
| Build | chosen policy | serialized artifact and manifest | scales/layout/reload mismatch |
| Evaluate | immutable test and shifted inputs | task, fidelity, latency/memory report | changed threshold, stale cache, biased sampling |

Do not silently change the model or deployment objective between stages. A packed
CPU reference artifact is its own format, not a TensorRT/SGLang checkpoint.

## 3. Required CPU ablations

1. Untouched trained FP32 reference.
2. Absmax FP4 reference weights.
3. Weight-MSE-selected clipping.
4. Calibration-output-MSE-selected clipping.
5. A deliberately mismatched input shift, applied after policy selection.

Keep the original reference and threshold fixed. Report payload including scales,
row padding and retained biases/normalization, then task metrics and output error.
Choose on development/calibration data; the held-out test cannot select a winner.
A smaller payload with worse missed-stop cost may fail the declared gate.

Add the exact categorical speculative-sampling proof from Module 08. State why
its correctness says nothing about the robot task, and why drafter acceptance
says nothing by itself about an LLM target's correctness. These are separate
contracts connected by disciplined evaluation.

## 4. Optional device extension

Confirm support for the model architecture, scale scheme, activation path,
accumulator, GPU and compiler. Record measured memory rather than assuming the
serialized artifact equals GPU residency. Begin with a feasible short context;
calculate long-context state before trying it. For speculation, keep target and
drafter ledgers separate and measure complete cycle times across K.

Choose a bounded ablation grid driven by the baseline. The current case already
quantizes `lm_head` and removes MTP, so “quantize the BF16 output head” and “remove
MTP” are historical changes, not new optimizations of today's pinned artifact.
Embedding/vision changes are useful controls for direct traffic but can affect
capacity and caching; a nonzero effect does not automatically invalidate the rig.

## 5. Success and handoff

The minimum submission contains runnable commands, source/dependency hashes,
calibration and evaluation IDs, a payload ledger, raw results and a one-page
interpretation. It passes when:

- Exact-format, zero/tail/rounding and reload checks pass.
- Candidate selection does not read held-out labels or shifted test outcomes.
- The selected policy is evaluated against explicit task/fidelity constraints.
- Storage estimates, timing observations and unexecuted device claims are labeled.
- At least one adverse case or unsupported assumption is explained.

There is no required 224/280 token/s target or universal 10% prediction accuracy.
If every quantizer fails the quality budget, retain the reference and document
why. For a stronger project, transfer the evaluator to a second architecture,
change the constraints and explain how the chosen policy changes.

**Final question:** which result would persuade you to reverse your decision?
A useful answer names a new workload slice, a runtime/kernel measurement, or a
failed correctness assumption that the current evidence does not cover.
