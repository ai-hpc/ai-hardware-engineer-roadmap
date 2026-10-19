# Lecture 06 — Audit an official MTP assistant and finish the deployment dossier

**Input:** the target-only baseline and Lecture 04's accounting. **Output:** a supported paired-assistant experiment or an explicit compatibility rejection. [Previous](Lecture-05.md) · [Course map](README.md).

## 1. Distinguish the target from its assistant

`google/gemma-4-E2B-it` is a target model. `google/gemma-4-E2B-it-assistant` is the separately published assistant checkpoint. Do not call the E2B target itself “the MTP drafter,” or treat it as a generic drop-in assistant for every larger Gemma model. Start from Google's [assistant repository](https://huggingface.co/google/gemma-4-E2B-it-assistant) and its files, then verify the supported target pairing in the selected runtime.

“Multi-token prediction” can refer to training objectives, an assistant architecture, or a speculative execution method. Those are related but not identical. A checkpoint name does not reveal its layer count, hidden-state inputs, cache sharing, or serving API.

## 2. Build a pairing audit

| Field | What to save |
|---|---|
| Target | Repository, immutable revision, config and weight hashes |
| Assistant | Repository, immutable revision, config and weight hashes |
| Token interface | Vocabulary, special tokens, chat template, preprocessing |
| Execution interface | Target-state dependencies, draft outputs, cache/state ownership |
| Runtime | Commit, implementation path, supported backend and modality |
| Acceptance | Algorithm, proposed/accepted/committed counts, correction policy |

Read the assistant's configuration and the actual runtime implementation together. An example command from a different speculative method, such as an arbitrary external draft model, does not prove this assistant is supported. Do not invent a `litert-lm` Python constructor or a vLLM option based on the checkpoint name.

## 3. Run a staged experiment

First load the target and reproduce Lecture 03's smoke result. Then verify that the runtime selects the assistant implementation rather than ignoring the configuration. Start with a small fixed prompt and bounded draft length; inspect proposal and verification logs. Keep target-only as a control throughout.

Repeat the exact workload suite from Lecture 04, including an unhelpful draft case. Record startup time and both peak and steady memory: an assistant may reduce decode time but worsen cold start or exceed the embedded budget.

## 4. Translate the result into accelerator requirements

If drafting helps, identify why: reused target state, smaller projection work, more efficient verification batches, or fewer expensive target steps. If it loses, identify whether acceptance, cache duplication, launch overhead, or bandwidth dominates. These become concrete design choices:

- Can SRAM retain the reused state without displacing the target's hot working set?
- Does the compiler express verification shapes and dynamic accepted-prefix lengths?
- Can the runtime cancel rejected candidate work without reusing a live buffer?
- Does the controller still receive a result within the observation validity horizon?

An unsupported assistant is a useful negative finding. Finish the target-only course path, record the missing runtime/backend capability, and avoid filling a timing table with estimates presented as measurements.

## Capstone acceptance

Deliver one dossier containing model/assistant identity, format, runtime/BSP compatibility, quality cases, memory accounting, raw timing, and the observation/result contract. Every performance number must identify hardware and workload; every unsupported or unrun configuration must say so.

End with one decision: deploy the baseline, deploy the measured optimization, or reject this hardware/model pairing. Justify it using task quality, sustained latency, memory, and energy evidence. Feed the largest unresolved bottleneck into the custom physical-AI accelerator specification rather than choosing a chip architecture from the model's marketing size.

## Supplied metadata audit

Run the [CPU exercise](README.md#start-with-the-supplied-cpu-exercises) to inspect the pinned assistant configuration. It declares `Gemma4AssistantForCausalLM`, four layers (three sliding, one full), hidden width 256, and `backbone_hidden_size=1536`. The target has hidden width 1,536. These are observed configuration facts, not an executed assistant/target inference result. The runtime's state-sharing, position, acceptance and processor behavior must still be validated before making a speed or quality claim.
