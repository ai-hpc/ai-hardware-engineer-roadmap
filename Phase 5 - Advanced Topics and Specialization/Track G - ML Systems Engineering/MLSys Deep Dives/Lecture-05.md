# Lecture 05 — Read a model as an executable system

[Previous: state and caches](Lecture-04.md) · [Course index](README.md) · [Next: speculative decoding](Lecture-06.md)

A model name is not an execution specification. A family can include dense and
MoE variants, different attention layouts, and separately converted checkpoints.
Begin by identifying a particular revision and the code that interprets it.
This skill remains useful when the newest model list changes.

**Prerequisites:** Lecture 04's memory terms and the distinction between a model
weight and a request cache. Your artifact is a one-page model record with facts,
derivations and untested assumptions marked separately. Allow 60–90 minutes.

## Identify three artifacts

Record the original model/config revision, the exact converted checkpoint if
used, and the inference runtime revision. A native FP8 release, an NVFP4
conversion and a GGUF conversion are different artifacts even when their names
share a prefix. They can retain different tensors at higher precision, omit an
auxiliary prediction head, or require different kernels.

Next record tokenizer, chat template and processor revisions. Changes here can
alter the actual input tokens, image patches or stopping behavior. Identical
text prompts are not enough to establish identical model inputs across stacks.

The [Part 1 CPU inference lab](../AI%20Inference%20Engineer%202026/Part%201%20-%20Fundamentals/labs/README.md)
contains pinned config snapshots and executable shape/cache calculations. Use
those files before downloading any weights. A config field such as
`num_attention_heads` must be interpreted in context: nested text/vision configs,
explicit head dimensions, linear-attention head counts and per-layer schedules
can all matter.

## Separate capacity, active work and transferred bytes

An MoE layer stores many experts but selects a subset for each token. For a
fully resident implementation, all expert weights need allocated storage across
the selected devices. Offload can move some storage elsewhere at the cost of
transfers and scheduling constraints. Expert parallelism is one placement choice;
MoE does not inherently require a multi-GPU all-to-all when all experts are local.

Consider 288 routed experts, one shared expert and eight routed selections per
token, as in the pinned GLM case study. A batch of 32 tokens invokes 32×9=288
expert applications. With disjoint routed selections, it touches at most
`32×8+1=257` distinct expert weight sets. With identical routed selections, it
touches nine. Arithmetic can stay the same while potential weight reuse differs.

Actual traffic depends on batching, layout, cache reuse and rereads. Dense
batch-one decode can be bandwidth-bound; MoE can be compute-, memory- or
communication-limited. Architecture labels do not substitute for a profile.
Likewise, “active parameters” is a useful summary rather than an exact FLOP
count: embeddings, attention, gates and data-dependent paths need separate care.

## A current worked model record

The [GLM-5.3-Flash source record and CPU lab](../GLM-5.3-Flash%20Architecture%20Mastery/CPU-Hybrid-State-Lab.md)
pin the official config separately from a converted deployment. Its text stack
has 45 layers: 34 KDA and 11 sparse MLA layers. The first three feed-forward
layers are dense, followed by 42 MoE layers. These are independent schedules;
“attention type” and “feed-forward type” are not one label per model.

From this record you can derive some persistent-state terms. You cannot infer
full latency, quantized quality or the maximum supported context of a particular
runtime from it alone. The configured maximum position count is an architectural
setting; memory capacity and correctness at that length need their own checks.

Use the [DeepSeek release-verification exercise](../AI%20Inference%20Engineer%202026/Part%205%20-%20DeepSeek%20V4.1%20Flash%20NVFP4/Release-Update-2026-10-18.md)
as a second example of why checkpoint revisions and metadata must be revisited.
A new file index or quantization policy can change the execution contract without
changing the course topic.

## Build a decision table

| Question | Evidence to record | What remains a measurement |
|---|---|---|
| Which operations execute? | Layer schedules, operator code, tensor shapes | Actual kernel dispatch and fallback paths |
| What must be stored? | Weight formats, scale metadata, cache representation | Allocated/reserved peak memory |
| How does state grow? | Attention/recurrent layer counts and cache shapes | Paging, replication and concurrency overhead |
| Can it draft tokens? | Auxiliary head or matching external drafter | Acceptance, verification cost and net speedup |
| Can it process images? | Processor, encoder, merge rules | Input-dependent preprocessing and vision latency |
| Is it suitable for the task? | Evaluation protocol and relevant model evidence | Held-out task outcomes on the deployed artifact |

## Guided variation

Take the same model and change only its checkpoint representation. Predict which
rows change. Then inspect the conversion metadata: do scales, unchanged tensors,
clamps or auxiliary heads differ? Write a quality check before timing it.

**Self-check:** a converted checkpoint is smaller and serves a plausible answer.
Is it a valid substitute for the original?

<details markdown="1">
<summary>Answer</summary>

That establishes neither numerical agreement nor task quality. Check the actual
input tokens, layer outputs or probability metrics where accessible, then run a
held-out task evaluation with fixed generation rules. Smaller storage can also
be slower if unpacking, fallback kernels or transfer costs dominate.

</details>

**Exit check:** every number in your record has a config field, derivation or
measurement behind it. You can identify what a new model revision would force
you to recheck. Sources and linked artifacts reviewed October 18, 2026; no
full-model deployment is claimed by this metadata exercise.
