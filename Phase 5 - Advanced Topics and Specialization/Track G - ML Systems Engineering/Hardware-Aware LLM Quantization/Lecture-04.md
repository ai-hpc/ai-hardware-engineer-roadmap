# Module 04 — Model Anatomy: Count the Stored Bytes, Then Model the Traffic

**Collection:** [Hardware-Aware LLM Quantization](README.md) | **Previous:** [Previous lesson](Lecture-03.md) | **Next:** [Next lesson](Lecture-05.md)

Your task is to separate three quantities: serialized tensor bytes, resident device
memory, and bytes transferred during an operation. They answer different questions.
A model index can establish the first. Runtime allocation measurements establish
the second. Execution traces and hardware counters are needed for the third.

## 1. Start from the actual architecture

Read [the pinned source evidence](Source-Evidence.md) before interpreting a model
name. The official Qwen3.8-27B configuration specifies hidden size **5120**, vocabulary
**248320**, and **64** language layers: **16 full-attention** and **48 linear-attention**.
It is a dense-parameter hybrid model; “dense” does not mean every layer stores a
full sequence of K/V vectors. The community NVFP4 target retains this architecture
but removes its built-in MTP head. DSpark is a **separate five-layer drafter**.

The BF16 input embedding matrix would contain:

```text
248320 × 5120 = 1,271,398,400 elements
× 2 bytes = 2,542,796,800 bytes = 2.543 GB = 2.368 GiB
one gathered row = 5120 × 2 = 10,240 bytes
```

Its dimensions cannot be inferred uniquely from its byte count. For example,
`155200 × 8192` gives the same element count. An earlier version of this lesson
selected the wrong factorization by assuming a familiar tokenizer size. Inspect
`vocab_size`, `hidden_size`, `tie_word_embeddings`, the tensor shape and the loader.
Equal input/output matrix sizes also do not prove that weights are untied.

**Self-check:** can an API's total model size reveal the number of attention heads?
**Answer:** no; many architectures and precision mixtures fit one scalar size.
Record unknowns instead of manufacturing a configuration.

## 2. Build a storage ledger from real serialized tensors

For ordinary safetensors entries, `data_offsets[1] - data_offsets[0]` is the payload
length. Sum each stored tensor once. Packed FP4 weights may appear as byte arrays;
the dtype of that array does not mean each logical weight consumes one byte.
Scale tensors, padding, auxiliary tensors and unquantized exclusions are separate.
A dtype-to-bytes table with an unknown-type default of 2 silently miscounts them.

Use the index to enumerate **all shards**, then their headers to count actual
payload offsets. Do not load model weights simply to count them. The supplied
[CPU lab](CPU-Quantization-Lab.md) checks a compact metadata inventory offline;
it deliberately does not claim to have inspected every safetensors payload.

For the pinned community target, `model.safetensors.index.json` reports
**17,915,815,528 tensor bytes across two shards**. That is **17.916 GB / 16.685 GiB**.
It excludes file headers and is not a measured GPU allocation. Its key list includes
`lm_head.weight_scale` and `lm_head.weight_scale_2`, and no MTP tensor names.
A config flag plus names is evidence of intended export structure, not proof that
a particular engine correctly executes it.

A useful ledger has these columns:

| Group | Stored bytes | Loaded bytes | Invocations | Expected access | Evidence |
|---|---:|---:|---:|---|---|
| Language projections | from headers | measured | each relevant layer | dense or selected experts | shapes, trace |
| Input embedding | from headers | measured | per input token | gathered rows | tied/shared storage checked |
| Output projection | from headers | measured | requested logits | vocabulary projection | pruning/fusion checked |
| Vision encoder | from headers | measured | image/video encoding | conditional | request type |
| Drafter | separate artifact | measured | speculative steps | own weights and state | draft configuration |
| KV/recurrent state | absent from weights | measured | request lifetime | architecture dependent | Module 09 |

A tensor's traffic class is a property of the **execution path**. Shared tensors,
MoE routing, prefix reuse, offloading and repeated kernels prevent an exclusive
name-only classification from being a complete analyzer. Unknown tensor names
should require review, not silently be classified as always streamed.

## 3. A worked historical estimate, with its limits

The community target card describes an earlier **18.80 GiB**, **81.6 tok/s** build.
These are publisher-reported historical numbers, not measurements from this course
or the current 17.916 GB artifact. Their branch identities are recorded in the
source evidence when available; copying today's model ID is not reproducing that run.

Suppose, for an **illustrative inventory**, resident tensors total 20.19 GB,
including a 2.54 GB gathered embedding, an inactive 0.92 GB vision tower and an
inactive 0.85 GB draft component. Subtracting them gives about **15.88 GB of
candidate streamed weights**. Then:

```text
15.88 GB/token × 81.6 token/s = 1295.8 GB/s
1295.8 / 1792 = 72.3% of the RTX 5090's advertised bandwidth
```

This is **estimated required weight bytes per second**, not measured DRAM
utilization. It omits activations, state, rereads and cache effects, and its inputs
may not describe the historical run exactly. It cannot prove that a 27% speedup
is available or that launch overhead is the bottleneck. A low estimate is a
reason to profile and refine the model.

## 4. Rank candidates under explicit assumptions

With `B=15.88 GB`, reducing a 2.54 GB output projection to 1.27 GB predicts
`15.88 / 14.61 = 1.087×` speed **if achieved bandwidth and other costs stay fixed**.
Reducing it to `2.54 × 0.5625/2 ≈ 0.714 GB` predicts `1.130×`.
These are sensitivity calculations. The current community target already has a
quantized output head, so neither is a fresh optimization for that artifact.

Quantizing embeddings removes far fewer directly accessed bytes per token than
quantizing an equally sized dense output head. It can nevertheless improve
capacity, avoid offload, change cache behavior or enable a larger batch. Thus it
is a useful control, not an operation guaranteed to change throughput by zero.

## 5. Change the ledger when the workload changes

For speculation, count **one cycle** first:

```text
B_cycle ≈ B_target_verify(K+1) + sum(B_draft_step_j) + state/temporary traffic
bytes per emitted token ≈ B_cycle / mean_tokens_emitted_per_cycle
```

A DSpark drafter is not the target's MTP head; their weights, state and invocation
costs differ. Target verification can reuse weights across proposed positions,
but its latency is not guaranteed to equal a one-token target step. Never multiply
the size of an inactive MTP head by draft depth to model the external drafter.

With images, account for image encoding and projected image tokens in prefill,
then the resulting language-model state during decode. Streaming video can invoke
the encoder again; “vision never contributes to decode latency” is too broad.

## 6. Independent exercise and completion

Run the offline inventory lab and submit a storage/traffic ledger with an evidence
column. Then change only `lm_head` from BF16 to the reference FP4 format, including
scale and row-padding bytes. State which numbers are exact payload counts and which
are traffic assumptions. For a device run, retain peak allocated/reserved memory,
state lengths, profiler counters and latency separately.

**Answer guidance:** storage counts must reconcile exactly to the selected files;
runtime memory also includes workspace, allocator behavior and state. A discrepancy
is not automatically a loader bug. A prediction outside a timing interval calls
for an explanation of missing traffic or compute, not a fabricated efficiency factor.
